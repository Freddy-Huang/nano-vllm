"""定义一条生成请求在 Nano-vLLM 内部的状态和 token 数据。

用户传入的一条 prompt 会被 ``LLMEngine.add_request()`` 包装成 ``Sequence``。
之后 Scheduler 不直接操作原始字符串，而是通过 Sequence 记录：

- 请求目前处于等待、运行还是完成状态；
- prompt 与已经生成的 token；
- 有多少 token 已进入 KV Cache、本轮还要计算多少 token；
- 该请求占用了哪些物理 KV Cache block；
- 采样温度和停止条件。

可以把 Sequence 理解成一张会随着每轮 Prefill/Decode 持续更新的“请求状态表”。
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """一条请求在 Scheduler 中可能处于的三种状态。"""

    # 刚加入引擎、正在进行 Prefill，或者因 KV Cache 不足而被抢占后等待重算。
    WAITING = auto()

    # Prompt 剩余部分已完整进入本轮 Prefill 调度；该轮结束后可以逐 token Decode。
    RUNNING = auto()

    # 已生成 EOS 或达到 max_tokens，不会再被调度，KV Cache 也已经释放。
    FINISHED = auto()


class Sequence:
    """保存一条 prompt 从进入队列到完成生成所需的全部请求状态。"""

    # 每个逻辑 KV Cache block 包含的 token 数。LLMEngine 初始化时会把它改成
    # config.kvcache_block_size，保证 Sequence 与 BlockManager 使用同一个值。
    block_size = 256

    # itertools.count() 会依次产生 0、1、2……，为当前 Python 进程中的每条请求
    # 分配唯一且递增的 seq_id。generate() 最后利用它恢复请求的输入顺序。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """用 prompt token 和采样配置创建一条处于 WAITING 状态的新请求。

        ``token_ids`` 必须至少包含一个 token，因为初始化时需要读取最后一个
        token；当前轻量实现没有额外处理空 prompt。
        """

        # next(counter) 为请求分配不会与之前请求重复的编号。
        self.seq_id = next(Sequence.counter)

        # 新请求需要先完成 Prompt Prefill，因此从 WAITING 状态开始。
        self.status = SequenceStatus.WAITING

        # 复制列表，避免后续 append_token() 意外修改调用者传入的原始列表。
        # 列表元素是整数，浅拷贝已经足够。
        self.token_ids = copy(token_ids)

        # Decode 每轮只输入上一次生成的 token，单独保存可以避免反复索引列表；
        # 在多卡场景中也可以只把这个值发送给其他 TP rank。
        self.last_token = token_ids[-1]

        # 当前序列的 token 总数 = prompt token 数 + 已生成 completion token 数。
        self.num_tokens = len(self.token_ids)

        # 记录最初 prompt 的长度。它保持不变，用于划分 prompt 和 completion。
        self.num_prompt_tokens = len(token_ids)

        # 已经执行过模型计算、其 K/V 已写入 KV Cache 的 token 数。
        # 新生成的 token 只是刚被采样并追加，通常要到下一轮 Decode 后才计入缓存。
        self.num_cached_tokens = 0

        # Scheduler 在每轮开始前设置：Prefill 可能大于 1，Decode 固定为 1；
        # postprocess() 处理完该轮结果后会把它重置为 0。
        self.num_scheduled_tokens = 0

        # 表示该请求当前是否需要走 Prefill 数据传输方式。请求第一次创建以及
        # 被抢占后为 True；进入正常 Decode 后由 Scheduler 改为 False。
        self.is_prefill = True

        # 逻辑 block 序号到物理 KV Cache block ID 的映射。例如 [7, 19] 表示
        # 该序列的第 0 个逻辑块存于物理块 7，第 1 个逻辑块存于物理块 19。
        self.block_table = []

        # 把采样参数复制到 Sequence，之后 Scheduler/ModelRunner 只需传递请求对象。
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """让 ``len(seq)`` 返回当前序列的 token 总数。"""

        return self.num_tokens

    def __getitem__(self, key):
        """让 Sequence 像列表一样支持索引和切片，例如 ``seq[start:end]``。"""

        return self.token_ids[key]

    @property
    def is_finished(self):
        """请求是否已经满足停止条件并完成资源释放。"""

        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已经生成的 token 数，不包含原始 prompt。"""

        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回序列最初的 prompt token 部分。"""

        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回模型新生成的 token；LLMEngine 最终只解码这一部分。"""

        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """返回保存当前全部 token 至少需要多少个固定大小的逻辑 block。

        ``(n + block_size - 1) // block_size`` 是整数的向上取整除法。例如
        block_size=4 时，长度 4 需要 1 块，长度 5 需要 2 块。
        """

        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """返回最后一个逻辑 block 当前装有多少个 token。

        例如 block_size=4、序列长度为 6，则共有 2 块，最后一块装有 2 个 token。
        如果序列长度正好是 block_size 的整数倍，结果为完整的 block_size。
        """

        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个逻辑 block 对应的 token ID 切片。"""

        # 先阻止负索引或超过当前块数的访问。
        assert 0 <= i < self.num_blocks

        # 最后一个 block 可能尚未填满，因此切片长度可能小于 block_size。
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """把本轮采样到的新 token 追加到序列末尾。

        此处不增加 num_cached_tokens：新 token 此刻只是采样结果，它的 K/V 要在
        下一轮作为模型输入完成 Decode 后，才真正写入 KV Cache。
        """

        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """自定义 pickle 内容，减少 rank 0 向其他 TP rank 发送的数据量。

        ModelRunner 使用 pickle 和共享内存把本轮 Sequence 发送给其他 GPU 进程。
        Prefill 需要按区间读取 prompt，因此发送完整 token_ids；Decode 每条请求
        只输入 last_token，因此不必反复发送越来越长的历史 token 列表。
        """

        last_state = self.last_token if not self.is_prefill else self.token_ids

        # 只发送其他 rank 准备模型输入所需的字段。status、采样参数和 seq_id 等
        # 调度状态仍由 rank 0 管理，无需复制到每个计算进程。
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """在其他 TP rank 反序列化 __getstate__() 返回的精简状态。"""

        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state

        # Prefill 的 last_state 是完整列表，需要恢复 token_ids 供切片；Decode 的
        # last_state 是单个 token，只恢复 last_token 即可，token_ids 可以保持为空。
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # Decode 的其他 rank 不负责维护完整输出，也不参与停止条件判断。
            self.token_ids = []
            self.last_token = last_state
