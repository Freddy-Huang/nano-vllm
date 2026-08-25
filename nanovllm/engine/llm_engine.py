"""Nano-vLLM 的顶层推理引擎与同步生成循环。

这个文件主要负责“组织工作”，而不是直接执行 Transformer 数学计算。它把
Tokenizer、Scheduler 和 ModelRunner 串成下面这条主链路：

``prompt``
    -> tokenizer 编码
    -> Sequence 请求对象
    -> Scheduler 选择本轮请求
    -> ModelRunner 执行 Prefill/Decode 并采样
    -> Scheduler 更新请求状态
    -> tokenizer 解码生成结果

初次阅读时，建议按照 ``generate() -> step() -> add_request() -> __init__()``
的顺序追踪，而不是机械地从文件第一行读到最后一行。
"""

import atexit
from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """管理模型资源、请求队列以及完整生成过程的推理引擎。"""

    def __init__(self, model, **kwargs):
        """创建一套可以接收生成请求的 Nano-vLLM 引擎。

        Args:
            model: 本地 Hugging Face 模型目录。
            **kwargs: 传给 Config 的可选引擎参数，例如 enforce_eager、
                tensor_parallel_size 和 max_num_seqs。

        这一阶段会真正加载模型权重并分配 GPU 资源，因此 ``LLM(...)`` 的主要
        初始化耗时发生在这里，而不是发生在 ``from nanovllm import LLM`` 时。
        """

        # dataclasses.fields(Config) 返回 Config 声明的全部字段。这里先取字段名，
        # 再从 kwargs 中筛选有效配置，避免把其他上层参数直接传进 Config。
        # 注意：当前实现会静默忽略名称不属于 Config 的 kwargs，而不会主动报错。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}

        # Config 会检查本地模型目录、读取 config.json，并修正 max_model_len。
        config = Config(model, **config_kwargs)

        # Sequence 的 block 计算必须与 KV Cache/BlockManager 使用相同的块大小。
        # 这里修改的是类属性，因此之后创建的所有 Sequence 都共享这个配置值。
        Sequence.block_size = config.kvcache_block_size

        # ps 保存非零 rank 的子进程；events 用于通知这些进程执行新任务。
        # tensor_parallel_size=1 时两个列表都保持为空。
        self.ps = []
        self.events = []

        # 使用 spawn 创建全新的 Python 子进程。相比 fork，它不会继承已经建立的
        # CUDA 上下文，更适合让每个进程独立绑定一张 GPU。
        ctx = mp.get_context("spawn")

        # 当前主进程充当 rank 0，因此只需要为 rank 1 ... world_size-1 创建子进程。
        for i in range(1, config.tensor_parallel_size):
            # 每个子进程拥有一个 Event。rank 0 把方法调用写入共享内存后，通过
            # Event 唤醒对应 rank，使所有 GPU 执行同一轮模型计算。
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank 0 的 ModelRunner 直接创建在当前进程中。其初始化会建立 NCCL 进程组、
        # 构造并加载 Qwen3、预热模型、计算 KV Cache 容量，并按需捕获 CUDA Graph。
        self.model_runner = ModelRunner(config, 0, self.events)

        # 引擎自己的 tokenizer 负责 generate() 中的字符串编码和最终结果解码。
        # use_fast=True 优先使用 Hugging Face 的 Rust 快速 tokenizer 实现。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # EOS token ID 属于 tokenizer 配置。把它写回 Config 后，新建的 Scheduler
        # 就可以判断模型是否生成了结束符。
        config.eos = self.tokenizer.eos_token_id

        # ModelRunner 初始化已经计算并写回 config.num_kvcache_blocks；Scheduler
        # 随后据此创建 BlockManager，并维护 waiting/running 请求队列。
        self.scheduler = Scheduler(config)

        # 即使用户没有显式调用 exit()，Python 解释器正常退出时也尝试释放
        # 多进程、共享内存、CUDA Graph 和 NCCL 等资源。
        atexit.register(self.exit)

    def exit(self):
        """停止所有 ModelRunner，并等待张量并行子进程退出。"""

        # call("exit") 会先通知其他 TP rank，再在 rank 0 执行 ModelRunner.exit()。
        self.model_runner.call("exit")
        del self.model_runner

        # join() 等待每个子进程完全结束，避免遗留僵尸进程。
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """将一条文本或 token ID 请求包装成 Sequence，加入等待队列。"""

        # generate() 同时支持字符串和已经编码好的 token ID 列表。后者适合基准
        # 测试，也能避免重复 tokenize。Chat Template 需要调用者预先处理。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # Sequence 会复制 token 列表，并保存 temperature、max_tokens、ignore_eos
        # 以及后续调度需要的状态字段。新 Sequence 的初始状态是 WAITING。
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """完成一轮调度、模型执行和请求状态更新。

        一次 step 是 Prefill 或 Decode 中的一种：Prefill 可以为每条请求计算多个
        prompt token；Decode 通常为每条运行中请求计算一个最新 token。
        """

        # Scheduler 返回本轮选中的 Sequence，以及本轮是否属于 Prefill。
        seqs, is_prefill = self.scheduler.schedule()

        # 这个 num_tokens 只用于吞吐率显示：
        # - Prefill 用正数表示本轮实际调度的 token 总数；
        # - Decode 用负数表示本轮序列数，因为每条序列恰好处理一个 token。
        # 负号只是让 generate() 能区分两个阶段，并不表示真的处理了负数个 token。
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        # rank 0 通知所有 TP rank 执行 run()。ModelRunner 会准备 GPU 张量、调用
        # Qwen3、计算 logits 并采样；只有 rank 0 返回采样得到的 token ID。
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 根据新 token 更新缓存进度和 Sequence；满足 EOS 或 max_tokens 的请求
        # 会变为 FINISHED，并释放它占用的 KV Cache block。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 一轮可能只有部分请求完成。这里只把刚刚完成的请求交给 generate() 收集，
        # 仍处于 WAITING/RUNNING 的请求会留在 Scheduler 中等待下一轮。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """当 Scheduler 的 waiting 和 running 队列都为空时返回 True。"""

        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """同步生成一批文本，直到所有请求完成后再返回。

        ``prompts`` 可以是字符串列表，也可以是 token ID 列表的列表。所有请求
        会共享一个 Scheduler，并在不同 step 中被动态组成 batch。

        当前实际返回值是字典列表，每项包含 ``text`` 和 ``token_ids``；函数签名
        中的 ``list[str]`` 是较宽松的旧标注，与实际结构并不完全一致。
        """

        # tqdm 的 total 是请求数量，而不是 token 数；每完成一条请求进度加一。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)

        # 传入单个 SamplingParams 时，所有 prompt 共用相同配置；也可以传入列表，
        # 为每条请求指定不同温度和最大生成长度。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # zip 按位置配对 prompt 与参数。当前实现不会检查两个列表是否等长，
        # 因此显式传入参数列表时，调用者应确保其长度与 prompts 相同。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 请求完成顺序不一定等于输入顺序，所以先用 seq_id -> token_ids 字典收集。
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # 每次循环完成一次 Prefill 或 Decode step。只要还有等待或运行中的请求，
        # 调度器就会继续选择下一批工作。
        while not self.is_finished():
            # perf_counter() 是适合测量短时间间隔的高精度单调时钟。
            t = perf_counter()
            output, num_tokens = self.step()

            # 此处显示的是“最近一轮”的阶段吞吐率，不是整个 generate() 的平均值。
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 一个 step 可能完成零条、一条或多条请求。
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()

        # seq_id 按请求创建顺序递增。排序后恢复输入顺序，并丢掉字典键。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 这里只解码 completion token，不把原 prompt 拼回结果。
        # 每项同时保留 token_ids，便于调用者做调试、统计或继续处理。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
