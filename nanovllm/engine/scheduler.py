"""请求调度器：决定每一轮让哪些 Sequence 执行 Prefill 或 Decode。

Scheduler 位于 LLMEngine 和 ModelRunner 之间：

``LLMEngine.step()``
    -> ``Scheduler.schedule()`` 选择请求并预留 KV Cache
    -> ``ModelRunner.run()`` 执行模型并采样 token
    -> ``Scheduler.postprocess()`` 更新请求状态并释放完成请求的缓存

它不执行 Transformer 计算，而是管理两个请求队列和一个 BlockManager：

- waiting：Prompt 尚未完整 Prefill，或者被抢占后等待重新计算的请求；
- running：Prompt 已完整进入 Prefill 调度，可以继续逐 token Decode 的请求；
- block_manager：分配、共享和释放物理 KV Cache block。

当前策略始终优先处理 waiting 中的 Prefill；只要本轮调度到了 Prefill，就不会
再把 Decode 混入同一个 batch。
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """维护请求状态队列，并在容量限制下构造每一轮推理 batch。"""

    def __init__(self, config: Config):
        # 每轮最多调度多少条 Sequence，限制 Prefill/Decode 的 batch 请求数。
        self.max_num_seqs = config.max_num_seqs

        # 一轮 Prefill 最多计算多少个新 token，每条请求可能贡献多个 token。
        # Decode 每条请求固定贡献一个 token，当前代码主要由 max_num_seqs 限制。
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # tokenizer 的结束符 ID，用于 postprocess() 判断请求是否应当停止。
        self.eos = config.eos

        # 一个 KV Cache block 可以容纳的 token 数。
        self.block_size = config.kvcache_block_size

        # ModelRunner 已根据 GPU 显存计算 num_kvcache_blocks；BlockManager 只管理
        # 这些物理块的元数据，真正的 K/V Tensor 位于 ModelRunner 中。
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # deque 支持从两端 O(1) 添加/移除，便于实现 FIFO 调度和尾部请求抢占。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """只有等待队列和运行队列都为空时，整批生成请求才算结束。"""

        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """把一条新请求加入 waiting 队列尾部，等待 Prompt Prefill。"""

        # 新 Sequence 的状态已经在构造函数中设为 WAITING。
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """选择本轮要执行的请求，返回 ``(seqs, is_prefill)``。

        ``is_prefill=True`` 表示整个返回 batch 都走 Prefill；False 表示都走
        Decode。函数同时会设置各 Sequence 的 num_scheduled_tokens，并通过
        BlockManager 为其预留所需的 KV Cache block。
        """

        # 保存本轮选中的请求；Prefill 和 Decode 共用这个列表，但不会同时出现。
        scheduled_seqs = []

        # 记录本轮已经安排的 token 数，用于执行 max_num_batched_tokens 限制。
        num_batched_tokens = 0

        # ------------------------------------------------------------------
        # 第一阶段：优先从 waiting 队列调度 Prefill
        # ------------------------------------------------------------------
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # 始终查看队首但暂不弹出：如果本轮只能完成部分 Prompt，这条请求仍要
            # 留在 waiting 队首，下一轮继续执行 chunked prefill。
            seq = self.waiting[0]

            # 当前 batch 还剩多少 token 预算。
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            # block_table 为空表示该请求是首次分配，或此前被抢占并释放了缓存。
            if not seq.block_table:
                # can_allocate() 一并检查：
                # 1. 是否存在可复用的完整 Prefix Cache block；
                # 2. 剩余物理块能否容纳该 Sequence。
                # 返回值是命中的连续前缀块数，-1 表示当前空间不足。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 连队首请求都无法分配时，本轮不能越过它调度后面的 Prefill。
                    break

                # 命中的前缀块不需要重新计算；这里只统计尚未缓存的 token。
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # block_table 非空且请求仍在 waiting，说明它正在进行分块 Prefill；
                # 从上轮已经缓存的位置继续计算剩余 Prompt。
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 为了避免一个 batch 中出现多个只完成一部分的 Prompt，当前实现只允许
            # batch 的第一条请求做 chunked prefill。若前面已有请求，本条放到下轮。
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break

            # 首次处理该请求时，建立逻辑块到物理块的 block_table；若命中前缀，
            # allocate() 还会增加共享块的引用计数并设置 num_cached_tokens。
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            # 本轮实际计算量取“剩余 Prompt”和“当前 token 预算”的较小值。
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 如果“已缓存 + 本轮调度”覆盖了完整序列，Prompt 已完整进入本轮
            # Prefill 调度，可以提前转入 RUNNING。实际 GPU 计算紧接着才会发生。
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            # 若只是部分 Prefill，seq 仍留在 waiting[0]；此时 token 预算通常已用尽，
            # 下一次 while 检查会退出，并在后续 step 中继续该请求。
            scheduled_seqs.append(seq)

        # Prefill 优先：只要选到至少一条 waiting 请求，就立即返回 Prefill batch，
        # 本轮不会继续从 running 队列挑选 Decode 请求。
        if scheduled_seqs:
            return scheduled_seqs, True

        # ------------------------------------------------------------------
        # 第二阶段：waiting 本轮无任务时，从 running 队列调度 Decode
        # ------------------------------------------------------------------
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从队首选择等待时间更久的运行中请求。
            seq = self.running.popleft()

            # 新生成的 token 可能跨入一个新逻辑块。若没有空闲物理块，就从队尾
            # 抢占其他较低优先级请求，释放其全部 KV Cache 后再检查当前请求。
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    # 没有其他请求可牺牲时，只能抢占当前请求本身。它会回到
                    # waiting，并在之后重新 Prefill。
                    self.preempt(seq)
                    break
            else:
                # Python 的 while...else 会在 while 条件正常变为 False 时执行；
                # 如果上面的 break 抢占了当前请求，则不会进入这里。

                # Decode 每条请求每轮只计算刚采样到的最后一个 token。
                seq.num_scheduled_tokens = 1

                # 该字段也控制多卡序列化：Decode 只向其他 rank 发送 last_token。
                seq.is_prefill = False

                # 如果当前 token 是新逻辑块的第一个 token，则实际分配一个物理块；
                # 不跨块时无需分配。
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # 正常情况下进入 Decode 分支意味着至少有一条请求可以执行。
        assert scheduled_seqs

        # 前面用 popleft() 临时移除了已选请求；这里按原顺序放回 running 队首，
        # 使 postprocess() 可以找到它们，下一轮也继续维持大致 FIFO 顺序。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占一条运行中请求，释放缓存并让它回到 waiting 队首。"""

        # 请求之后需要重新进行 Prefill，因此恢复 WAITING/is_prefill 状态。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True

        # 当前实现没有 swap 到 CPU；抢占会直接释放全部 KV Cache block，并把
        # num_cached_tokens 清零，所以恢复执行时需要重新计算（可再次尝试前缀命中）。
        self.block_manager.deallocate(seq)

        # appendleft() 让被抢占请求优先于普通新请求恢复，减少饥饿风险。
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """处理一轮模型输出，更新缓存进度、token 列表和结束状态。"""

        # ModelRunner 为每条调度请求返回一个采样 token，按相同顺序逐一处理。
        for seq, token_id in zip(seqs, token_ids):
            # 在增加 num_cached_tokens 之前，为本轮刚填满的完整 block 计算链式 hash，
            # 使之后具有相同 Prompt 前缀的请求可以复用这些 KV Cache。
            self.block_manager.hash_blocks(seq)

            # 本轮调度 token 已经完成模型计算，其 K/V 已写入 Cache。
            seq.num_cached_tokens += seq.num_scheduled_tokens

            # 本轮工作已经结束，清空“待计算”计数。
            seq.num_scheduled_tokens = 0

            # Chunked Prefill 尚未覆盖完整 Prompt 时，本轮虽然计算了 logits 并完成
            # 采样，但该 token 不应加入序列；继续等待下一块 Prompt 即可。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # 完整 Prefill 会追加第一个 completion token；正常 Decode 会追加下一个
            # completion token。这个新 token 尚未经过模型，因此暂未计入缓存数。
            seq.append_token(token_id)

            # 默认遇到 EOS 即结束；ignore_eos=True 时只在生成数量达到 max_tokens
            # 后结束。num_completion_tokens 不包含 Prompt。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED

                # 完成请求不再需要历史 K/V：降低 block 引用计数，释放不再共享的块，
                # 同时清空 seq.block_table 和 seq.num_cached_tokens。
                self.block_manager.deallocate(seq)

                # 请求已经完成，不再参与后续 Decode 调度。
                self.running.remove(seq)
