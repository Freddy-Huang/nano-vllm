"""Paged KV Cache 的物理块元数据管理与 Prefix Cache。

ModelRunner 在 GPU 上创建真正保存 K/V Tensor 的大块显存；本文件不保存 K/V
数值，而是管理“哪些物理 block 正被哪些 Sequence 使用”。核心关系是：

``Sequence.block_table[逻辑块编号] -> GPU KV Cache 的物理 block ID``

例如 ``block_table = [7, 19]`` 表示一条序列的前两个逻辑 token 块分别存放在
物理块 7 和 19。固定大小的离散块使不同长度的请求不必占用连续显存，也使具有
相同完整 Prompt 前缀的请求可以共享已有 K/V。
"""

from collections import deque

import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """一个物理 KV Cache block 的 CPU 侧元数据。"""

    def __init__(self, block_id):
        # 物理块编号。ModelRunner 会用它定位 GPU Cache 的第一维。
        self.block_id = block_id

        # 当前有多少条 Sequence 引用这个物理块。Prefix Cache 共享时可能大于 1；
        # 变为 0 后物理块进入 free_block_ids，可以被缓存命中或覆盖复用。
        self.ref_count = 0

        # 完整 token block 的链式哈希。-1 表示尚未形成可复用的稳定缓存项。
        self.hash = -1

        # 与 hash 对应的 token ID，用于命中后再次比较，防止哈希碰撞造成错误复用。
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """把一个刚填满的物理块登记为可查找的 Prefix Cache。"""

        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """物理块被分配给新内容时，重置引用计数和旧缓存元数据。"""

        # 新分配的独占块首先只有一条 Sequence 使用。
        self.ref_count = 1

        # 这里只清除 CPU 元数据，不会主动把 GPU K/V Tensor 清零；后续 Prefill
        # 或 Decode 会覆盖实际使用的位置。
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """分配、共享、释放 KV Cache block，并维护可复用的完整前缀块。"""

    def __init__(self, num_blocks: int, block_size: int):
        # 每个物理 block 能保存多少个连续 token 的 K/V。
        self.block_size = block_size

        # 为每个 GPU 物理块创建一个同编号的 CPU 元数据对象。
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 链式 hash -> 物理 block ID，用于查询 Prefix Cache。
        self.hash_to_block_id: dict[int, int] = dict()

        # 初始时所有物理块都空闲。deque 让分配和回收可以从两端 O(1) 操作。
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 正被至少一条 Sequence 引用的物理块集合。
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算一个 token block 的 64 位链式哈希。

        第 0 块只哈希自身 token；后续块同时哈希“前一块的链式 hash”和自身
        token，因此相同 token 块出现在不同上下文后面时不会被误认为相同前缀：

        ``H0 = hash(block0)``
        ``H1 = hash(H0, block1)``
        ``H2 = hash(H1, block2)``
        """

        h = xxhash.xxh64()
        if prefix != -1:
            # xxhash 返回 64 位整数，固定转成 8 字节加入下一块的哈希输入。
            h.update(prefix.to_bytes(8, "little"))

        # 把 token 整数数组转换为稳定的连续字节，再交给 xxHash。
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """取出一个空闲物理块，把它重置为一个新的独占块。"""

        # 调用方必须先确认存在足够空闲块；popleft() 选择最早进入空闲队列的块。
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0

        # “空闲”不等于“内容已清除”：刚释放的块仍可能保留可复用前缀。如果现在
        # 要覆盖这个块，就先删除指向它的旧 hash 映射，防止以后命中失效内容。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]

        # 清除旧前缀元数据，并将引用计数设为 1。
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """把引用计数已归零的物理块放回空闲队列。"""

        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

        # 这里刻意不清除 block.hash、block.token_ids 和 GPU K/V：只要该物理块
        # 尚未被新内容覆盖，之后相同 Prompt 仍可以从 free 队列中重新认领它，
        # 这就是请求结束以后 Prefix Cache 仍然可能命中的原因。

    def can_allocate(self, seq: Sequence) -> int:
        """检查能否容纳整条 Sequence，并返回可复用的连续前缀块数。

        Returns:
            ``-1``：空闲物理块不足，当前不能分配；
            ``0``：可以分配，但没有命中 Prefix Cache；
            正整数：可以分配，并且开头这些完整块已有可复用的 K/V。

        此方法只检查和计数，不修改 Sequence 或块状态；真正认领物理块由
        ``allocate()`` 完成。
        """

        # -1 表示当前还没有前一块的链式 hash。
        h = -1
        num_cached_blocks = 0

        # 先假设 Sequence 的每个逻辑块都需要占用一个空闲物理块；下面每命中一个
        # “仍在使用”的共享块，就减去一个新块需求。
        num_new_blocks = seq.num_blocks

        # 最后一块不参与前缀复用：部分块内容还不稳定；即使它刚好填满，本实现也
        # 会保留最后一块重新计算，以得到最后位置的隐藏状态/logits 用于首次采样。
        # Prefix Cache 只保存 K/V，并没有保存可直接采样的隐藏状态或 logits。
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)

            # 把前一块的 hash 串进来，要求从 block 0 开始连续命中。
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)

            # 除了 hash 相等，还要比较真实 token，避免极小概率的哈希碰撞。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # Prefix 必须连续；中间一块未命中后，后面的块也不能跳跃复用。
                break
            num_cached_blocks += 1

            # 已在使用的命中块可以直接共享，只增加 ref_count，不消耗 free 队列；
            # 如果命中块目前空闲，则 allocate() 需要从 free 队列重新认领它，
            # 所以它仍计入 num_new_blocks。
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        # 除可共享的使用中块外，其余每个逻辑块都需要一个 free block。
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为 Sequence 建立完整 block_table，并认领命中的 Prefix Cache。"""

        # 一条 Sequence 只能在尚未持有物理块时进行首次整体分配。
        assert not seq.block_table
        h = -1

        # 先按逻辑顺序挂接命中的连续前缀块。
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 物理块正在被其他 Sequence 使用：共享它并增加引用计数。
                block.ref_count += 1
            else:
                # 物理块已释放，但 K/V 和 hash 尚未被覆盖：从 free 队列重新认领，
                # 无需 reset，因为其中正是本次要复用的缓存内容。
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)

            # block_table 的顺序就是 Sequence 的逻辑 token 块顺序。
            seq.block_table.append(block_id)

        # 未命中的逻辑块各自获得一个新的物理块。其中也包括最后一个必须重新
        # 计算的块；_allocate_block() 会让旧缓存元数据失效。
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        # 一个命中块必定是完整块，因此已缓存 token 数可以直接用块数相乘。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放 Sequence 对所有物理块的引用，并清空其缓存状态。"""

        # 逆序释放与逻辑前缀无关，主要让 Sequence 的尾部块先回到空闲队列。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1

            # 共享块只有在最后一个引用者释放后才真正进入 free 队列。
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # Sequence 已不再拥有任何 K/V。token_ids 本身仍保留，可在被抢占后重算，
        # 或在请求完成后交给 LLMEngine 返回结果。
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Decode 前检查最新 token 是否需要且能够获得一个新物理块。"""

        # bool 在数值比较中等价于 0/1：
        # - len(seq) % block_size == 1：最新 token 是新逻辑块的第一个 token，
        #   block_table 还没有对应物理块，因此要求至少 1 个 free block；
        # - 否则最新 token 仍落在已分配的末块中，需要 0 个新块，检查恒成立。
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """如果最新 token 开启了新逻辑块，就实际分配并追加一个物理块。"""

        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """登记本轮计算后新填满的完整块，使其可被后续 Prefix Cache 查询。

        Scheduler.postprocess() 会在增加 num_cached_tokens 之前调用本方法，所以
        ``num_cached_tokens`` 是本轮开始位置，``num_scheduled_tokens`` 是本轮
        新计算的长度。只有跨过完整 block 边界的块才会被登记。
        """

        # start 是本轮开始前已越过的完整块数；end 是本轮结束后完整块的数量。
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size

        # 没有新填满任何完整块时，不产生 Prefix Cache 条目。
        if start == end: return

        # 从中间块开始登记时，接续前一个逻辑块已经计算好的链式 hash。
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)

            # 同时把 hash/token 元数据记在物理块上，便于释放后继续保留缓存身份。
            block.update(h, token_ids)

            # 如果相同前缀存在多个物理副本，字典保留最近登记的一个候选块。
            self.hash_to_block_id[h] = block.block_id
