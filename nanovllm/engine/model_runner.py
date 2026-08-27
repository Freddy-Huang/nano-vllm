"""把 Scheduler 选出的 Sequence 转换成 GPU 张量，并执行一次模型推理。

ModelRunner 是“控制面”和“模型计算”之间的桥梁：

``Scheduler``
    -> Sequence、block_table、Prefill/Decode 标志
    -> ``ModelRunner`` 准备 input_ids、positions 和 Attention Context
    -> ``Qwen3ForCausalLM`` 执行 GPU 计算
    -> ``Sampler`` 采样 token ID
    -> ``Scheduler.postprocess()`` 更新请求状态

除此之外，它还负责初始化张量并行进程、加载权重、测量显存、一次性创建 KV
Cache，以及为 Decode 捕获和重放 CUDA Graph。
"""

import pickle

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """一张 GPU（一个 TP rank）对应的模型执行器。"""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """初始化当前 rank 的通信环境、模型、KV Cache 和性能优化资源。"""

        self.config = config
        hf_config = config.hf_config

        # 保存后续输入准备和执行路径所需的核心配置。
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        # rank 0 收到的是所有子进程 Event 的列表；非零 rank 只收到自己的 Event。
        self.event = event

        # 每个 TP rank 都加入同一个 NCCL 进程组。world_size=1 时也会初始化一个
        # 单进程组，使模型层可以统一调用 dist.get_world_size()/get_rank()。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)

        # 当前实现直接使用 rank 作为本机 CUDA 设备编号：rank 0 -> GPU 0，以此类推。
        torch.cuda.set_device(rank)

        # 暂存进程原来的默认 dtype，模型初始化结束后还原。
        default_dtype = torch.get_default_dtype()

        # Qwen3Config.dtype 决定新建参数和 KV Cache 的精度，例如 float16/bfloat16。
        torch.set_default_dtype(hf_config.dtype)

        # 在初始化阶段把默认设备设成 CUDA，因此 Qwen3 内部的 torch.empty/ones
        # 会直接在当前 GPU 上创建，不需要先建 CPU Tensor 再搬运。
        torch.set_default_device("cuda")

        # 每个 rank 构造相同的模型结构；张量并行层只创建并加载属于本 rank 的分片。
        self.model = Qwen3ForCausalLM(hf_config)

        # 从本地 .safetensors 加载权重，并按 TP rank/融合层映射写入模型参数。
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # 先用最大规模的虚拟请求运行一次，触发 torch.compile 并测量峰值显存。
        self.warmup_model()

        # 根据显存预算创建真正的 GPU KV Cache，并挂接到每层 Attention。
        self.allocate_kv_cache()

        # eager 模式适合学习调试；非 eager 模式额外捕获常见 Decode batch size。
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 只还原默认设备和 dtype；模型参数本身仍然保留在当前 GPU 上。
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多卡时，rank 0 负责接收 LLMEngine 调用并广播命令；其他 rank 初始化完成
        # 后进入常驻 loop，等待共享内存中的 run/exit 等方法调用。
        if self.world_size > 1:
            if rank == 0:
                # 共享内存用于发送方法名和轻量参数，固定容量为 1 MiB。
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)

                # 等待其他 rank 完成初始化，再开始接受真实推理请求。
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """释放当前 rank 的共享内存、CUDA Graph 和 NCCL 进程组。"""

        if self.world_size > 1:
            self.shm.close()

            # 确保所有 rank 都已停止工作并关闭句柄后，再由 rank 0 删除共享内存名。
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        # 等待当前 GPU 上已提交的异步 kernel 完成，再销毁通信资源。
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """非零 rank 的常驻命令循环。"""

        while True:
            method_name, args = self.read_shm()

            # 子 rank 调用 call() 时不会再次广播，只会在本地执行对应方法。
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """等待 rank 0 通知，并从共享内存反序列化一次方法调用。"""

        assert self.world_size > 1 and self.rank > 0

        # Event 避免子进程不断轮询共享内存；rank 0 写完数据后才会唤醒它。
        self.event.wait()

        # 前 4 字节是 pickle 数据长度，使用小端整数编码。
        n = int.from_bytes(self.shm.buf[0:4], "little")

        # 数据格式是 [method_name, arg0, arg1, ...]。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """由 rank 0 将方法调用写入共享内存，并唤醒所有其他 rank。"""

        assert self.world_size > 1 and self.rank == 0

        # Sequence 自定义了 pickle 状态：Prefill 发送完整 token_ids，Decode 只发送
        # last_token，从而减少多卡控制数据的传输量。
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data

        # 每个子 rank 拥有独立 Event，但它们读取同一份只读命令数据。
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """让所有 TP rank 执行同名方法，并返回当前 rank 的本地结果。"""

        # rank 0 先广播命令；world_size=1 或非零 rank 不执行这一分支。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)

        # getattr("run") 等价于取 self.run，随后用 *args 展开参数调用。
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """以接近配置上限的虚拟 Prefill 预热模型，并记录显存峰值。"""

        # empty_cache() 只释放 PyTorch 缓存分配器中未使用的显存，不删除模型参数。
        torch.cuda.empty_cache()

        # 后续 allocate_kv_cache() 需要本轮预热产生的 peak 统计。
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len

        # 单条虚拟序列不超过一轮 token 预算，也不超过配置的最大模型长度。
        seq_len = min(max_num_batched_tokens, max_model_len)

        # 尽量填满 token 预算，同时不超过 max_num_seqs。
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)

        # token 0 只用于构造合法形状；预热不关心生成内容。
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            # 预热绕过 Scheduler，所以需要手动说明整个虚拟 Prompt 都在本轮执行。
            seq.num_scheduled_tokens = seq_len

        # KV Cache 尚未创建，prepare_prefill() 看到空 block_table 会跳过写缓存映射。
        # 该调用还会触发带 @torch.compile 模块和 Sampler 的首次编译。
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据显存预算计算物理 block 数，一次性创建所有层的 K/V Cache。"""

        config = self.config
        hf_config = config.hf_config

        # mem_get_info() 返回当前 CUDA 设备的空闲显存和总显存字节数。
        free, total = torch.cuda.mem_get_info()
        used = total - free

        # peak/current 来自刚才的预热：peak-current 是一次真实大请求可能额外需要的
        # 临时显存，不能全部拿给常驻 KV Cache。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # GQA 的 KV 头按 TP rank 切分，每张 GPU 只保存本地 KV heads。
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # 一个“物理 block”包含当前 rank 上所有 Transformer 层的 K 和 V：
        # 2(K/V) × 层数 × block token 数 × KV头数 × head维度 × 每元素字节数。
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # 可给 Cache 的预算 = 目标总占用 - 当前已用 - 预留的峰值临时空间。
        # 公式中的 -peak+current 等价于减去 (peak-current)。整除得到完整块数量。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 概念 shape：
        # [K/V, layer, physical_block, token_offset, local_kv_head, head_dim]
        # 默认设备/dtype 仍是当前 GPU 和模型精度。
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        # 依次找到每层底层 Attention，并把总 Cache 中对应 layer 的视图挂进去。
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """将不同长度的 Python block_table 补齐并传到 GPU。"""

        # FlashAttention 需要一个矩形 Tensor，列数取本 batch 最长 block_table。
        max_len = max(len(seq.block_table) for seq in seqs)

        # -1 是无效物理块占位符，不属于任何 Sequence。
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]

        # pinned CPU 内存配合 non_blocking=True，允许更高效的异步 H2D 复制。
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """把一批变长 Sequence 压平为 Prefill 所需的 GPU 输入和 Context。"""

        # 多条请求的新 token 会串成一维列表；FlashAttention 用 cu_seqlens 恢复边界。
        input_ids = []
        positions = []

        # cumulative sequence lengths：第 i 和 i+1 个值之差就是第 i 条序列长度。
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        # slot_mapping 与 input_ids 一一对应，指定每个新 token 的 K/V 物理写入位置。
        slot_mapping = []

        # Fresh Prefill 可直接使用本轮局部 K/V；存在历史缓存时才需要 block_tables。
        block_tables = None
        for seq in seqs:
            # 本轮 Query 从已缓存位置开始，长度由 Scheduler 设置。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q

            # Key/Value 的可见上下文包含历史缓存和本轮 token，因此总长度是 end。
            seqlen_k = end

            # 只把尚未缓存且已被本轮调度的 token 加入模型输入。
            input_ids.extend(seq[start:end])

            # RoPE 使用 token 在各自原序列中的绝对位置，而不是压平 batch 后的位置。
            positions.extend(range(start, end))

            # 例如两条 q 长度为 3、2 的请求会得到 [0, 3, 5]。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)

            # k 长度可能大于 q，因为前缀的 K/V 已存在于 Cache。
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # 模型预热时没有 BlockManager/block_table，也尚未创建 KV Cache。
            if not seq.block_table:    # warmup
                continue

            # 找出本轮 token 区间 [start, end) 覆盖的逻辑 block 范围。
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                # 物理扁平 slot = 物理 block ID × block_size + block 内偏移。
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    # 中间完整块写到该物理块末尾。
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 最后一块只写到本轮 end 对应的逻辑偏移。
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        # 只要整个 batch 的 K 总长度大于 Q 总长度，就说明至少有一条请求带历史
        # Prefix/Chunked Prefill Cache；Attention 需要 block_tables 找到历史 K/V。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # 先在 pinned CPU 内存构造，再异步复制到当前 GPU。token/position 用 int64，
        # FlashAttention 的长度和 slot 元数据使用 int32。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Qwen3.forward() 只显式接收 input_ids/positions；Attention 通过全局 Context
        # 取得批次边界、Cache 写入位置和 block table，避免这些参数逐层传递。
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """为每条运行中请求准备一个最新 token 的 Decode 输入。"""

        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # Prefill 已产生第一个 completion token；Decode 只输入尚未缓存的 last_token。
            input_ids.append(seq.last_token)

            # 最新 token 是当前 Sequence 的最后一个位置，使用从 0 开始的位置编号。
            positions.append(len(seq) - 1)

            # Attention 可见长度包含本轮最新 token，因此等于当前 Sequence 总长度。
            context_lens.append(len(seq))

            # Scheduler.may_append() 已保证 block_table 覆盖最新 token。最后一个物理
            # block 的起始 slot，加上块内 token 数减一，就是最新 token 的写入位置。
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)

        # Decode 同样使用 pinned CPU -> CUDA 异步复制。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Decode 必须读取完整历史 KV Cache，所以始终提供每条请求的 block_table。
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """把每条请求的采样温度组成 rank 0 上的 GPU Tensor。"""

        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """选择 eager 或 CUDA Graph 路径，返回当前 batch 的词表 logits。"""

        # Prefill 形状变化大，始终 eager 执行；调试配置或超过已捕获最大 batch 512
        # 的请求也走 eager。self.model() 返回隐藏状态，compute_logits() 再投影到词表。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 以下分支仅用于 Decode。真实 bs 可能没有对应图，因此选择第一个不小于
            # bs 的已捕获 batch size，多出的行用静态缓冲区作为 padding。
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # CUDA Graph 要求重放时 Tensor 地址和 shape 不变，所以不能传入新 Tensor；
            # 这里把本轮数据复制到捕获时创建的静态缓冲区。
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions

            # padding 行的 slot=-1，Triton store_kvcache kernel 会跳过这些行，避免
            # 虚拟 token 污染真实 KV Cache。
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping

            # padding 行上下文长度为 0；真实行复制本轮长度。
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens

            # 只复制真实 block_table 使用的行列；FlashAttention 根据 context_lens
            # 判断有效历史长度，不会把 padding 请求当成真实输出。
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放已经记录的 Qwen3 GPU kernel，不再逐个通过 Python 发起 kernel。
            graph.replay()

            # 图中捕获的是 Transformer 主干；只取真实 bs 行，并在图外计算 LM Head。
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """执行一轮完整推理：准备输入、运行模型、采样并清理 Context。"""

        # Scheduler 保证同一个 batch 不混合 Prefill 和 Decode。
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        # TP 的完整词表 logits 最终只 gather 到 rank 0，因此只有 rank 0 需要温度。
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 所有 rank 都执行本地模型分片，并在模型层内部通过 collective 协作。
        logits = self.run_model(input_ids, positions, is_prefill)

        # 只有 rank 0 拥有完整 logits 并负责采样；其他 rank 返回 None。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # Context 只属于当前一轮，避免下一轮 Attention 误读旧的批次元数据。
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """为若干常用 Decode batch size 捕获 Qwen3 主干 CUDA Graph。"""

        config = self.config
        hf_config = config.hf_config

        # 当前只为不超过 512 的 Decode 捕获图，更大的 batch 在 run_model() 中 eager。
        max_bs = min(self.config.max_num_seqs, 512)

        # 一条最大长度 Sequence 最多需要多少个 block，决定静态 block table 列数。
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # __init__ 此时尚未恢复默认 CPU 设备，因此这些固定缓冲区直接创建在 GPU。
        # 它们在所有 graph replay 中复用，地址和最大 shape 始终不变。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 小 batch 使用密集档位，大 batch 每 16 条捕获一个图。运行时向上选择最近
        # 档位，例如 bs=10 使用 bs=16 的图，其余 6 行作为 padding。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 先捕获大图，再捕获小图，并让它们尽量共享同一个 CUDA memory pool。
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            # 捕获时 Attention 也通过 Context 读取这些静态缓冲区的视图。
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # 捕获前先在当前 shape 上预热，完成可能存在的延迟初始化。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup

            # with 块内发起的 GPU 操作会被记录进 graph，之后可重复 replay。
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                # 第一张图创建 memory pool，后续图复用它以减少多份静态显存。
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存所有固定缓冲区，run_model() 重放前会把真实输入复制到这些 Tensor。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
