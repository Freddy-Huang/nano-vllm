"""Qwen3 的输入词嵌入和输出语言模型头。

调用路径：
- Qwen3Model.embed_tokens(input_ids) 把 token ID 查成隐藏向量；
- Qwen3ForCausalLM.compute_logits(hidden_states) 调用 LM Head，得到词表 logits。

两个模块的权重形状都是 [词表大小, hidden_size]。张量并行（TP）时沿词表
这一维切分，每个 GPU/rank 只保存一段连续的词表权重。输入端通过 all-reduce
合成完整隐藏向量；输出端把各段 logits 收集到 rank 0 供采样。
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """将全局 token ID 映射为隐藏向量，并按词表切分 Embedding 权重。

    例如词表大小为 8、TP 大小为 2：rank 0 负责 ID [0, 4)，rank 1 负责
    ID [4, 8)。每个 rank 只查自己持有的权重，随后汇总成相同的输出向量。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # ModelRunner 在创建模型之前已初始化分布式进程组。
        # tp_rank 是当前 GPU 编号，tp_size 是参与张量并行的 GPU 数量。
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # 当前实现将词表等分给各 rank，所以词表大小必须能被 TP 大小整除。
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size

        # 当前 rank 负责的全局 token ID 区间是 [start, end)，end 不包含在内。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 全量权重是 [num_embeddings, embedding_dim]；这里只创建本 rank 的词表行。
        # 例如 V=8、H=3、TP=2 时，每张 GPU 上的 weight 形状为 [4, 3]。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

        # 通用权重加载器会读取参数上的 weight_loader，并调用它切出当前分片。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """从 checkpoint 的完整词表权重中复制当前 rank 的连续行。"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # dim=0 是词表维：rank 1 在上例中取完整权重的第 4～7 行。
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """输入全局 token ID [T]，返回隐藏向量 [T, embedding_dim]。"""
        if self.tp_size > 1:
            # x 中混有其他 rank 的 token ID；mask 标出当前 rank 负责的那些位置。
            # 例如 rank 1 管 [4, 8)，全局 ID 6 对应本地权重的第 2 行。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)

            # 属于本 rank 的 ID 减去 start，转成本地权重下标；其他 ID 变成 0，
            # 这样 F.embedding 不会访问当前分片范围以外的权重。
            x = mask * (x - self.vocab_start_idx)

        # 每个 rank 只用自己的权重查表；单 GPU 时这里直接得到完整结果。
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # mask 从 [T] 扩为 [T, 1]，广播到每个向量维度；清零不属于本 rank
            # 的占位结果。对同一个 token，最多只有一个 rank 保留真实向量。
            y = mask.unsqueeze(1) * y

            # 默认执行求和：各 rank 的局部结果相加后，每个 rank 都得到相同的
            # [T, embedding_dim] 完整向量，供后续 Qwen3 DecoderLayer 使用。
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """把隐藏向量转换成词表 logits，只在 rank 0 汇总完整结果。

    继承父类是为了复用词表分片、权重形状与加载方法；这里的 forward 使用
    F.linear 做投影，不会调用父类的 F.embedding 查表。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        # 当前实现只支持无 bias 的 LM Head，与这里加载的模型权重结构一致。
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """返回每条请求的 logits：rank 0 为 [B, 词表大小]，其他 rank 为 None。"""
        # ModelRunner 在执行模型前写入 Context；这里用它判断当前轮次属于
        # Prefill 还是 Decode，以及 Prefill 时各请求在压平张量中的边界。
        context = get_context()
        if context.is_prefill:
            # Prefill 的 x 为 [本轮所有新 token 数, hidden_size]。生成下一 token
            # 只需要每条请求本轮最后一个位置的隐藏状态，不必投影整个 prompt。
            # cu_seqlens_q 是累计长度：若两条请求本轮长度为 3、2，则为 [0, 3, 5]；
            # [1:] - 1 得到末位下标 [2, 4]。Chunked Prefill 同样取本轮 chunk 末位。
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # Decode 时每条请求本轮只有一个新 token，x 已直接是 [B, hidden_size]。
        # F.linear(x, weight) 等价于 x @ weight.T；各 rank 只计算自己那段词表
        # 的候选 token 分数，局部 logits 形状为 [B, 每 rank 的词表大小]。
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # gather 按 rank 顺序把各段 logits 收集到 rank 0。其他 rank 不需要
            # 保存完整词表分数，因为只有 rank 0 会调用 Sampler 选下一个 token。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        # 单 GPU 时 logits 已是完整词表；多 GPU 时只有 rank 0 返回 [B, V]。
        # 这里不做 softmax：Sampler 后续才会应用 temperature 并采样。
        return logits
