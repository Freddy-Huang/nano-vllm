"""执行 Qwen3 的底层注意力计算，并读写当前层的 KV Cache。

Qwen3Attention 已经完成 QKV 投影、Q/K 归一化和 RoPE，这里接收处理后的
q、k、v。ModelRunner 通过 Context 提供批次边界、物理缓存位置和调度阶段：
Prefill 对变长 prompt 计算注意力；Decode 每条请求只处理一个新 token，
复用 KV Cache 中的历史 K/V。每个 DecoderLayer 都有独立的一份 Attention
和 KV Cache 视图。
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    # 一个 Triton program 处理一个新 token；idx 是它在本轮压平 batch 中的下标。
    idx = tl.program_id(0)

    # slot 是该 token 在物理 KV Cache 中的扁平位置，由 ModelRunner 根据
    # block_table 和 block 内偏移计算。CUDA Graph 的填充行使用 -1，不写缓存。
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    # 一个 token 的 K/V 各含 D = 本地 KV 头数 × head_dim 个元素。
    # key_stride/value_stride 是相邻 token 的起始地址间隔；头和头内维度连续，
    # 因此用 arange(0, D) 一次读取该 token 的整条 K/V 向量。
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # Cache 的每个物理 token slot 同样占 D 个连续元素。K/V 分别写入
    # 当前层的 k_cache/v_cache；之后 Decode 会从这些位置读取历史信息。
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """把本轮 K/V 写到当前层的物理 KV Cache。

    key/value: [N, 本地 KV 头数, head_dim]；
    k_cache/v_cache: [物理 block 数, block_size, 本地 KV 头数, head_dim]；
    slot_mapping: [N]，给出每个新 token 对应的物理 slot。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # Triton kernel 把一个 token 的所有 KV 头看作连续的 D 个元素。
    # 因此头内元素必须相邻、相邻头必须相隔 head_dim；Cache 的相邻
    # token slot 也必须相隔 D。若布局不满足这些要求，扁平地址计算会错误。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 网格中启动 N 个 program，让每个新 token 独立写入自己的物理 slot。
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """根据当前 Context 选择 Prefill 或 Decode，并复用当前层的 KV Cache。"""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        # 这些都是当前 TP rank 上的头数；GQA 允许 Q 头多于 KV 头。
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Qwen3Attention 传入 head_dim ** -0.5，作为点积注意力的缩放系数。
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        # 模型初始化时尚未按显存预算分配 KV Cache，先用空 Tensor 占位。
        # ModelRunner.allocate_kv_cache() 随后把当前层的 K/V Cache 视图挂到这里。
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """计算因果注意力；输入的 Q/K 已完成 RoPE，返回各 Q 头的结果。

        Prefill 输入形状：q [T, Q 头数, head_dim]，k/v [T, KV 头数, head_dim]；
        Decode 时 T 等于请求数 B，每条请求只输入一个新 token。
        """
        # ModelRunner 在本轮前向前设置 Context；每层 Attention 读到相同的
        # 请求边界和 slot_mapping，但使用自己独立的 K/V Cache 视图。
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 有真实 Cache 时，先把本轮新 token 的 K/V 写进去。这样后面的
        # Prefix/Chunked Prefill 或 Decode 都能从 Cache 看见这些新 token。
        # 初始化预热时 Cache 仍为空，直接使用传入的 k/v，不执行写入。
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            # 普通 Prefill 没有历史前缀，直接使用本轮算出的 k/v；若有已缓存
            # 前缀，block_tables 会指向物理块，K/V 改从 Cache 读取历史和本轮数据。
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache

            # 多条不同长度的请求压平在一个 batch 中；cu_seqlens_q/k 恢复各自
            # 的边界，max_seqlen_q/k 给出本批最大长度。causal=True 保证每个
            # Query 只能看见自己和更早位置，不能看见未来 token。
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # Decode 的 q 原本是 [B, Q 头数, head_dim]；增加长度为 1 的
            # 序列维，符合 flash_attn_with_kvcache 的输入格式。
            # Cache 已包含当前 token 的 K/V；context_lens 指定每条请求的有效
            # 上下文长度，block_tables 把逻辑块映射到物理 Cache 块。
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)

        # Qwen3Attention 随后会合并 Q 头，并用 o_proj 投影回 hidden_size。
        return o
