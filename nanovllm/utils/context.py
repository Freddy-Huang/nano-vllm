"""保存一次模型前向计算所需的批次与 KV Cache 元数据。

ModelRunner 在 Prefill/Decode 前调用 set_context()；每层 Attention 通过
get_context() 读取相同的请求边界、缓存位置等信息，LM Head 也用它判断
Prefill 时每条请求的最后一个 token。这样 Qwen3 各层只需显式传递
input_ids/positions，无须逐层传递整套调度数据。

_CONTEXT 是当前 Python 进程内的状态。一个 TP rank 对应一个 ModelRunner
进程，各进程会分别设置自己的 Context；一轮执行后调用 reset_context()。
"""

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """一轮模型计算的运行时信息；B 为请求数，T 为本轮新 token 总数。

    dataclass 自动生成构造方法；slots 限定字段，避免意外添加别的属性。
    Prefill 主要使用累计长度，Decode 主要使用 context_lens 和 block_tables。
    """

    # True：处理一段或多段 prompt（Prefill）；False：为每条请求生成下一步
    # 所需的一个 token（Decode）。Attention 据此选择不同的 FlashAttention 路径。
    is_prefill: bool = False

    # Prefill 的 Query 累计长度，形状 [B + 1]。例如本轮两条请求分别有 3、2
    # 个新 token，则为 [0, 3, 5]；相邻元素之差就是各请求本轮的 Query 长度。
    # Attention 用它恢复压平 batch 的边界；LM Head 用 [1:] - 1 找每条末位。
    cu_seqlens_q: torch.Tensor | None = None

    # Prefill 的 Key/Value 累计可见长度，形状也为 [B + 1]。与 Query 不同，
    # 每条 K/V 长度可能包含已经缓存的前缀，所以 cu_seqlens_k 不一定等于
    # cu_seqlens_q；FlashAttention 用它确定每条请求可见的 K/V 范围。
    cu_seqlens_k: torch.Tensor | None = None

    # 本轮各请求的最大 Q 长度、最大 K 长度，供变长 Prefill Attention 调用。
    # Decode 路径不使用这两个值，保持默认 0。
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # 本轮新 token 的 K/V 应写入哪些物理 Cache slot，通常形状 [T]。
    # ModelRunner 根据请求的 block_table 计算；CUDA Graph 的填充行用 -1，
    # 底层 store_kvcache kernel 遇到 -1 会跳过该行。
    slot_mapping: torch.Tensor | None = None

    # Decode 时每条请求当前的完整上下文长度，形状 [B]，包含本轮新 token。
    # flash_attn_with_kvcache 用它限定从 KV Cache 读取的有效历史长度。
    # Prefill 不读取此字段，因此可以为 None。
    context_lens: torch.Tensor | None = None

    # 形状 [B, 最大 block 数]：每条请求的逻辑 block 编号映射到物理 KV Cache
    # block ID。Decode 必须用它寻找历史 K/V；Prefill 只有在复用缓存前缀等
    # 需要读取历史 K/V 时才提供，普通首次 Prefill 可以为 None。
    block_tables: torch.Tensor | None = None


# 初始对象不包含有效批次信息。ModelRunner 每轮会替换它，结束后再清空。
_CONTEXT = Context()


def get_context():
    """返回当前进程本轮的 Context，供 Attention、LM Head 等模块读取。"""
    # 返回的是当前对象本身，不会复制 GPU Tensor 或创建新的批次元数据。
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """由 ModelRunner 在模型前向前设置本轮 Prefill/Decode 的元数据。"""
    global _CONTEXT
    # 这里的参数顺序与 Context 字段声明顺序一致。重新绑定全局对象，使
    # 后续 get_context() 读到本轮数据；Prefill/Decode 不需要的字段保留默认值。
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    """清除本轮元数据，避免下一轮模型计算误用旧的请求或缓存位置。"""
    global _CONTEXT
    # ModelRunner.run() 在采样后调用；CUDA Graph 捕获完一张图后也会调用。
    _CONTEXT = Context()
