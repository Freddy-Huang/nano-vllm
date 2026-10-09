"""Qwen3 注意力使用的 RoPE（旋转位置编码）。

Qwen3Attention 先把隐藏向量投影为 Q、K、V，再调用这里的 RotaryEmbedding
为 Q、K 注入 token 的位置信息；V 不经过旋转。Prefill 和 Decode 使用同一套
预计算的 cos/sin 表，positions 始终是 token 在各自请求中的序列位置。
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """按位置旋转 Q 或 K，保持输入的形状和 dtype 不变。

    x 为 [T, 头数, head_size]，cos/sin 为 [T, 1, head_size / 2]；
    中间的 1 会广播到所有注意力头，让同一 token 的各个头使用相同位置角度。
    """
    # 沿最后一维分成前后两半；第 i 个和第 i + head_size/2 个分量配成一对。
    # 转成 float32 计算旋转，再在返回前转回 Q/K 原来的精度。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 对每一对分量应用二维旋转矩阵：
    # [x1'] = [ cos  -sin] [x1]；[x2'] = [sin  cos] [x2]。
    # 位置越靠后，cos/sin 对应的角度随之变化。
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    # 拼回原有的 head_size 维，输出仍是 [T, 头数, head_size]。
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """预计算各序列位置的旋转角，并在前向时应用到 Q、K。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size

        # 当前实现要求整个注意力头都参与旋转，不支持只旋转前 rotary_dim 维。
        assert rotary_dim == head_size

        # 每对特征使用一种旋转频率。偶数下标 0, 2, ... 对应 head_size/2 个
        # 频率，base（Qwen3 的 rope_theta）控制不同维度的角度变化速度。
        # inv_freq: [rotary_dim / 2]。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # t 列出允许的序列位置 [0, max_position_embeddings)；einsum 做外积，
        # 得到每个位置、每个特征对的旋转角 freqs: [max_position, rotary_dim/2]。
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # 沿最后一维存成 [cos | sin]，再插入头维：
        # cache: [max_position, 1, rotary_dim]；前向索引后可广播给所有 Q/K 头。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # buffer 跟随模块移动设备，但不是可训练参数。persistent=False 表示
        # 不写入 checkpoint：这些值只由构造参数决定，初始化时可重新计算。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据 positions 旋转 Q/K，返回形状不变的 (query, key)。

        positions: [T]，每个值是 token 在自己序列中的位置；
        query: [T, Q 头数, head_size]；key: [T, KV 头数, head_size]。
        Q/K 头数可以不同，但共享同一 token 的位置角度。
        """
        # 按位置直接查预计算表；位置必须落在 [0, max_position_embeddings) 内。
        # cos_sin: [T, 1, rotary_dim]，第二维可广播到各自的注意力头。
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 只旋转 Q、K。二者参与注意力分数计算，因此需要携带位置信息；
        # V 保存内容向量，由 Q/K 算出的注意力权重决定如何汇总。
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """获取 RoPE 模块；相同参数的连续调用复用同一个预计算表。"""
    # Qwen3 的各 DecoderLayer 使用相同配置，缓存可避免每层重复生成 cos/sin。
    # lru_cache(1) 只保留最近一组参数对应的 RotaryEmbedding 实例。
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
