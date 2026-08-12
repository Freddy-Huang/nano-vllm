"""第三章配套练习：一个用于理解数据流的 CPU Transformer 小模型。"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """q/k/v: [batch, heads, seq_len, head_dim]。"""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    seq_len = q.size(-2)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms.to(x.dtype)) * self.weight


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        batch, seq_len, hidden_size = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        output, weights = causal_attention(heads(q), heads(k), heads(v))
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
        return self.out(output), weights


class GatedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = TinyCausalSelfAttention(hidden_size, num_heads)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = GatedMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor):
        attn_output, weights = self.attn(self.attn_norm(x))
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, weights


def check_tiny_transformer() -> None:
    torch.manual_seed(0)
    vocab_size, hidden_size = 16, 8
    embedding = nn.Embedding(vocab_size, hidden_size)
    layer = TinyDecoderLayer(hidden_size, num_heads=2, intermediate_size=16)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    token_ids = torch.tensor([[1, 5, 2, 9]])  # [batch=1, seq_len=4]
    hidden = embedding(token_ids)
    output, weights = layer(hidden)
    logits = lm_head(output[:, -1])

    assert hidden.shape == (1, 4, 8)
    assert output.shape == (1, 4, 8)
    assert weights.shape == (1, 2, 4, 4)
    assert logits.shape == (1, 16)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2, 4))

    # causal mask 上三角（未来位置）的注意力必须为 0。
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[0, :, future] == 0)


def main() -> None:
    check_tiny_transformer()
    print("All Transformer basics checks passed!")


if __name__ == "__main__":
    main()

