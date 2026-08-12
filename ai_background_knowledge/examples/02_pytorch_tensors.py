"""第二章配套练习：只需 PyTorch，默认在 CPU 上运行。"""

import torch
from torch import nn
import torch.nn.functional as F


def check_shape_and_indexing() -> None:
    x = torch.arange(12).view(3, 4)
    assert x.shape == (3, 4)
    assert x.dtype == torch.int64
    assert x.device.type == "cpu"
    assert x[:, -1].tolist() == [3, 7, 11]


def check_reshape_and_qkv() -> None:
    torch.manual_seed(0)
    num_tokens, hidden_size = 3, 8
    num_heads, head_dim = 2, 4

    hidden = torch.randn(num_tokens, hidden_size)
    weight = torch.randn(3 * hidden_size, hidden_size)
    qkv = F.linear(hidden, weight)
    q, k, v = qkv.chunk(3, dim=-1)

    assert qkv.shape == (num_tokens, 3 * hidden_size)
    assert q.shape == k.shape == v.shape == (num_tokens, hidden_size)

    q = q.view(num_tokens, num_heads, head_dim)
    assert q.shape == (num_tokens, num_heads, head_dim)
    assert q.flatten(1, -1).shape == (num_tokens, hidden_size)


def check_broadcast_and_softmax() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    temperatures = torch.tensor([0.5, 2.0])
    probs = torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)

    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
    # 更低的 temperature 会让最大概率更大，分布更尖锐。
    assert probs[0].max() > probs[1].max()


class ScaleAndShift(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.register_buffer("shift", torch.arange(size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def check_module_state() -> None:
    layer = ScaleAndShift(3)
    output = layer(torch.ones(2, 3))

    assert output.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert [name for name, _ in layer.named_parameters()] == ["scale"]
    assert [name for name, _ in layer.named_buffers()] == ["shift"]


def check_attention_shapes() -> None:
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 2, 3, 4, 5
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    scores = q @ k.transpose(-2, -1) / (head_dim**0.5)
    probs = torch.softmax(scores, dim=-1)
    output = probs @ v

    assert scores.shape == (batch, heads, seq_len, seq_len)
    assert output.shape == (batch, heads, seq_len, head_dim)


def main() -> None:
    check_shape_and_indexing()
    check_reshape_and_qkv()
    check_broadcast_and_softmax()
    check_module_state()
    check_attention_shapes()
    print("All PyTorch tensor checks passed!")


if __name__ == "__main__":
    main()

