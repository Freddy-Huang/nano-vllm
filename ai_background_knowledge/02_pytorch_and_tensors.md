# 02｜PyTorch 与 Tensor：先学会追踪 shape

> 这章只讲阅读 Nano-vLLM 必需的 PyTorch。建议先运行配套练习，再对照源码阅读。全章练习只使用 CPU。

## 本章目标

学完后，你应该能：

- 用 `shape + dtype + device + 维度含义` 描述一个 Tensor；
- 看懂索引、切片、`view`、`flatten`、`split`、`chunk` 和广播；
- 手算线性层、矩阵乘法和 softmax 的输入输出 shape；
- 理解 `nn.Module`、`Parameter`、buffer、训练模式和推理模式；
- 初步解释 `torch.distributed` 中 `all_reduce`、`gather` 和 `barrier`。

本章先不讲自动求导细节、优化器和训练循环，因为 Nano-vLLM 做的是推理，不训练模型。

## 重点地图

| 优先级 | 知识 | 在项目中的用途 |
| --- | --- | --- |
| 必会 | shape、dtype、device | 阅读所有模型计算和显存分配 |
| 必会 | 索引、reshape、split、flatten | QKV 拆分、多头变形、选最后 token |
| 必会 | 广播、矩阵乘法、softmax | Attention、采样、归一化 |
| 必会 | `nn.Module`、`Parameter`、buffer | 模型层、权重和 RoPE 缓存 |
| 看懂即可 | contiguous、stride、in-place | FlashAttention/Triton 输入约束和性能 |
| 先理解概念 | distributed collective | Tensor Parallel 多卡通信 |

---

## 1. Tensor 是带统一类型的多维数组

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

print(x.shape)   # torch.Size([2, 3])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
print(x.ndim)    # 2
print(x.numel()) # 6
```

读 AI 代码时，不要只写“x 是 Tensor”，而要写完整：

```text
x: [2, 3], float32, CPU
第 0 维是 batch，第 1 维是 feature
```

这四项是 Tensor 的身份证：

- **shape**：每一维有多长；
- **dtype**：每个元素怎样存储；
- **device**：数据位于 CPU 还是某张 GPU；
- **维度含义**：代码作者赋予每一维的业务含义。

两个 Tensor shape 相同，不代表含义相同。`[8, 128]` 既可能是 8 个 token 的 hidden states，也可能是一个权重矩阵。

### 常见 dtype

| dtype | 常见用途 | 每元素字节数 |
| --- | --- | ---: |
| `torch.int64` | token ID、普通索引 | 8 |
| `torch.int32` | block table、长度、slot mapping | 4 |
| `torch.float32` | 高精度计算、softmax、统计量 | 4 |
| `torch.float16` | 模型权重和计算 | 2 |
| `torch.bfloat16` | 模型权重和计算，数值范围较好 | 2 |
| `torch.bool` | mask | 1 |

项目中 `input_ids` 是 `int64`，很多元数据是 `int32`，模型权重通常采用模型配置指定的浮点类型。

---

## 2. 创建 Tensor 与移动设备

```python
a = torch.tensor([1, 2, 3])
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
empty = torch.empty(2, 3)  # 只分配内存，内容未初始化
positions = torch.arange(4)  # [0, 1, 2, 3]
random = torch.randn(2, 3)
```

`torch.empty` 不是全零。它适合随后立刻写满的内存，Nano-vLLM 用它分配权重和 KV Cache。

如果机器有 CUDA：

```python
x_gpu = x.to("cuda")
x_cpu = x_gpu.to("cpu")
```

同一次运算的 Tensor 通常必须在同一设备：CPU Tensor 不能直接与 CUDA Tensor 做矩阵乘法。

项目中常见：

```python
torch.tensor(data, pin_memory=True).cuda(non_blocking=True)
```

含义是先在可锁页（pinned）CPU 内存中创建数据，再尝试异步复制到 GPU。它是数据传输优化，不会改变数学结果。

---

## 3. 索引、切片与布尔 mask

```python
x = torch.tensor([
    [10, 11, 12],
    [20, 21, 22],
])

assert x[0, 1].item() == 11
assert x[:, 1].tolist() == [11, 21]
assert x[-1].tolist() == [20, 21, 22]
```

- `:` 表示这一维全选；
- `.item()` 把只有一个元素的 Tensor 转成 Python 标量；
- `.tolist()` 转成 Python 列表，通常会触发 GPU 到 CPU 的同步/传输，因此热路径中不要滥用。

布尔 mask 用于筛选或置零：

```python
ids = torch.tensor([1, 7, 3, 9])
mask = ids >= 5
assert mask.tolist() == [False, True, False, True]
assert ids[mask].tolist() == [7, 9]
```

词表并行 Embedding 用 mask 判断 token ID 是否属于当前 GPU 保存的词表分片。

### 高级索引：每个序列取不同位置

```python
hidden = torch.arange(5 * 2).view(5, 2)
last_indices = torch.tensor([1, 4])
selected = hidden[last_indices]
assert selected.shape == (2, 2)
```

prefill 把多个序列压平后，LM Head 使用每个序列最后一个 query 的下标，只计算这些位置的 logits。

---

## 4. 改 shape：数据通常没有变，观察方式变了

```python
x = torch.arange(24)       # [24]
y = x.view(2, 3, 4)       # [2, 3, 4]
z = y.reshape(6, 4)        # [6, 4]
flat = y.flatten(1, -1)    # [2, 12]
```

- `view` 要求内存布局兼容，通常共享原数据；
- `reshape` 尽量返回 view，不行时会复制，更宽容；
- `flatten(start_dim, end_dim)` 合并一段连续维度；
- `-1` 可以让 PyTorch 自动推导这一维，但一个操作中只能有一个 `-1`。

Qwen3 Attention 中：

```text
投影后 q: [T, num_heads * head_dim]
view 后 q: [T, num_heads, head_dim]
Attention 输出: [T, num_heads, head_dim]
flatten 后: [T, num_heads * head_dim]
```

这里 `T` 是本轮所有 token 的总数，不一定等于 batch size。

### `split` 与 `chunk`

```python
x = torch.arange(12).view(2, 6)
a, b, c = x.split([2, 1, 3], dim=-1)
assert [a.shape, b.shape, c.shape] == [(2, 2), (2, 1), (2, 3)]

left, right = x.chunk(2, dim=-1)
assert left.shape == right.shape == (2, 3)
```

- `split([q_size, kv_size, kv_size])` 按指定长度拆分 Q/K/V；
- `chunk(2, -1)` 尝试平均拆成两份，项目用它实现 gated MLP 和 RoPE。

---

## 5. 广播：自动补维和扩展

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # [2, 3]
bias = torch.tensor([10.0, 20.0, 30.0])  # [3]
y = x + bias  # bias 被视为 [1, 3]，对两行复用
```

广播从最后一维向前比较，两维在以下情况兼容：

- 大小相同；或
- 其中一个大小为 1；或
- 某个 Tensor 没有这一维。

采样器中：

```python
logits:                  [batch_size, vocab_size]
temperatures:            [batch_size]
temperatures.unsqueeze(1): [batch_size, 1]
logits / temperatures:   [batch_size, vocab_size]
```

这样每个请求能用自己的 temperature 缩放整行词表分数。

**常见陷阱：** 广播成功不代表业务正确。每次都写出对齐后的 shape，确认扩展的是预期维度。

---

## 6. 矩阵乘法与线性层

线性层的数学形式：

```text
y = x W^T + b
```

若：

```text
x: [T, in_features]
W: [out_features, in_features]
b: [out_features]
```

则：

```text
y: [T, out_features]
```

PyTorch 示例：

```python
import torch.nn.functional as F

x = torch.randn(4, 8)
weight = torch.randn(12, 8)
bias = torch.randn(12)
y = F.linear(x, weight, bias)
assert y.shape == (4, 12)
```

为什么权重是 `[out, in]` 而不是 `[in, out]`？这是 PyTorch 线性层的存储约定；计算时内部使用 `weight.T`。

### 批量矩阵乘法

Attention 常见 shape：

```text
Q: [batch, heads, q_len, head_dim]
K: [batch, heads, kv_len, head_dim]
K^T: [batch, heads, head_dim, kv_len]
Q @ K^T: [batch, heads, q_len, kv_len]
```

`torch.matmul` 会对前面的 batch/head 维批量处理，只在最后两维做矩阵乘法。

---

## 7. softmax：把分数变成概率

```python
scores = torch.tensor([[1.0, 2.0, 3.0]])
probs = torch.softmax(scores, dim=-1)

assert probs.shape == scores.shape
assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
```

softmax 在指定维度上满足：

```text
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

`dim=-1` 表示沿最后一维。在 Attention 中，对每个 query 面向所有 key 的分数做 softmax；在采样中，对整个词表的 logits 做 softmax。

实现通常会先减去最大值以提高数值稳定性。使用 `torch.softmax` 时内部已经处理，不要自己直接对大数 `exp()`。

### temperature 的影响

```text
probs = softmax(logits / temperature)
```

- temperature 小于 1：差距被放大，结果更确定；
- temperature 大于 1：分布更平，结果更多样；
- Nano-vLLM 当前不允许接近 0 的 temperature，因此没有单独的 greedy 分支。

---

## 8. `nn.Module`：PyTorch 模型的基本积木

```python
from torch import nn

class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.up = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))

model = TinyMLP(4)
x = torch.randn(2, 4)
y = model(x)  # 实际会调用 model.forward(x)
assert y.shape == (2, 4)
```

重点：

- 必须调用 `super().__init__()`；
- 把子模块赋给 `self.xxx`，PyTorch 才能自动注册它；
- 通常调用 `model(x)`，不要直接调用 `model.forward(x)`；
- `model.modules()` 能递归遍历模型本身和所有子模块；
- `nn.ModuleList` 用来注册一组层，普通 Python list 不会自动注册其中模块。

Qwen3Model 用 `ModuleList` 保存多个 DecoderLayer，再在 `forward` 中逐层执行。

---

## 9. Parameter、buffer 与普通属性

一个 Module 中常见三种状态：

```python
class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.register_buffer("cache", torch.zeros(4), persistent=False)
        self.description = "ordinary Python attribute"
```

| 状态 | 跟随 `.to(device)` | 出现在 `parameters()` | 通常进 state dict |
| --- | --- | --- | --- |
| `Parameter` | 是 | 是 | 是 |
| buffer | 是 | 否 | 是；`persistent=False` 时否 |
| 普通属性 | 否 | 否 | 否 |

- 模型权重是 Parameter；
- RoPE 预计算的 cos/sin 不是要训练的权重，用 buffer；
- `persistent=False` 表示它可重新计算，不必保存在模型权重文件里。

Nano-vLLM 给 Parameter 对象动态挂上 `weight_loader`，让不同并行层知道如何加载自己的权重分片。

---

## 10. 推理模式、精度与 in-place 操作

```python
@torch.inference_mode()
def predict(model, x):
    return model(x)
```

推理不需要记录反向传播图。`torch.inference_mode()` 能减少相关开销和内存使用。

`model.eval()` 与它解决的问题不同：

- `model.eval()` 把 Dropout、BatchNorm 等层切换到推理行为；
- `torch.inference_mode()` 关闭自动求导记录；
- 实际推理通常两者都用。Nano-vLLM 的 Qwen3 结构没有 Dropout/BatchNorm 热点，但仍用 `inference_mode` 避免创建求导图。

反过来，`model.train()` 只是切换层的训练行为，也不会自动开启或执行反向传播。是否记录梯度由 grad mode 和 Tensor 状态决定。

项目的 RMSNorm 会暂时转成 float32：

```text
原模型 dtype -> float32 计算均方值 -> 转回原 dtype
```

这是常见的数值稳定性选择。

带下划线的方法通常是 in-place 操作：

```python
x.add_(1)  # 直接修改 x 的存储
x.mul_(2)
```

它能减少临时 Tensor，但也更容易因共享存储产生意外。阅读时要特别标记“谁被修改了”。

---

## 11. contiguous 与 stride：先懂概念

Tensor 是“存储 + shape + stride”的视图。stride 表示某一维前进一步，要跨过多少个元素：

```python
x = torch.arange(12).view(3, 4)
y = x.transpose(0, 1)

print(x.stride())  # 常见为 (4, 1)
print(y.stride())  # 常见为 (1, 4)
print(y.is_contiguous())  # False
```

转置通常只改变观察方式，不搬数据，因此不连续。某些 kernel 要求最后一维连续：

```python
z = y.contiguous()  # 必要时复制为连续布局
```

`attention.py` 的 KV Cache 写入函数会 assert stride，确保 Triton kernel 使用的地址计算成立。

第一次阅读时只记住：

1. shape 描述逻辑结构；
2. stride 描述物理跳步；
3. `contiguous()` 可能复制数据，所以不是免费的。

---

## 12. distributed collective：多个 rank 一起操作

Tensor Parallel 中，每个进程通常控制一张 GPU，并有唯一 `rank`。总进程数叫 `world_size`。

### `all_reduce`

```text
rank 0: tensor A ─┐
rank 1: tensor B ─┼─ sum -> 每个 rank 都得到 A + B
rank 2: tensor C ─┘
```

Row Parallel Linear 中，每个 rank 只算一部分输入维度带来的局部结果，最后 `all_reduce(sum)` 得到完整输出。

### `gather`

```text
rank 0: logits_0 ─┐
rank 1: logits_1 ─┼─ gather -> 只有目标 rank 收到全部分片
rank 2: logits_2 ─┘
```

词表并行 LM Head 中，每个 rank 只算一段词表 logits，最终 gather 到 rank 0 并拼接。

### `barrier`

所有 rank 都到达屏障后才能继续。项目在共享内存创建/连接和退出时用它同步。

collective 必须由进程组中的参与者以兼容顺序调用，否则可能永久等待。

---

## 13. 把 shape 映射回 Nano-vLLM

以 `T` 表示本轮 token 总数，`H` 表示 hidden size：

| 源码位置 | 输入 | 关键变化 | 输出 |
| --- | --- | --- | --- |
| `embed_tokens` | token IDs `[T]` | 查表 | `[T, H]` |
| `qkv_proj` | `[T, H]` | 一次线性投影后 split | q/k/v |
| q `view` | `[T, heads*D]` | 拆出 head 维 | `[T, heads, D]` |
| Attention | q/k/v | 因果注意力 | `[T, heads, D]` |
| `flatten(1, -1)` | `[T, heads, D]` | 合并 head 和 D | `[T, heads*D]` |
| MLP gate/up | `[T, H]` | 投影、SiLU、逐元素乘 | `[T, intermediate]` |
| LM Head | `[batch, H]` | 与词表权重线性变换 | `[batch, vocab]` |

建议精读：

- [`qwen3.py`](../nanovllm/models/qwen3.py)：沿每个 `forward` 标 shape；
- [`sampler.py`](../nanovllm/layers/sampler.py)：观察 temperature 广播与 softmax；
- [`embed_head.py`](../nanovllm/layers/embed_head.py)：观察索引和 distributed；
- [`attention.py`](../nanovllm/layers/attention.py)：观察 shape、stride 和 KV Cache。

---

## 14. 动手练习

运行：

```bash
python ai_background_knowledge/examples/02_pytorch_tensors.py
```

然后尝试：

1. 把练习中的 `T=3, H=8, heads=2` 改成自己的值，保证 `H` 能被 heads 整除。
2. 手算 `[4, 8]` 经过权重 `[24, 8]` 的线性层后 shape，再拆成三个 `[4, 8]`。
3. 创建 logits `[2, 5]` 和 temperatures `[2]`，解释为什么必须 `unsqueeze(1)`。
4. 找出 `Qwen3Attention.forward()` 中所有改变 shape 的语句。
5. 解释为什么 `torch.empty` 创建的 KV Cache 不需要先清零：有效位置由长度和 block table 决定，并会在使用前写入。

<details>
<summary>部分答案</summary>

2. 线性输出为 `[4, 24]`，沿最后一维拆成 q/k/v 后各为 `[4, 8]`。
3. `[2]` 与 `[2, 5]` 从末维对齐时不兼容；`unsqueeze(1)` 后为 `[2, 1]`，每行温度可广播到 5 个词表项。
4. `split`、三个 `view`、Attention 内部计算、`flatten`；线性层也改变最后一维。

</details>

## 15. 自测清单

- [ ] 描述 Tensor 时能同时写出 shape、dtype、device 和维度含义。
- [ ] 能手算 `F.linear` 的输出 shape。
- [ ] 能解释 `view(-1, heads, head_dim)` 中 `-1` 的含义。
- [ ] 能区分 `split`、`chunk` 和 `flatten`。
- [ ] 能判断一个广播是否符合业务意图。
- [ ] 能解释 softmax 的维度和 temperature 的作用。
- [ ] 能区分 Parameter、buffer 和普通属性。
- [ ] 知道 in-place 操作会修改原 Tensor。
- [ ] 能用一句话解释 `all_reduce`、`gather` 和 `barrier`。

下一章会把这些操作组合成 Transformer。阅读时始终带着一张纸记录 shape；这比记住层的名字更重要。
