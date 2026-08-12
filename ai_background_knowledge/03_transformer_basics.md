# 03｜Transformer 基础：跟着一个 token 走完 Qwen3

> 本章从“下一 token 预测”出发，逐层搭起 Nano-vLLM 使用的 decoder-only Transformer。公式只保留理解代码所需的部分。

## 本章目标

学完后，你应该能：

- 画出 token IDs 到 next-token logits 的完整数据流；
- 解释 Embedding、Self-Attention、MLP、残差和 RMSNorm 的职责；
- 手算 Q/K/V、Attention scores 和输出的 shape；
- 说明 causal mask、MHA/GQA、RoPE 和 gated MLP；
- 对照 `qwen3.py` 读懂一个 DecoderLayer 的执行顺序。

## 重点地图

| 优先级 | 知识 | Nano-vLLM 对应位置 |
| --- | --- | --- |
| 必会 | decoder-only、自回归、causal mask | Qwen3 整体 |
| 必会 | Embedding、Attention、MLP | `models/qwen3.py` |
| 必会 | 残差、Pre-Norm、RMSNorm | `layers/layernorm.py` |
| 必会 | Multi-Head / Grouped-Query Attention | `Qwen3Attention` |
| 看懂即可 | RoPE | `layers/rotary_embedding.py` |
| 看懂即可 | SiLU gated MLP | `layers/activation.py` |

---

## 1. 大语言模型在做什么

给定前面的 token，模型输出“下一个 token”的分数：

```text
[今, 天, 天, 气] -> 模型 -> 下一个 token 的 logits -> “很”
[今, 天, 天, 气, 很] -> 模型 -> 新 logits -> “好”
```

`logits` 是尚未归一化的分数，每个词表项一个：

```text
hidden state [H]
  -> LM Head
  -> logits [vocab_size]
  -> temperature + softmax
  -> 采样一个 token ID
```

Qwen3 是 **decoder-only Transformer**：只用 Transformer 的 decoder 风格堆叠，通过 causal attention 保证当前位置看不到未来 token。

### 训练与推理的视角不同

训练时，一段长度为 `L` 的文本可以并行预测每个位置的下一个 token；推理生成时，未来 token 尚不存在，只能采样一个，再把它接回上下文继续运行。

Nano-vLLM 专注后者，因此核心问题不只是模型数学，还包括如何避免重复计算。

---

## 2. 总数据流

先记住这条主线：

```text
文本
  -> Tokenizer（不属于神经网络）
  -> token IDs [T]
  -> Embedding [T, H]
  -> N × DecoderLayer
       RMSNorm -> Self-Attention -> 残差
       RMSNorm -> MLP            -> 残差
  -> Final RMSNorm [T, H]
  -> LM Head [需要预测的位置数, vocab_size]
  -> Sampler
  -> next token IDs
```

符号约定：

- `T`：本轮进入模型的 token 总数；
- `H`：hidden size；
- `D`：每个 attention head 的维度；
- `Nq`：query head 数；
- `Nkv`：key/value head 数；
- `V`：vocab size；
- 通常 `H = Nq × D`。

---

## 3. Tokenizer 与 Embedding 是两件事

Tokenizer 在模型外把字符串映射为离散整数：

```text
"hello world" -> [15339, 1917]
```

Embedding 是模型内的一张可学习查找表：

```text
weight: [V, H]
input_ids: [T]
weight[input_ids] -> hidden_states: [T, H]
```

每个 token ID 只是词表行号，本身的大小没有“语义距离”。Embedding 把它变成 H 维向量，后续层才能做连续数学运算。

### 位置从哪里来

仅靠 Embedding，交换 token 顺序只会交换向量，Attention 不知道谁先谁后。Nano-vLLM 不把位置向量直接加到 hidden states，而是用 RoPE 把位置信息作用到 Q/K。

---

## 4. Self-Attention 的直觉

一句话：**每个 query 根据自己与所有可见 key 的匹配程度，对相应 value 做加权汇总。**

从 hidden states 线性投影出三组向量：

- Query：当前位置“正在寻找什么”；
- Key：每个位置“能用什么特征被匹配”；
- Value：匹配后实际取走的信息。

单头公式：

```text
scores = Q K^T / sqrt(D)
weights = softmax(scores + causal_mask)
output = weights V
```

### shape 推导

先忽略 batch：

```text
Q:       [Lq, D]
K:       [Lk, D]
K^T:     [D, Lk]
scores:  [Lq, Lk]
weights: [Lq, Lk]
V:       [Lk, D]
output:  [Lq, D]
```

`scores[i, j]` 表示第 i 个 query 对第 j 个 key 的关注分数。每行 softmax 后总和为 1。

### 为什么除以 `sqrt(D)`

维度增大时，点积数值通常也会增大，softmax 更容易饱和。缩放能把数值保持在较稳定范围。源码里是：

```python
self.scaling = self.head_dim ** -0.5
```

也就是 `1 / sqrt(head_dim)`。

---

## 5. Causal mask：不能偷看未来

长度为 4 时，允许看到的位置：

```text
            key 位置
            0  1  2  3
query 0     ✓  ×  ×  ×
位置  1     ✓  ✓  ×  ×
      2     ✓  ✓  ✓  ×
      3     ✓  ✓  ✓  ✓
```

实现上，被屏蔽的位置在 softmax 前加上负无穷，概率就变成 0：

```python
scores = scores.masked_fill(~causal_mask, float("-inf"))
```

FlashAttention 接口使用 `causal=True` 在 kernel 内处理，不需要在 Python 中真的创建一个巨大 mask。

### 一个常见误解

causal mask 限制“未来”，不限制当前位置看自己。decode 时新 token 的 query 可以看到完整历史，也可以看到自己刚算出的 K/V。

---

## 6. Multi-Head Attention：从不同子空间看关系

一个大向量被拆成多个 head：

```text
q 投影: [T, Nq * D]
view:   [T, Nq, D]
```

每个 head 独立做 attention，最后拼回去：

```text
各 head 输出 [T, Nq, D]
flatten       [T, Nq * D]
o_proj        [T, H]
```

多个 head 可以学习不同关系，但不要把它理解成每个 head 必然具有固定、人类可命名的功能。

---

## 7. GQA：Query head 多，KV head 少

经典 Multi-Head Attention 通常 `Nq = Nkv`。Grouped-Query Attention 允许多个 query head 共享一组 K/V head：

```text
Nq = 8, Nkv = 2

Q heads:  q0 q1 q2 q3 | q4 q5 q6 q7
               ↓                ↓
KV heads:       kv0             kv1
```

每 4 个 query head 共用一个 KV head。

好处是 KV Cache 更小。每个历史 token 的缓存大小大致与 `2 × Nkv × D` 成正比，`Nkv` 从 8 降到 2，K/V 存储约降为四分之一。

Qwen3Attention 中 shape 是：

```text
q: [T, num_heads_per_gpu, head_dim]
k: [T, num_kv_heads_per_gpu, head_dim]
v: [T, num_kv_heads_per_gpu, head_dim]
```

FlashAttention 负责在计算时按组匹配 Q 与 KV。

---

## 8. RoPE：把位置编码进 Q 和 K

RoPE（Rotary Position Embedding）把向量按二维小组旋转，旋转角度由 token 位置和频率决定：

```text
[x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
```

直觉上：

- 同一个 token 出现在不同位置，旋转角不同；
- Q/K 点积因而包含相对位置信息；
- V 是被汇总的内容，不参与匹配，所以不应用 RoPE。

Nano-vLLM 的实现流程：

```text
positions [T]
  -> 从 cos_sin_cache 取对应行
  -> 把 q/k 最后一维分成两半
  -> 应用旋转公式
  -> 拼回原 shape 和 dtype
```

`cos_sin_cache` 是 buffer，不是训练参数；同一配置的 RoPE 模块还通过 `lru_cache` 复用。

第一次学习不必推导不同频率如何编码相对距离，只需能解释输入输出 shape 不变，以及为什么只处理 Q/K。

---

## 9. MLP：每个 token 独立变换

Attention 负责 token 之间交换信息；MLP 对每个 token 的 hidden vector 独立做非线性变换。

Qwen3 使用 gated MLP，简化公式：

```text
gate = x W_gate^T
up   = x W_up^T
middle = SiLU(gate) * up
output = middle W_down^T
```

shape：

```text
x:      [T, H]
gate:   [T, I]
up:     [T, I]
middle: [T, I]
output: [T, H]
```

`I` 是 intermediate size，通常大于 H。

Nano-vLLM 把 gate 和 up 两次投影合并：

```text
gate_up_proj: [T, H] -> [T, 2I]
chunk(2): gate [T, I], up [T, I]
SiLU(gate) * up -> [T, I]
down_proj -> [T, H]
```

合并权重和操作可以减少 kernel launch 与内存读写。

---

## 10. RMSNorm：控制数值尺度

RMSNorm 对最后一维计算均方根：

```text
rms(x) = sqrt(mean(x²) + eps)
output = x / rms(x) * weight
```

shape 始终不变：`[T, H] -> [T, H]`。

与 LayerNorm 相比，RMSNorm 不减均值。Qwen3 使用可学习 `weight [H]`，它会广播到所有 token。

源码暂时将 x 转为 float32 计算统计量，之后转回原 dtype，这是为了数值稳定性。

---

## 11. 残差连接与 Pre-Norm

残差连接把子层输入直接加回输出：

```text
y = x + Attention(Norm(x))
z = y + MLP(Norm(y))
```

它为信息和梯度提供直接路径。Qwen3 是 Pre-Norm：先归一化，再进入子层。

Nano-vLLM 为减少显存读写，把“上一次输出 + residual”和 RMSNorm 融合。源码第一次看有些绕，可以把 `Qwen3DecoderLayer.forward()` 等价理解为：

```python
attn_input = input_layernorm(x)
x = x + self_attn(attn_input)
mlp_input = post_attention_layernorm(x)
x = x + mlp(mlp_input)
```

实际实现让 `hidden_states` 暂存子层输出，让 `residual` 保存待相加的主干；最后一层外的 final norm 会完成最后一次融合相加。

### 用符号跟一层

若进入某层前的完整主干状态为 `r0`：

```text
n0 = RMSNorm(r0)
a  = Attention(n0)
r1 = r0 + a
n1 = RMSNorm(r1)
m  = MLP(n1)
```

函数返回 `(hidden_states=m, residual=r1)`；下一次 fused norm 或最终 norm 会先得到 `r2 = r1 + m`。

---

## 12. LM Head：从 hidden state 回到词表

```text
hidden: [B, H]
lm_head weight: [V, H]
logits: [B, V]
```

Embedding 是 `token ID -> hidden vector`，LM Head 是 `hidden vector -> 每个 token 的分数`。某些模型共享两者的权重，Qwen3 配置开启 `tie_word_embeddings` 时，项目会让它们使用同一份数据。

### prefill 为什么只取每条序列最后一个位置

推理只需要采样“接在 prompt 后的第一个 token”。prompt 中间位置的 logits 不会被使用，因此 LM Head 只选择每条序列最后一个 query 的 hidden state：

```text
压平 hidden: [所有新 token, H]
last_indices: [batch]
选择后:       [batch, H]
logits:       [batch, V]
```

decode 本来每个请求就只输入一个新 token，因此输入已经是 `[batch, H]`。

---

## 13. 完整 Qwen3 DecoderLayer 导读

按以下顺序读 [`qwen3.py`](../nanovllm/models/qwen3.py)：

1. `Qwen3Model.forward`：Embedding、逐层循环、Final Norm；
2. `Qwen3DecoderLayer.forward`：两条残差和两个 Norm；
3. `Qwen3Attention.forward`：QKV 投影、split、view、RoPE、Attention、输出投影；
4. `Qwen3MLP.forward`：合并的 gate/up 与 down；
5. `Qwen3ForCausalLM.compute_logits`：LM Head。

在纸上维护下面的 shape 表：

| 变量 | 概念 shape |
| --- | --- |
| `input_ids` | `[T]` |
| `hidden_states` | `[T, H]` |
| `q` | `[T, Nq/tp, D]` |
| `k`, `v` | `[T, Nkv/tp, D]` |
| attention output | `[T, Nq/tp, D]` |
| MLP gate/up | `[T, I/tp]` 各一份 |
| rank 0 logits | `[batch, V]` |

这里 `/tp` 表示 Tensor Parallel 下每张 GPU 保存的 head 或中间维分片；单卡时忽略。

---

## 14. 常见误解

- **Attention 参数最多。** 通常 MLP 参数也非常大，甚至占单层多数。
- **每个 token 只关注前一个 token。** 它可以关注所有可见历史位置，只是权重不同。
- **causal mask 让模型一次只能算一个 token。** 训练和 prefill 仍可并行算多个位置；mask 只是限制信息方向。
- **KV Cache 缓存整个层输出。** 它只缓存每层 Attention 已计算的 K/V。
- **RoPE 改变 Tensor shape。** 它旋转数值，shape 不变。
- **Residual 就是把原始 Embedding 一直加回去。** 每一层加回的是该子层入口处的主干状态。

---

## 15. 动手练习

运行一个只有 CPU 的缩小版 Transformer：

```bash
python ai_background_knowledge/examples/03_transformer_basics.py
```

练习：

1. 打印 causal attention 的概率矩阵，确认右上角全为 0、每行和为 1。
2. 把 token 序列长度从 4 改成 6，列出每个中间 Tensor shape。
3. 令 `num_query_heads=4`、`num_kv_heads=2`，画出 query head 到 KV head 的共享关系。
4. 手写 `SiLU(gate) * up` 的 shape 推导。
5. 对照源码解释 Q/K 为什么在 RoPE 前做 RMSNorm，而 V 不做。

最后一题不需要推导论文结论：它是 Qwen3 的模型设计选择，代码如实复现；关键是识别只有 Q/K 经过 q_norm/k_norm 和 RoPE。

## 16. 自测清单

- [ ] 能画出 token IDs 到 logits 的主链。
- [ ] 能用一句话解释 Q、K、V。
- [ ] 能手算 `Q @ K.T` 和 `weights @ V` 的 shape。
- [ ] 能画出 causal mask。
- [ ] 能解释多头 Attention 和 GQA 的区别。
- [ ] 能解释 RoPE 为什么作用于 Q/K 而非 V。
- [ ] 能写出 RMSNorm 和 gated MLP 的计算顺序。
- [ ] 能还原 fused residual + RMSNorm 的概念计算。
- [ ] 能解释 prefill 时 LM Head 为什么只选最后位置。

下一章会从模型结构转到推理系统：tokenizer、自回归循环、prefill/decode、KV Cache、Paged KV Cache、调度和采样。

