# 04｜大模型推理基础：从 prompt 到逐 token 生成

> 模型结构回答“怎么算”，推理引擎回答“怎样把很多请求算得快且不浪费显存”。本章是理解 Nano-vLLM 调度器、KV Cache 和 ModelRunner 的核心背景。

## 本章目标

学完后，你应该能：

- 解释 tokenizer、chat template、logits、采样和停止条件；
- 区分 prefill 与 decode 的输入、计算特点和性能瓶颈；
- 说明 KV Cache 避免了哪些重复计算；
- 区分 logical block、physical block、block table 和 slot mapping；
- 解释 continuous batching、chunked prefill、preemption 和 prefix cache；
- 手算一个简化请求从 waiting 到 finished 的过程。

## 重点地图

| 优先级 | 知识 | Nano-vLLM 对应位置 |
| --- | --- | --- |
| 必会 | tokenization、自回归、采样 | `llm_engine.py`、`sampler.py` |
| 必会 | prefill 与 decode | `scheduler.py`、`model_runner.py` |
| 必会 | KV Cache | `attention.py` |
| 必会 | Paged KV Cache、block table、slot | `block_manager.py` |
| 必会 | continuous batching 与抢占 | `scheduler.py` |
| 看懂即可 | prefix cache、chunked prefill | BlockManager 与 Scheduler |

---

## 1. 一次请求的五个阶段

```text
1. 格式化对话
   messages -> chat template -> prompt 字符串

2. 分词
   prompt -> tokenizer.encode -> prompt token IDs

3. Prefill
   一次处理 prompt 中尚未缓存的多个 token
   -> 保存每层 K/V
   -> 得到第一个 next-token logits

4. Decode 循环
   采样 token -> 输入这个 token -> 更新 K/V -> 再采样

5. 停止和解码
   EOS / max_tokens -> completion token IDs -> 文本
```

Nano-vLLM 的公开 API 把这五步包进 `LLM.generate()`。学习源码时要不断确认当前逻辑属于哪个阶段。

---

## 2. Tokenizer：模型只认识 token ID

Tokenizer 负责：

```text
字符串 <-> token ID 列表
```

一个 token 可能是一个字、半个单词、空格与单词的组合、标点或特殊标记。不能假设“一 token = 一词”。不同 tokenizer 对同一文本也可能给出不同结果。

```python
token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(token_ids)
```

### Chat template

聊天模型训练时看到的不只是用户文字，还包含角色和轮次边界等特殊格式：

```python
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "你好"}],
    tokenize=False,
    add_generation_prompt=True,
)
```

`add_generation_prompt=True` 通常在末尾补上“现在轮到 assistant”的格式。格式错误时，即使模型和推理引擎完全正确，回答质量也会下降。

### 三种长度不要混淆

- 字符数：Python `len(text)`；
- token 数：`len(tokenizer.encode(text))`；
- 模型最大上下文长度：prompt token + completion token 必须遵守的限制。

显存和 Attention 计算更关心 token 数，不是字符数。

---

## 3. 自回归生成循环

最朴素的生成伪代码：

```python
tokens = tokenizer.encode(prompt)

for _ in range(max_tokens):
    logits = model(tokens)       # 每次重算整个上下文，低效
    next_token = sample(logits[-1])
    tokens.append(next_token)
    if next_token == eos_token_id:
        break
```

每轮只能确定一个新 token，因为下一轮的概率取决于刚才实际采样的结果。不同请求可以组成 batch 并行，但单个请求的 decode 轮次仍有时间依赖。

Nano-vLLM 的 `Sequence` 同时保存：

- `token_ids`：prompt + 已生成 token；
- `num_prompt_tokens`：两部分的边界；
- `last_token`：decode 下一轮真正需要输入的 token；
- `temperature`、`max_tokens`、`ignore_eos`：采样与停止配置。

---

## 4. logits、temperature 与采样

模型为词表中每个 token 产生一个 logit：

```text
logits: [batch_size, vocab_size]
```

Nano-vLLM 当前流程：

```text
logits / temperature
  -> softmax
  -> 按概率随机采样
  -> token ID
```

temperature 越低，较大 logit 更占优势；越高，概率更均匀。项目的采样器使用 exponential-race/Gumbel-Max 等价技巧实现分类采样，最后通过 `argmax` 选出 token。

### 当前实现边界

`SamplingParams` 只有：

- `temperature`；
- `max_tokens`；
- `ignore_eos`。

它没有 top-k、top-p、重复惩罚、stop strings 等生产系统常见选项。阅读时要区分“大模型推理的一般能力”和“这个教学项目已经实现的能力”。

### 停止条件

调度器在采样后检查：

```text
token 是 EOS 且没有 ignore_eos
或
num_completion_tokens == max_tokens
```

完成后释放该请求占用的 KV blocks，并从 running 队列移除。

---

## 5. Prefill 与 Decode

这是本章最重要的区别。

### Prefill

输入 prompt 中尚未缓存的一段，通常每个请求有多个 token：

```text
请求 A 新 token: [a0, a1, a2]
请求 B 新 token: [b0, b1]
压平 input_ids: [a0, a1, a2, b0, b1]
```

这些位置在 causal mask 下仍能并行计算。prefill 通常矩阵较大、并行度高，更偏计算密集。

### Decode

每个运行中请求只输入上轮刚生成的一个 token：

```text
请求 A: last_token_a
请求 B: last_token_b
input_ids: [last_token_a, last_token_b]
```

每个 query 仍要读取完整历史的 K/V。单轮矩阵小但反复执行，通常更受显存带宽、kernel launch 和调度开销影响。

| 对比 | Prefill | Decode |
| --- | --- | --- |
| 每请求本轮输入 | 多个 token | 1 个 token |
| 是否可在序列内并行 | 是 | 当前轮只有 1 个 |
| K/V | 计算并写入 | 新增一个并读取历史 |
| 常见瓶颈 | 计算量 | 内存带宽与启动开销 |
| Nano-vLLM 输入总长 | 所有新 token 数 | batch size |

`LLMEngine.step()` 用正 `num_tokens` 表示 prefill token 数，用负的序列数表示 decode，方便进度显示分别计算吞吐。

---

## 6. 为什么需要 KV Cache

没有缓存时，第 n 轮 decode 会再次为所有历史 token 计算 K/V：

```text
轮 1: prompt 的 K/V
轮 2: prompt + token1 的 K/V
轮 3: prompt + token1 + token2 的 K/V
```

但模型权重不变，历史 token 在每层产生的 K/V 也不变。KV Cache 保存它们：

```text
prefill: 计算 prompt 的 K/V -> 写 cache
decode:  只算新 token 的 Q/K/V
         新 K/V 写 cache
         Q 与 cache 中完整 K/V 做 attention
```

每一层都有自己的 K/V，不能只缓存一份。

### KV Cache 概念 shape

Nano-vLLM 为整个模型预分配：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads_per_gpu, head_dim]
```

第一维的 2 分别是 K 和 V。

每个 block 占用字节数：

```text
2 × num_layers × block_size × num_kv_heads_per_gpu × head_dim × dtype_bytes
```

这解释了为什么 GQA（减少 KV heads）能显著减小推理显存。

---

## 7. 为什么 KV Cache 要分页

如果为每个请求提前分配“最大上下文长度”的连续 KV 空间，会浪费大量显存；请求实际长度不同，结束时间也不同，连续大块还容易产生碎片。

Paged KV Cache 把显存划成固定大小的物理 block：

```text
物理 KV blocks:
block 0 | block 1 | block 2 | block 3 | block 4 | ...

请求 A 的逻辑上下文: [逻辑块0, 逻辑块1, 逻辑块2]
请求 A 的 block table: [3, 0, 4]
```

请求逻辑上连续，物理上可以散布。Attention kernel 通过 block table 找到每段历史 K/V。

### 四个容易混淆的概念

| 概念 | 含义 | 示例 |
| --- | --- | --- |
| logical block index | 一个请求内部第几个块 | 0、1、2 |
| physical block ID | 全局 KV Cache 中哪一个块 | 3、0、4 |
| block table | logical -> physical 的映射 | `[3, 0, 4]` |
| slot | 物理 block 中某个 token 的线性位置 | `block_id * block_size + offset` |

若 block size 为 4、物理 block ID 为 3、块内 offset 为 2：

```text
slot = 3 * 4 + 2 = 14
```

`slot_mapping` 为本轮每个新 token 指定 K/V 写入位置。

---

## 8. 手算一次 block 与 slot

假设 block size 为 4，请求 token 为：

```text
[10, 11, 12, 13, 14, 15]
```

它需要：

```text
logical block 0: [10, 11, 12, 13]（已满）
logical block 1: [14, 15]        （未满）
num_blocks = ceil(6 / 4) = 2
last_block_num_tokens = 2
```

若 block table 为 `[5, 2]`：

```text
token 10..13 -> physical block 5 -> slots 20..23
token 14..15 -> physical block 2 -> slots 8..9
```

下一个 token 16 应写入最后一个块的 offset 2，也就是 slot `2*4+2=10`。

注意源码的 decode 准备发生在新 token 已经 append 到 Sequence 之后，因此它写的是 `last_block_num_tokens - 1` 对应的位置。时间点不同，公式外观会差 1。

---

## 9. Prefix Cache：复用相同 prompt 前缀

两个请求可能共享完整前缀：

```text
A: [系统提示的 256 tokens] + 用户问题 A
B: [系统提示的 256 tokens] + 用户问题 B
```

如果系统提示已经完成 prefill，B 可以复用 A 的前缀 KV blocks，不再重复计算。

Nano-vLLM 只缓存并复用**完整 block**，因为未满 block 还可能继续写入，不适合作为稳定共享单元。

### 链式 hash

每个完整 token block 的 hash 会包含：

```text
当前 block token IDs + 前一个 block 的 hash
```

因此同一组 token 出现在不同前缀后，通常不是同一个缓存项。hash 命中后还会比较 `token_ids`，用于防御 hash collision。

### 引用计数

多个请求复用一个物理 block 时：

```text
ref_count = 持有该 block 的请求数
```

一个请求结束只把计数减一；降为 0 时物理块才真正回到空闲队列。即使空闲 block 的 hash 元数据暂时保留，也可以被后续相同前缀重新激活；再次分配给其他内容时旧映射才失效。

---

## 10. Continuous Batching：每一轮重新组 batch

静态 batching 会等整批请求全部结束，短请求完成后留下空位。Continuous batching 在每个推理 step 重新选择请求：

```text
step 1: A prefill, B prefill
step 2: A decode,  B decode
step 3: A 完成，B decode，C 加入 prefill
...
```

这样 GPU 更容易持续有工作，同时新请求不必等待一个完整批次结束。

Nano-vLLM 的 Scheduler 维护：

- `waiting`：等待或被抢占、需要 prefill 的序列；
- `running`：prefill 完成、可以 decode 的序列；
- `finished` 不单独放队列，完成后从 running 移除并回传结果。

状态图：

```text
新增请求                 prefill 完成
WAITING ---------------------------------> RUNNING
   ^                                          |
   | KV block 不足时 preempt                   | EOS / max_tokens
   +------------------------------------------+----> FINISHED
```

---

## 11. Scheduler 的选择逻辑

Nano-vLLM 每轮要么调度 prefill，要么调度 decode；只要本轮选到了 waiting 请求，就直接返回 prefill batch。

### Prefill 限制

- 请求数不超过 `max_num_seqs`；
- token 总数不超过 `max_num_batched_tokens`；
- 需要足够 KV blocks；
- 只有 batch 中第一个请求允许 chunked prefill。

### Chunked prefill

超长 prompt 可能超过单轮 token 预算。调度器只处理一段：

```text
prompt 10 tokens, 本轮预算 6
step 1 prefill: positions 0..5
step 2 prefill: positions 6..9
然后进入 RUNNING
```

`num_scheduled_tokens` 是当前这轮要算多少；`num_cached_tokens` 是已经真正算完并写入 KV Cache 的数量。调度前后不要把它们混为一谈。

### Decode 与抢占

decode 每请求只调度一个 token。若新 token 需要开启物理 block 但无空闲块，调度器会抢占一个请求：

1. 状态改回 WAITING；
2. 释放其 block table；
3. 清零缓存进度；
4. 以后重新 prefill。

这是简单但代价较高的 recomputation 策略。生产级系统可能有更复杂的 swap 或优先级机制。

---

## 12. 变长 batch 为什么要压平

prefill 的两个请求长度不同：

```text
A: 3 tokens
B: 2 tokens
```

若 padding 到相同长度，会浪费计算。FlashAttention varlen 接口把它们压平：

```text
input: [A0, A1, A2, B0, B1]
cu_seqlens_q: [0, 3, 5]
```

`cu_seqlens` 是累计长度：

- A 位于 `[0:3]`；
- B 位于 `[3:5]`。

若 B 已复用 4 个 prefix tokens，只需要计算 2 个新 query：

```text
query length: 2
key/value context length: 4 + 2 = 6
```

于是 `cu_seqlens_k` 的末值可能大于 `cu_seqlens_q`。这正是 ModelRunner 判断存在 prefix cache 上下文的依据。

---

## 13. ModelRunner 的输入元数据

| Tensor | Prefill | Decode | 含义 |
| --- | --- | --- | --- |
| `input_ids` | `[本轮新 token 总数]` | `[batch]` | 真正输入模型的 token |
| `positions` | 同上 | `[batch]` | token 在各自完整序列中的位置 |
| `slot_mapping` | 同上 | `[batch]` | 新 K/V 写入哪些物理 slots |
| `cu_seqlens_q/k` | `[batch+1]` | 不使用 | 压平序列的边界 |
| `context_lens` | 不使用 | `[batch]` | 当前完整上下文长度 |
| `block_tables` | prefix 命中时 | 总是 | 查找物理 KV blocks |

这些元数据被写进全局 `Context`，Attention 层再读取。这样模型每一层不必显式传递所有调度参数，但也引入了全局状态，阅读时要跨文件追踪。

---

## 14. 一轮 step 的完整状态变化

```text
Scheduler.schedule()
  -> 选 seqs
  -> 分配/扩展 blocks
  -> 设置 num_scheduled_tokens

ModelRunner.run()
  -> prepare_prefill/decode
  -> 模型 forward，写 KV Cache
  -> LM Head + sampler 得到 token IDs

Scheduler.postprocess()
  -> 为刚填满的 blocks 建 hash
  -> num_cached_tokens += num_scheduled_tokens
  -> num_scheduled_tokens = 0
  -> 若本轮可采样，append 新 token
  -> 检查 EOS/max_tokens
  -> 完成则释放 blocks
```

一个容易忽略的点：prefill chunk 尚未覆盖完整 prompt 时，本轮不会 append 采样 token；只有完整 prompt prefill 完成后才使用 logits 采第一个 completion token。

---

## 15. 延迟、吞吐与显存

推理性能至少要区分：

- **TTFT（Time To First Token）**：提交请求到首 token，主要受排队与 prefill 影响；
- **TPOT（Time Per Output Token）**：首 token 之后每个 token 的平均间隔，主要受 decode 影响；
- **吞吐**：单位时间处理/生成的 token 总数；
- **显存占用**：模型权重、KV Cache、临时激活和 CUDA Graph 等。

更大 batch 往往提高吞吐，却可能增加单请求排队或延迟。不能只用一个 tokens/s 数字代表所有体验。

Nano-vLLM 的 `bench.py` 主要测离线总生成吞吐，不等价于在线服务的尾延迟测试。

---

## 16. 源码阅读顺序

1. [`sequence.py`](../nanovllm/engine/sequence.py)：请求有哪些状态数据；
2. [`llm_engine.py`](../nanovllm/engine/llm_engine.py)：请求入口与 step 循环；
3. [`scheduler.py`](../nanovllm/engine/scheduler.py)：prefill/decode 选择和状态迁移；
4. [`block_manager.py`](../nanovllm/engine/block_manager.py)：物理 block 生命周期和前缀复用；
5. [`model_runner.py`](../nanovllm/engine/model_runner.py)：输入元数据如何构造；
6. [`attention.py`](../nanovllm/layers/attention.py)：K/V 何时写入和读取；
7. [`sampler.py`](../nanovllm/layers/sampler.py)：从 logits 到 token。

每看一个函数记录四项：输入、输出、修改的状态、需要/释放的资源。

---

## 17. 动手练习

配套脚本模拟 block table、slot mapping、chunked prefill 和 decode，不需要 GPU：

```bash
python ai_background_knowledge/examples/04_llm_inference.py
```

手算题：block size 为 4，请求 prompt 长 6，block table 为 `[3, 7]`。

1. prompt 六个 token 的 slots 是什么？
2. prefill 后采样 token 6 并 append，下一轮 decode 的 position、context length、slot 是什么？
3. 再 append 一个 token 后是否需要新 block？

<details>
<summary>答案</summary>

1. 物理 block 3 的 slots 12～15，物理 block 7 的 slots 28～29。
2. Sequence 长度变为 7；下一轮输入刚生成 token，position 为 6，context length 为 7，写 slot 为 `7*4+2=30`。
3. 长度变为 8 时仍填在 block 7 的最后一个位置；再 append 第 9 个 token 时才需要第三个 block。调度器会在准备写这个新 token 前保证 block 已分配。

</details>

## 18. 自测清单

- [ ] 能区分字符串长度、token 数和上下文长度。
- [ ] 能写出自回归生成循环。
- [ ] 能解释 prefill 与 decode 的输入和瓶颈差异。
- [ ] 能说明 KV Cache 缓存什么、不缓存什么。
- [ ] 能画出 block table 并计算 slot。
- [ ] 能解释 prefix cache 为什么只复用完整 block。
- [ ] 能解释引用计数的作用。
- [ ] 能画出 WAITING、RUNNING、FINISHED 和 preempt。
- [ ] 能区分 `num_cached_tokens` 与 `num_scheduled_tokens`。
- [ ] 能解释压平变长序列和 `cu_seqlens`。

下一章会补齐运行这些机制的硬件背景：GPU 显存层次、kernel launch、同步、NCCL、Tensor Parallel、FlashAttention、Triton 和 CUDA Graph。

