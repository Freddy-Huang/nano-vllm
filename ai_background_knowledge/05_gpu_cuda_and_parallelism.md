# 05｜GPU、CUDA 与并行基础：读懂 Nano-vLLM 的性能优化

> 本章目标是建立足够准确的硬件心智模型。读完应能解释“为什么这样更快”，不要求立即会写 CUDA/Triton kernel。

## 本章目标

学完后，你应该能：

- 区分 CPU 内存、GPU 显存和 pinned memory；
- 解释 GPU kernel、并行线程、launch overhead 与同步；
- 判断一段推理更可能受计算、带宽还是启动开销限制；
- 解释 Tensor Parallel 的 Column/Row/Vocab 切分与 collective；
- 说明 FlashAttention、Triton、`torch.compile` 和 CUDA Graph 各优化什么；
- 看懂 Nano-vLLM 如何预算 KV Cache 显存和组织多 GPU 进程。

## 重点地图

| 优先级 | 知识 | Nano-vLLM 对应位置 |
| --- | --- | --- |
| 必会 | 显存组成、数据传输、异步执行 | `model_runner.py` |
| 必会 | kernel launch、同步、带宽 | 所有性能优化的基础 |
| 必会 | Tensor Parallel 与 NCCL collective | `layers/linear.py`、`embed_head.py` |
| 看懂即可 | FlashAttention | `layers/attention.py` |
| 看懂即可 | `torch.compile`、CUDA Graph | 各层与 `capture_cudagraph()` |
| 最后再学 | Triton kernel 索引 | `store_kvcache_kernel` |

---

## 1. CPU 与 GPU 为什么分工不同

可以先用一个不完全但有用的比喻：

- CPU 像少量能力很强、擅长复杂控制的工人；
- GPU 像大量擅长同时做相似算术的工人。

大模型中的矩阵乘法会对海量元素执行规则一致的乘加，非常适合 GPU。请求队列、字符串、Python 对象和分支复杂的调度更适合 CPU。

Nano-vLLM 的分工：

```text
CPU:
  tokenizer、Sequence、Scheduler、BlockManager
  准备 input_ids / positions / block tables

GPU:
  Embedding、Attention、MLP、LM Head、采样
  实际 KV Cache 数据
```

CPU 上的 `BlockManager` 只管理 block ID、hash 和引用计数等元数据；真正的大块 K/V Tensor 在 GPU。

---

## 2. 两套内存与数据传输

在常见独立显卡系统中：

```text
CPU RAM <---- PCIe / interconnect ----> GPU VRAM
```

Tensor 在哪台设备上，运算就由哪台设备执行。把数据传过总线通常比 GPU 内部访问显存慢，因此应减少来回搬运。

### pinned memory

普通 CPU 内存页可能被操作系统换出。Pinned（page-locked）memory 固定在物理内存中，GPU DMA 传输更高效，也能配合 non-blocking copy：

```python
cpu = torch.tensor(data, pin_memory=True)
gpu = cpu.cuda(non_blocking=True)
```

`non_blocking=True` 表示允许异步，不保证调用一返回数据就已经复制完成。后续同一 CUDA stream 中依赖它的操作会遵守顺序。

Pinned memory 是有限资源，不应该把所有普通数据都固定。

### Nano-vLLM 的传输路径

每个 step，Scheduler 在 CPU 选择 Sequence；ModelRunner 把小型 Python 列表整理为 pinned Tensor，再复制到当前 GPU。模型权重和 KV Cache 一直驻留 GPU，不会每轮搬回 CPU。

---

## 3. GPU 显存花在哪里

推理显存主要包括：

```text
模型权重
+ KV Cache
+ 临时激活 / kernel workspace
+ CUDA context、通信 buffer、CUDA Graph 内存池等
```

训练还要保存梯度、优化器状态和更多激活；纯推理使用 `inference_mode` 后不需要这些。

### 权重粗算

若模型有 `P` 个参数，每参数 `B` 字节：

```text
权重显存约为 P × B
```

0.6B 参数、bfloat16/float16（2 字节）仅权重约 1.2 GB，实际进程还会有额外开销。

### KV block 粗算

每个物理 KV block：

```text
bytes = 2 × layers × block_size × kv_heads_per_gpu × head_dim × dtype_bytes
```

2 表示 K 与 V。总 KV Cache 再乘 `num_blocks`。

Nano-vLLM 在 warmup 后读取：

- GPU 总/空闲显存；
- 当前分配量；
- warmup 峰值；
- `gpu_memory_utilization` 允许使用的总比例。

剩余预算除以每 block 字节数，得到 `num_kvcache_blocks`。这不是模型理论常量，而取决于硬件、配置和运行时开销。

---

## 4. Kernel：GPU 上的一次并行程序

GPU 不直接执行 Python 循环。PyTorch 的一个 Tensor 操作通常会发起一个或多个 GPU kernel：

```python
y = torch.softmax(x, dim=-1)  # CPU 发起命令，GPU kernel 执行
```

kernel 中大量线程处理不同数据。CUDA 用 grid / block / thread 组织它们；Triton 更常让一个 program instance 处理一个数据 tile。

第一次学习只需记住两种“block”完全不同：

- **CUDA thread block**：一组协作 GPU 线程；
- **Paged KV Cache block**：一段 token 的 K/V 存储。

它们只是碰巧同名。

### Kernel launch overhead

CPU 发起 kernel 有固定开销。大矩阵一次计算很久时，开销占比小；decode 的 batch/矩阵较小时，许多小 kernel 的启动开销可能很显著。

这解释了为什么项目会：

- 融合 residual add + RMSNorm；
- 合并 gate/up 或 Q/K/V 投影；
- 用 `torch.compile` 融合逐元素操作；
- 用 CUDA Graph 重放完整 decode 路径。

---

## 5. 异步执行与同步

CUDA 操作通常对 CPU 是异步的：

```python
start = time.perf_counter()
y = x.cuda() @ w.cuda()
elapsed = time.perf_counter() - start
```

这个时间可能只覆盖命令提交，不包含 GPU 真正完成计算。准确测量 GPU 区间需要 CUDA events，或在边界调用：

```python
torch.cuda.synchronize()
```

同步会让 CPU 等 GPU，放在热路径会损害并行度，因此只在确有依赖、计时或资源释放时使用。

常见隐式同步包括把 CUDA Tensor `.item()`、`.cpu()` 或 `.tolist()`。Nano-vLLM 采样后 `.tolist()` 必须把少量 token IDs 交回 CPU 调度器，这是一处必要边界。

### Stream 的最低限度理解

CUDA stream 是有顺序的命令队列。同一 stream 内操作按提交顺序执行，不同 stream 可能重叠。Nano-vLLM 大多依赖 PyTorch 默认 stream；初读不必展开 stream 编程。

---

## 6. 计算受限、带宽受限与启动受限

性能瓶颈可先分三类：

- **计算受限**：算术很多，计算单元忙；
- **显存带宽受限**：主要时间花在搬数据，计算单元吃不饱；
- **launch/CPU 受限**：GPU 工作很小，发命令和 Python 调度占比高。

prefill 的大矩阵乘法通常更容易充分利用计算单元；decode 每轮每请求只有一个 query，却要读取大量模型权重和历史 KV，常更偏内存/启动受限。

不能只凭操作名称下结论，实际还取决于 batch size、序列长度、GPU 型号、dtype 和 kernel 实现。profiling 才是最终证据。

---

## 7. Fusion：少读写几次，少启动几个 kernel

未融合：

```text
kernel 1: residual add -> 写显存
kernel 2: square/mean -> 读写显存
kernel 3: normalize/scale -> 再读写
```

融合后，一个 kernel 可以把中间值保留在更快的寄存器/片上存储中，并减少 launch。

项目中的例子：

- `RMSNorm.add_rms_forward`：残差加法 + RMSNorm；
- `SiluAndMul`：SiLU + 逐元素乘；
- `qkv_proj`：Q/K/V 合并权重；
- `gate_up_proj`：gate/up 合并权重。

合并线性层并不表示数学上把 Q、K、V 混在一起；只是把三个输出沿最后一维拼接，一次矩阵乘法后再 split。

---

## 8. FlashAttention 优化什么

朴素 Attention 会显式产生大的 score/probability 矩阵：

```text
[heads, query_len, key_len]
```

长序列时，反复把这个矩阵写入/读出显存代价很高。FlashAttention 通过分块计算和在线 softmax，让中间块尽量留在片上存储，减少高带宽显存读写，并保持数学结果等价（允许正常浮点误差）。

Nano-vLLM 使用两个接口：

- `flash_attn_varlen_func`：prefill，支持压平的变长序列和 causal；
- `flash_attn_with_kvcache`：decode，直接读取 paged KV Cache。

FlashAttention 不是 KV Cache。前者优化 Attention kernel 的执行，后者避免重算历史 K/V；两者解决不同问题，并一起使用。

---

## 9. Triton：用 Python 风格写 GPU kernel

Triton 是一种 GPU kernel 编程语言/编译器。项目只用一个自定义 Triton kernel 把新 K/V 写到指定 slot。

阅读 `store_kvcache_kernel` 的方法：

1. `tl.program_id(0)`：当前 program 处理第几个新 token；
2. 从 `slot_mapping[idx]` 读目标 slot；
3. `D = num_kv_heads * head_dim`；
4. 加载这个 token 的连续 K/V 向量；
5. 写到 cache 的 `slot * D` 开始处。

概念伪代码：

```python
for token_index in parallel:
    slot = slot_mapping[token_index]
    k_cache[slot] = key[token_index]
    v_cache[slot] = value[token_index]
```

真正 kernel 使用 pointer + stride 做地址运算，所以调用前 assert 最后一维布局连续。刚入门时能把指针代码还原成这段伪代码就足够。

---

## 10. `torch.compile`：编译与融合 PyTorch 运算

`@torch.compile` 会捕获 PyTorch 运算图，经过编译和优化生成更高效代码：

```python
@torch.compile
def forward(...):
    ...
```

适合项目中的 RMSNorm、RoPE、SiLU+Mul 和 Sampler 等操作组合。常见权衡：

- 首次调用有编译开销；
- shape、分支或输入属性变化可能触发重新编译；
- 稳定重复执行后才更可能收回成本；
- 调试编译问题比 eager 模式复杂。

这就是学习路线建议先 `enforce_eager=True`：它只关闭 CUDA Graph 路径；源码装饰的 `torch.compile` 仍可能由 PyTorch 编译机制处理，二者不是同一开关。

---

## 11. CUDA Graph：录制并重放固定工作流

普通 decode 每轮 CPU 都要逐个发起 kernel。CUDA Graph 可以把一串 GPU 操作捕获成图，之后一次 replay：

```text
普通：CPU launch A -> B -> C -> D
Graph：CPU graph.replay() -> GPU 执行 A/B/C/D
```

它主要减少 CPU launch overhead，不会改变模型数学。

### 为什么要求静态

捕获时会固定许多内容：

- Tensor 地址；
- shape；
- 执行路径；
- 某些内存分配行为。

因此 Nano-vLLM：

1. 为多个 batch size 分别捕获图；
2. 准备固定地址的 `graph_vars`；
3. 实际 batch 找到不小于它的已捕获 size；
4. 把真实输入复制到前 `bs` 行；
5. 其余 slot mapping 填 `-1`、长度清零作为 padding；
6. replay 并只取前 `bs` 个输出。

prefill shape 变化大，项目始终走普通 eager 路径；decode 更规律，适合 CUDA Graph。batch size 大于 512 时也回到 eager。

`enforce_eager=True` 是正确性调试的好起点；关闭后初始化会因 capture 更慢，但稳定 decode 可能更快并额外占用图内存。

---

## 12. 多 GPU 的进程模型

Nano-vLLM 采用“一进程控制一张 GPU”：

```text
rank 0 process -> GPU 0
rank 1 process -> GPU 1
...
world_size = tensor_parallel_size
```

每个 rank 构造同样的模型结构，但参数是分片 shape。`torch.distributed.init_process_group("nccl", ...)` 建立 GPU collective 通信组。

rank 0 还负责主调度。它通过共享内存发送方法名和轻量参数，用 Event 唤醒其他进程；所有 rank 随后在同样顺序执行模型，并在层内调用 NCCL collective。

两个通信层次不要混淆：

- Python SharedMemory/Event：告诉工作进程“现在调用什么”；
- NCCL collective：真正合并 GPU Tensor 计算结果。

---

## 13. Tensor Parallel：把一层拆到多张 GPU

Data Parallel 是每张 GPU 放完整模型、处理不同请求；Tensor Parallel 是把同一个模型层的 Tensor 切到多张 GPU。Nano-vLLM 实现后者。

### Column Parallel Linear：切输出维

权重：

```text
W: [out_features, in_features]
```

两张 GPU 沿第 0 维切：

```text
GPU 0: W0 [out/2, in] -> y0 [T, out/2]
GPU 1: W1 [out/2, in] -> y1 [T, out/2]
```

两者逻辑拼接就是完整输出。QKV 与 MLP gate/up 使用这种切法；下一步计算也能在各自分片上继续，暂时不必通信。

### Row Parallel Linear：切输入维

权重沿第 1 维切，同时输入也已分片：

```text
GPU 0: x0 [T, in/2], W0 [out, in/2] -> partial y0 [T, out]
GPU 1: x1 [T, in/2], W1 [out, in/2] -> partial y1 [T, out]
```

完整结果：

```text
y = y0 + y1
```

所以使用 `dist.all_reduce(y)`。每个 rank 最终都有完整 y，方便残差连接和下一层归一化。

### 一对搭配

```text
完整 hidden
  -> Column Parallel（产生分片中间维，无通信）
  -> 各 rank 独立激活/Attention
  -> Row Parallel（局部投影）
  -> all_reduce（恢复完整 hidden）
```

这套搭配用于 Attention 的 qkv/o_proj 和 MLP 的 gate_up/down_proj。

---

## 14. Vocab Parallel Embedding 与 LM Head

词表权重 `[V, H]` 沿词表维切分。

### Embedding

每张 GPU 只保存一段 token ID 范围：

```text
GPU 0: IDs [0, V/2)
GPU 1: IDs [V/2, V)
```

各 rank：

1. mask 出属于自己的 token；
2. 非本地 token 输出置零；
3. `all_reduce(sum)`。

由于每个 token 只在一个 rank 有非零结果，求和后所有 rank 得到完整 embedding。

### LM Head

每张 GPU 计算自己词表分片的 logits：

```text
local logits: [batch, V/tp]
```

采样需要完整词表分布，所以 gather 到 rank 0，再沿最后一维拼成 `[batch, V]`。只有 rank 0 执行采样和 CPU 调度。

---

## 15. Collective 的成本与正确性

多卡并不自动线性加速。收益要扣除：

- `all_reduce` / `gather` 通信；
- 多进程同步；
- 每卡工作太小时利用率下降；
- PCIe、NVLink 等互连差异。

Tensor Parallel 常用于单卡放不下模型，或模型计算足够大时降低延迟。小模型在普通 PCIe 多卡上反而可能更慢。

所有 rank 必须按相同顺序进入 collective，并使用兼容 shape；某个 rank 分支不同或异常退出，其他 rank 可能一直等待。

---

## 16. Warmup 为什么重要

首次模型执行可能包含：

- CUDA context 与库初始化；
- kernel/JIT 编译；
- allocator 建立缓存；
- 算法选择和 workspace 分配；
- CUDA Graph capture。

所以首次耗时不代表稳定运行速度。项目的 warmup 还承担一个关键任务：用最大规模请求测到临时内存峰值，再把剩余显存安全地分给 KV Cache。

benchmark 通常应先 warmup，再同步并计时，记录完整配置。`bench.py` 先生成一次 `"Benchmark: "` 就是预热。

---

## 17. OOM 与常见排查顺序

发生 CUDA out of memory 时，可先确认：

1. 模型权重是否适合单卡，是否需要 Tensor Parallel；
2. `gpu_memory_utilization` 是否过高；
3. `max_num_batched_tokens` / `max_num_seqs` 是否导致 warmup 激活过大；
4. `max_model_len` 是否让 CUDA Graph block table 或运行需求过大；
5. 其他进程是否占显存；
6. eager 能否运行，Graph capture 是否增加额外内存；
7. dtype 是否符合预期。

不要看到 OOM 就只调用 `empty_cache()`。它只能释放 PyTorch allocator 中未被活跃 Tensor 使用的缓存，不能释放仍被模型、KV Cache 或 Graph 引用的内存。

---

## 18. 源码阅读顺序

1. [`model_runner.py`](../nanovllm/engine/model_runner.py) 的 `__init__`：进程组、设备、dtype；
2. `warmup_model()` 与 `allocate_kv_cache()`：显存预算；
3. [`linear.py`](../nanovllm/layers/linear.py)：Column/Row 切分；
4. [`embed_head.py`](../nanovllm/layers/embed_head.py)：词表并行；
5. [`model_runner.py`](../nanovllm/engine/model_runner.py) 的共享内存 `call/loop`；
6. `capture_cudagraph()` 与 `run_model()`：capture/replay 选择；
7. [`attention.py`](../nanovllm/layers/attention.py)：FlashAttention 与 Triton 存 cache；
8. [`loader.py`](../nanovllm/utils/loader.py)：完整权重如何装入分片 Parameter。

阅读性能代码时分别写下：数学结果是否改变、减少了哪类成本、引入了什么约束。

---

## 19. 动手练习

配套脚本只做显存和 Tensor Parallel shape 推导，不需要 GPU：

```bash
python ai_background_knowledge/examples/05_gpu_parallelism.py
```

练习：

1. 某模型 28 层、block size 256、每 GPU 4 个 KV heads、head dim 128、bf16。算一个 KV block 字节数和 MiB。
2. `W=[4096, 2048]` 用 4 卡 Column Parallel，每卡权重 shape 是什么？
3. 同一 W 用 Row Parallel，每卡权重和输入分片 shape 是什么？结果为何要相加？
4. 解释 `torch.compile` 和 CUDA Graph 的区别。
5. 解释 SharedMemory/Event 与 NCCL 的区别。

<details>
<summary>部分答案</summary>

1. `2*28*256*4*128*2 = 14,680,064` 字节，即 14 MiB。
2. 每卡 `[1024, 2048]`。
3. 每卡权重 `[4096, 512]`，输入最后一维从 2048 切成 512；完整线性结果等于各输入分片贡献之和。
4. `torch.compile` 编译/融合 PyTorch 运算；CUDA Graph 录制并重放相对静态的一串 GPU 工作，主要减少 launch 开销。两者可以同时使用。
5. 前者传 CPU 控制命令，后者在 GPU 间合并 Tensor。

</details>

## 20. 自测清单

- [ ] 能列出推理显存的主要组成。
- [ ] 能解释 pinned memory 和 non-blocking copy。
- [ ] 能解释 kernel launch 与 GPU 异步执行。
- [ ] 能区分计算、带宽和启动受限。
- [ ] 能说明 fusion 减少了什么。
- [ ] 能区分 FlashAttention、KV Cache、Triton、`torch.compile` 和 CUDA Graph。
- [ ] 能手算一个 KV block 的字节数。
- [ ] 能画出 Column Parallel 与 Row Parallel。
- [ ] 能解释 Embedding 的 all-reduce 和 LM Head 的 gather。
- [ ] 能解释 warmup 对显存预算和 benchmark 的意义。

完成这章后，你已具备阅读 Nano-vLLM 全部主流程的背景知识。Triton 指针、FlashAttention 算法证明、NCCL 通信算法和 CUDA stream 编程可以等主调用链读通后再深入。

