# Nano-vLLM 学习路线

这份路线适合通过阅读、运行、画数据流和添加注释来学习 Nano-vLLM。项目代码量不大，但同时包含 Transformer 推理、请求调度、Paged KV Cache、Prefix Cache、Tensor Parallel、FlashAttention 和 CUDA Graph。推荐沿着一次生成请求的实际调用链学习，不要一开始就钻进 CUDA/Triton 细节。

## 1. 学习目标

完成这份路线后，应当能够回答以下问题：

- 一个 prompt 如何从字符串变成 token，并逐 token 生成结果？
- prefill 和 decode 有什么区别？各自输入多少 token？
- Scheduler 如何在 waiting、running 和 finished 之间移动请求？
- KV Cache 为什么按 block 管理？block table、slot mapping 分别是什么？
- Prefix Cache 如何复用已经计算过的完整 token block？
- Qwen3 的 Attention、MLP、RMSNorm 和 RoPE 如何连接？
- Tensor Parallel 如何切分权重，多个 GPU 如何合并结果？
- CUDA Graph 和 `torch.compile` 优化了哪部分开销？

## 2. 开始前的知识准备

不必全部学完再读项目。遇到相关代码时，按需补充以下知识：

- Python：类、继承、dataclass、property、迭代器和 multiprocessing。新手可以先学习 [面向 Nano-vLLM 的 Python 入门](ai_background_knowledge/01_python_for_nano_vllm.md)，并运行配套的纯 CPU 练习。
- PyTorch：Tensor shape、`nn.Module`、矩阵乘法、softmax、distributed collective。参见 [PyTorch 与 Tensor：先学会追踪 shape](ai_background_knowledge/02_pytorch_and_tensors.md)。
- Transformer：Embedding、Self-Attention、MLP、残差连接、RMSNorm、RoPE。参见 [Transformer 基础：跟着一个 token 走完 Qwen3](ai_background_knowledge/03_transformer_basics.md)。
- 大模型推理：tokenizer、autoregressive generation、prefill、decode、KV Cache。参见 [大模型推理基础：从 prompt 到逐 token 生成](ai_background_knowledge/04_llm_inference_basics.md)。
- CUDA 基础：GPU 显存、kernel launch；Triton 和 CUDA Graph 可以最后再学。参见 [GPU、CUDA 与并行基础](ai_background_knowledge/05_gpu_cuda_and_parallelism.md)。

本项目运行依赖 NVIDIA GPU、CUDA/NCCL、FlashAttention 和 Triton。没有合适 GPU 时仍然可以完成静态阅读、画调用链以及为调度器和块管理器编写纯 CPU 单元测试。

### 2.1 创建 Conda 学习环境

项目支持 Python `>=3.10,<3.13`。推荐使用 Python 3.11，并将环境统一命名为 `vllm`：

```bash
conda create -n vllm python=3.11 pip -y
conda activate vllm
python -m pip install --upgrade pip
```

如果只运行 `ai_background_knowledge/examples/` 下的 CPU 练习，安装 CPU 版 PyTorch 即可：

```bash
python -m pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu
```

不要为 CPU 背景练习执行 `python -m pip install -e .`，因为它会安装 Nano-vLLM 声明的 Triton、FlashAttention 等 GPU 依赖。更完整的命令、环境验证方式和示例运行条件参见 [AI 背景知识目录](ai_background_knowledge/README.md#conda-环境准备)。

### 2.2 哪些示例需要 GPU

| 示例 | 运行设备 | 额外条件 |
| --- | --- | --- |
| `ai_background_knowledge/examples/01_python_basics.py` | CPU | 仅标准库 |
| `ai_background_knowledge/examples/02_pytorch_tensors.py` | CPU | CPU 版 PyTorch |
| `ai_background_knowledge/examples/03_transformer_basics.py` | CPU | CPU 版 PyTorch |
| `ai_background_knowledge/examples/04_llm_inference.py` | CPU | 仅标准库 |
| `ai_background_knowledge/examples/05_gpu_parallelism.py` | CPU | 仅标准库；不执行 CUDA |
| `example.py` | **NVIDIA GPU** | CUDA/NCCL、CUDA 版 PyTorch、Triton、FlashAttention、模型权重 |
| `bench.py` | **NVIDIA GPU** | 与 `example.py` 相同，且基准负载更大 |

因此，没有 NVIDIA GPU 时可以完整学习五篇背景知识并运行其全部配套脚本，但不能运行项目根目录的 `example.py` 和 `bench.py`。PyTorch 的 CUDA 安装命令取决于机器驱动与 CUDA 环境，应通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 获取。

## 3. 项目地图

| 文件 | 作用 | 建议顺序 |
| --- | --- | ---: |
| `example.py` | 最小使用示例，展示公开 API | 1 |
| `nanovllm/config.py` | 引擎配置与模型配置加载 | 2 |
| `nanovllm/sampling_params.py` | 采样参数 | 2 |
| `nanovllm/engine/llm_engine.py` | 请求入口和生成主循环 | 3 |
| `nanovllm/engine/sequence.py` | 单个请求的状态与 token 数据 | 4 |
| `nanovllm/engine/scheduler.py` | prefill/decode 调度、抢占和结束判断 | 5 |
| `nanovllm/engine/block_manager.py` | KV Cache block 分配、释放和前缀复用 | 6 |
| `nanovllm/engine/model_runner.py` | GPU 初始化、输入整理、执行模型、CUDA Graph | 7 |
| `nanovllm/models/qwen3.py` | Qwen3 模型结构 | 8 |
| `nanovllm/layers/` | Attention、并行线性层、RoPE、RMSNorm、采样器 | 9 |
| `nanovllm/utils/loader.py` | safetensors 权重加载与合并权重映射 | 10 |
| `bench.py` | 吞吐量基准测试 | 11 |

`nanovllm/llm.py` 中的 `LLM` 只是 `LLMEngine` 的空子类，用于提供简洁、接近 vLLM 的公开 API，不是核心逻辑所在。

## 4. 先记住这条主调用链

```text
example.py
  -> LLM(...)
     -> LLMEngine.__init__()
        -> Config
        -> ModelRunner
        -> Scheduler
  -> LLM.generate()
     -> add_request()
        -> tokenizer.encode()
        -> Sequence
        -> Scheduler.add()
     -> 循环调用 step()
        -> Scheduler.schedule()
        -> ModelRunner.run()
           -> prepare_prefill() 或 prepare_decode()
           -> Qwen3ForCausalLM.forward()
           -> compute_logits()
           -> Sampler
        -> Scheduler.postprocess()
     -> tokenizer.decode()
```

整个推理循环可以先简化成四步：调度请求、准备张量、执行模型并采样、更新请求状态。

## 5. 分阶段学习步骤

### 阶段 0：确认独立的学习分支

- [ ] 查看当前分支：`git branch --show-current`
- [ ] 确认工作区状态：`git status`
- [ ] 如果输出已经是 `annotations`，直接在当前分支学习，不要重复创建。
- [ ] 如果仍在 `main`，再执行：`git switch -c annotations`
- [ ] 每完成一个阶段做一次小提交，避免把大量注释混在一个提交中。

推荐提交方式：

```bash
git add STUDY_GUIDE.md nanovllm/engine/llm_engine.py
git commit -m "docs: annotate generation workflow"
```

### 阶段 1：运行最小示例，理解公开 API

阅读顺序：

1. `README.md`
2. `example.py`
3. `nanovllm/__init__.py`
4. `nanovllm/llm.py`
5. `nanovllm/config.py`
6. `nanovllm/sampling_params.py`

重点问题：

- `LLM` 构造时真正执行的是哪个类的初始化函数？
- `Config.__post_init__()` 做了哪些限制和自动修正？
- prompt 在示例中为什么先经过 `apply_chat_template()`？
- `SamplingParams` 的三个字段分别在何处被使用？

实践：

- [ ] 把示例中的模型目录改成自己的 Qwen3 模型目录。
- [ ] 先使用 `enforce_eager=True`，避开 CUDA Graph，降低调试复杂度。
- [ ] 把 `max_tokens` 改为较小值，例如 8，观察输出。
- [ ] 分别传入字符串 prompt 和 token ID 列表，确认两种入口的差别。

安装与运行命令：

```bash
python -m pip install -e .
python example.py  # 需要 NVIDIA GPU
```

### 阶段 2：读懂生成主循环

核心文件：`nanovllm/engine/llm_engine.py`

建议按以下方法阅读：

1. 从 `generate()` 开始，找出请求何时加入 Scheduler。
2. 跟进 `step()`，记录 Scheduler 和 ModelRunner 的输入输出。
3. 回到 `__init__()`，了解 tokenizer、子进程和 runner 的创建时机。
4. 最后读 `exit()`，理解多进程资源如何释放。

重点结论：

- `generate()` 是同步离线推理接口，会一直循环到所有请求完成。
- `step()` 每次执行一轮 prefill 或一轮 decode。
- `num_tokens > 0` 表示本轮是 prefill；负数表示 decode 的序列数量。
- 最终结果先按 `seq_id` 排序，再由 tokenizer 解码。

实践：

- [ ] 画出 `generate()` 的 while 循环。
- [ ] 给 `add_request()`、`step()`、`generate()` 添加说明“输入、输出、状态变化”的注释。
- [ ] 临时打印每轮 `is_prefill`、序列数和 token 数，观察阶段切换；实验后撤销打印。

### 阶段 3：理解 Sequence 和 Scheduler

阅读顺序：

1. `nanovllm/engine/sequence.py`
2. `nanovllm/engine/scheduler.py`

先记住状态转换：

```text
新增请求                 prompt 预填充完成
WAITING ---------------------------------> RUNNING
   ^                                          |
   | KV Cache 不足时抢占                       | EOS 或达到 max_tokens
   +------------------------------------------+----> FINISHED
```

重点问题：

- `num_prompt_tokens`、`num_cached_tokens`、`num_scheduled_tokens` 有何区别？
- 为什么调度器优先处理 waiting 队列中的 prefill？
- chunked prefill 为什么只允许当前 batch 的第一个序列被切分？
- decode 阶段为什么每个序列只调度一个 token？
- 显存块不足时，`preempt()` 为什么会清空被抢占序列的 block table？
- `Sequence.__getstate__()` 为什么只传 prompt tokens 或 last token？

建议用一个小例子手算：block size 假设为 4，prompt 长度分别为 3、6、10，计算每个 Sequence 的 `num_blocks`、`last_block_num_tokens` 和每轮被调度的 token 数。

实践：

- [ ] 为状态字段添加“含义和更新时机”注释。
- [ ] 不依赖 GPU，为 Sequence 的 block 计算编写简单断言。
- [ ] 用纸笔模拟两个请求从 waiting 到 finished 的过程。

### 阶段 4：理解 Paged KV Cache 和 Prefix Cache

核心文件：`nanovllm/engine/block_manager.py`

核心对象：

- `Block`：一个物理 KV Cache 块的元数据，保存 block ID、引用计数、hash 和 token IDs。
- `Sequence.block_table`：该请求的逻辑 block 顺序到物理 block ID 的映射。
- `free_block_ids` / `used_block_ids`：物理块的空闲与占用集合。
- `hash_to_block_id`：根据完整 token block 的链式 hash 查找可复用前缀。

阅读顺序：

1. `_allocate_block()` / `_deallocate_block()`：理解物理块生命周期。
2. `allocate()` / `deallocate()`：理解一个 Sequence 如何持有多个块。
3. `can_append()` / `may_append()`：理解 decode 时何时增加新块。
4. `compute_hash()` / `hash_blocks()` / `can_allocate()`：理解 Prefix Cache。

注意：只有已经填满并完成 hash 的 block 才适合作为稳定前缀复用；hash 还会串联上一个 block 的 hash，因此相同 token block 出现在不同前缀后不一定是同一个缓存项。

实践：

- [ ] 将 block size 在纸上缩小为 4，模拟分配、共享、引用计数递增、释放。
- [ ] 写一个不需要 GPU 的 BlockManager 小测试，验证两个相同前缀请求能复用 block。
- [ ] 解释为什么 hash 命中后还要比较 `token_ids`。

### 阶段 5：区分 prefill 和 decode 的张量准备

核心文件：`nanovllm/engine/model_runner.py`

先只阅读这些函数：

1. `run()`
2. `prepare_prefill()`
3. `prepare_decode()`
4. `prepare_block_tables()`
5. `run_model()`

prefill 的特点：

- 每个请求可能一次送入多个尚未缓存的 token。
- 多个变长请求被压平，通过 `cu_seqlens_q` / `cu_seqlens_k` 标记边界。
- `positions` 是每个 token 在原序列中的位置。
- `slot_mapping` 指定新生成的 K/V 要写入哪个物理 cache slot。
- 存在缓存前缀时，query 长度可能小于 key/value 的上下文长度，并传入 `block_tables`。

decode 的特点：

- 每个运行中请求只输入 `last_token`。
- `context_lens` 表示各请求当前完整上下文长度。
- `block_tables` 让 attention 找到每个请求散布在物理 KV Cache 中的块。

建议为每个张量记录 dtype 和概念 shape，例如：

| 张量 | prefill 概念 shape | decode 概念 shape |
| --- | --- | --- |
| `input_ids` | `[本轮所有新 token 数]` | `[batch_size]` |
| `positions` | `[本轮所有新 token 数]` | `[batch_size]` |
| `slot_mapping` | `[本轮所有新 token 数]` | `[batch_size]` |
| `block_tables` | `[batch_size, max_blocks]`，仅需要时使用 | `[batch_size, max_blocks]` |
| `context_lens` | 不使用 | `[batch_size]` |

实践：

- [ ] 用两个不同长度的 Sequence 手算 `prepare_prefill()` 产生的列表。
- [ ] 解释 prefix cache 情况下为什么 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`。
- [ ] 解释 decode 的 slot 计算公式。

### 阶段 6：理解 Qwen3 前向计算

阅读顺序：

1. `nanovllm/models/qwen3.py` 中的 `Qwen3Model`
2. `Qwen3DecoderLayer`
3. `Qwen3Attention`
4. `Qwen3MLP`
5. `Qwen3ForCausalLM`
6. `nanovllm/layers/embed_head.py`
7. `nanovllm/layers/rotary_embedding.py`
8. `nanovllm/layers/layernorm.py`
9. `nanovllm/layers/activation.py`

前向主线：

```text
token IDs
  -> Embedding
  -> N × [RMSNorm -> Attention -> 残差 -> RMSNorm -> MLP -> 残差]
  -> Final RMSNorm
  -> LM Head
  -> logits
  -> Sampler
  -> next token
```

阅读时持续标注 tensor shape。以 `T` 表示本轮 token 总数、`H` 表示 hidden size：

- hidden states：`[T, H]`
- q：`[T, num_heads_per_gpu, head_dim]`
- k/v：`[T, num_kv_heads_per_gpu, head_dim]`
- logits：rank 0 上为 `[batch_size, vocab_size]`

重点问题：

- GQA 中 query head 数和 KV head 数为什么可以不同？
- RoPE 为什么作用于 q 和 k，而不是 v？
- prefill 时 LM Head 为什么只选择各序列最后一个位置？
- `SiluAndMul` 如何实现 gated MLP？
- fused residual + RMSNorm 为什么能减少内存读写？

### 阶段 7：理解 Attention 和 KV Cache 的连接点

核心文件：`nanovllm/layers/attention.py`

阅读顺序：

1. `Attention.forward()` 的 prefill 分支。
2. `Attention.forward()` 的 decode 分支。
3. `store_kvcache()` 的输入约束。
4. 最后再看 `store_kvcache_kernel()` 的 Triton 索引计算。

把 `utils/context.py` 与这里一起阅读。ModelRunner 将调度相关元数据写入全局 Context，Attention 层再读取它，从而避免把大量参数逐层传过 Qwen3 模型。

实践：

- [ ] 找出 K/V 写入 cache 的时机。
- [ ] 对照 FlashAttention 两个函数的参数，说明 prefill 和 decode 调用差异。
- [ ] 手算一个 token 的 `slot_mapping` 如何变成 KV Cache 中的物理位置。

### 阶段 8：理解 Tensor Parallel 和权重加载

阅读顺序：

1. `nanovllm/layers/linear.py`
2. `nanovllm/layers/embed_head.py`
3. `nanovllm/utils/loader.py`
4. `nanovllm/engine/model_runner.py` 的初始化、共享内存与 `call()`。

重点问题：

- Column Parallel 为什么切输出维度？
- Row Parallel 为什么切输入维度并在最后 `all_reduce()`？
- Q、K、V 权重如何装入合并后的 `qkv_proj`？
- gate 和 up 权重如何装入 `gate_up_proj`？
- 词表 embedding 如何按 GPU 切分？LM Head 的 logits 如何聚合到 rank 0？
- rank 0 如何通过共享内存和 Event 通知其他 rank 执行同一个方法？

建议先以 `tensor_parallel_size=1` 理解模型，再假设为 2 手算权重 shape，最后才实际多卡运行。

### 阶段 9：最后学习性能优化

阅读内容：

- `ModelRunner.warmup_model()`：测量模型峰值显存前先执行最大规模请求。
- `ModelRunner.allocate_kv_cache()`：根据显存预算计算可分配 block 数。
- `ModelRunner.capture_cudagraph()`：为多个 decode batch size 捕获图。
- `run_model()`：prefill、eager decode 和 CUDA Graph decode 的选择逻辑。
- 各层上的 `@torch.compile`。
- `bench.py`：吞吐量测量方法。

实验顺序：

- [ ] `enforce_eager=True` 跑通正确性。
- [ ] 改为 `False`，比较首次运行耗时和稳定运行吞吐。
- [ ] 比较不同 `max_num_seqs`、`max_num_batched_tokens` 的影响。
- [ ] 比较 `tensor_parallel_size=1` 和多卡配置。
- [ ] 使用相同随机种子和生成长度进行基准对比。

基准结果不能只看 tokens/s，也要记录 GPU 型号、模型、输入/输出长度分布、batch 参数、是否启用 CUDA Graph 和显存占用。

## 6. 推荐的注释方式

注释应解释“为什么这样设计”和“状态如何变化”，避免逐句翻译 Python 语法。每个核心函数优先记录：

```python
# 输入：seqs 中哪些字段必须已经准备好
# 输出：返回值的类型、shape 和所属设备
# 状态变化：修改了 Sequence、BlockManager 或全局 Context 的哪些字段
# 性能原因：为什么使用 flatten、pin_memory、cache、compile 或 collective
```

特别值得注释的变量：

- `num_cached_tokens`
- `num_scheduled_tokens`
- `block_table`
- `slot_mapping`
- `cu_seqlens_q` / `cu_seqlens_k`
- `context_lens`
- `is_prefill`

建议把个人推导和长篇解释写在本文件或单独笔记中，源代码注释只保留理解代码所必需的内容。

## 7. 每阶段自测清单

- [ ] 能不看代码画出一次 generate 调用链。
- [ ] 能解释 prefill 和 decode 的输入差异。
- [ ] 能手算 Sequence 的 block 数和最后一个 block 的 token 数。
- [ ] 能画出 waiting、running、finished 及 preempt 的状态转换。
- [ ] 能说明 block table 与 slot mapping 的区别。
- [ ] 能说明 Prefix Cache 的命中条件和引用计数作用。
- [ ] 能写出 Qwen3DecoderLayer 的计算顺序与主要 tensor shape。
- [ ] 能解释 Column Parallel、Row Parallel 各自的切分和通信。
- [ ] 能说明 eager decode 和 CUDA Graph decode 的选择条件。
- [ ] 能设计一个公平、可复现的推理吞吐基准。

## 8. 推荐学习节奏

可以按 8～12 次学习完成，每次 60～90 分钟：

1. 示例、配置和生成主循环。
2. Sequence 与状态机。
3. Scheduler 的 prefill/decode。
4. BlockManager 与 Prefix Cache。
5. ModelRunner 的输入准备。
6. Qwen3 模型结构。
7. Attention、KV Cache 与 Context。
8. Tensor Parallel 与权重加载。
9. CUDA Graph、`torch.compile` 与 benchmark。
10. 回顾调用链，补测试和整理注释。

每次学习结束，至少留下一个可验证产物：一张图、一段 shape 推导、一个小测试、一组运行日志或一个注释提交。

## 9. 完成路线后的练习项目

按难度从低到高选择：

1. 为 Sequence 和 BlockManager 补充 CPU 单元测试。
2. 为每轮调度增加可选的 debug 日志，显示请求状态和 block 使用率。
3. 增加 `top_k` 采样参数并贯通 SamplingParams、Sequence、Sampler。
4. 统计 prefill/decode 延迟、吞吐量和 KV Cache 命中率。
5. 增加一个不依赖全局 Context 的显式上下文传递实验版本。
6. 对照正式 vLLM 的同类模块，记录简化点和生产级实现需要补充的能力。

建议先完成测试和可观测性练习，再修改调度或显存管理算法；这样后续实验更容易验证正确性。
