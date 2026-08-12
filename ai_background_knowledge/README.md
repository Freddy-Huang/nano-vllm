# Nano-vLLM 的 AI 背景知识

这个目录用于补充阅读 Nano-vLLM 所需的背景知识。它不是完整的计算机课程，而是一套“学完马上能在项目中找到对应代码”的入门笔记。

## 推荐顺序

| 顺序 | 主题 | 学完后能够做什么 | 状态 |
| ---: | --- | --- | --- |
| 01 | [面向 Nano-vLLM 的 Python 入门](01_python_for_nano_vllm.md) | 看懂配置、请求对象、主循环和多进程代码 | 已整理 |
| 02 | [PyTorch 与 Tensor](02_pytorch_and_tensors.md) | 看懂 shape、`nn.Module` 和矩阵运算 | 已整理 |
| 03 | [Transformer 基础](03_transformer_basics.md) | 画出 Attention、MLP、残差和归一化的数据流 | 已整理 |
| 04 | [大模型推理基础](04_llm_inference_basics.md) | 解释 tokenizer、prefill、decode 和 KV Cache | 已整理 |
| 05 | [GPU、CUDA 与并行基础](05_gpu_cuda_and_parallelism.md) | 理解显存、kernel launch 和 collective | 已整理 |

## Conda 环境准备

项目要求 Python `>=3.10,<3.13`，这里推荐 Python 3.11，并统一把 Conda 环境命名为 `vllm`：

```bash
conda create -n vllm python=3.11 pip -y
conda activate vllm
python -m pip install --upgrade pip
```

### 只运行背景知识的 CPU 练习

五个配套练习中只有第 02、03 章使用第三方包，安装 CPU 版 PyTorch 即可：

```bash
python -m pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu
```

验证环境：

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())"
```

CPU 版 PyTorch 的最后一项应输出 `False`。CPU 学习环境**不要执行** `python -m pip install -e .`：这会继续安装项目声明的 Triton、FlashAttention 等 GPU 依赖，而运行本目录练习并不需要它们。

如果暂时只学习第 01、04、05 章，连 PyTorch 也不需要；Conda 环境自带的 Python 就能运行对应脚本。

## 使用方法

1. 先读每篇的“本章目标”和“重点地图”。
2. 亲手运行代码，不要只看。
3. 完成章末自测；答不上来时再回到对应小节。
4. 打开文中链接的 Nano-vLLM 源码，确认知识点如何落地。

### 示例运行条件

| 示例 | CPU 可运行 | 需要 NVIDIA GPU | 说明 |
| --- | :---: | :---: | --- |
| `ai_background_knowledge/examples/01_python_basics.py` | ✓ |  | 仅 Python 标准库 |
| `ai_background_knowledge/examples/02_pytorch_tensors.py` | ✓ |  | 需要 CPU 版 PyTorch |
| `ai_background_knowledge/examples/03_transformer_basics.py` | ✓ |  | 需要 CPU 版 PyTorch |
| `ai_background_knowledge/examples/04_llm_inference.py` | ✓ |  | 纯 Python 调度与 KV Cache 模拟 |
| `ai_background_knowledge/examples/05_gpu_parallelism.py` | ✓ |  | 只计算显存和并行 shape，不执行 CUDA |
| 项目根目录 `example.py` |  | ✓ | 真正加载 Qwen3，并运行 Nano-vLLM 推理 |
| 项目根目录 `bench.py` |  | ✓ | 真正加载模型并执行吞吐量基准，负载更大 |

五章背景练习可以依次运行：

```bash
python ai_background_knowledge/examples/01_python_basics.py
python ai_background_knowledge/examples/02_pytorch_tensors.py
python ai_background_knowledge/examples/03_transformer_basics.py
python ai_background_knowledge/examples/04_llm_inference.py
python ai_background_knowledge/examples/05_gpu_parallelism.py
```

练习使用断言自动检查关键结论。上面五个背景练习全部在 CPU 上运行；即使第 05 章讲 GPU，它的配套脚本也只是做数值和 shape 推导。

真正运行 Nano-vLLM 的 `example.py`、`bench.py` 时，需要 NVIDIA GPU、可用的 CUDA/NCCL、与环境匹配的 CUDA 版 PyTorch、Triton、FlashAttention，以及本地模型权重。此时应根据机器的驱动和 CUDA 情况，使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 生成对应命令，再安装项目；不要继续使用上面的 CPU-only PyTorch 作为完整推理环境。
