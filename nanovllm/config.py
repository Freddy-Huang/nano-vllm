"""Nano-vLLM 的引擎配置。

``Config`` 把用户传给 ``LLM(...)`` 的运行参数和模型目录中的 Hugging Face
配置集中到一个对象中。后续的 LLMEngine、Scheduler 和 ModelRunner 都读取
这个对象，以保证调度、模型执行和 KV Cache 使用同一组参数。
"""

import os
from dataclasses import dataclass

from transformers import AutoConfig


# dataclass 会根据下面声明的字段自动生成 __init__()、__repr__() 等方法，
# 因而不需要手写大量 self.xxx = xxx。
# slots=True 会限制实例只能拥有这些已声明字段，并减少配置对象的额外开销。
@dataclass(slots=True)
class Config:
    """一次 Nano-vLLM 引擎实例所使用的全部配置。

    字段可以分成三类：

    1. 用户配置：例如 model、max_num_seqs、tensor_parallel_size；
    2. 模型配置：hf_config 从模型目录的 config.json 自动读取；
    3. 运行时结果：eos 和 num_kvcache_blocks 会在引擎初始化过程中计算并回填。
    """

    # 本地 Hugging Face 模型目录。它用于读取 tokenizer、config.json 和
    # .safetensors 权重，而不是 Hugging Face Hub 上的模型名称。
    model: str

    # 一次调度允许送入模型的最大 token 总数。Prefill 时，不同请求本轮需要
    # 计算的 token 数之和不能超过它；它也参与决定模型预热的输入规模。
    max_num_batched_tokens: int = 16384

    # 一轮最多同时调度多少条序列。它限制并发 batch 大小，也限制 CUDA Graph
    # 捕获时准备的最大 decode batch size。
    max_num_seqs: int = 512

    # 配置的单条序列长度上限，模型预热和 CUDA Graph 缓冲区会参考它。初始化
    # 结束时它会被限制为不超过模型自身声明的 max_position_embeddings。
    # 当前轻量实现没有在此处检查 prompt长度 + max_tokens，调用者也应避免超限。
    max_model_len: int = 4096

    # 允许模型权重、运行时张量和 KV Cache 合计使用的目标显存比例。
    # ModelRunner 会结合设备总显存与预热得到的峰值显存，推算可分配的 Cache 块数。
    gpu_memory_utilization: float = 0.9

    # 张量并行使用的 GPU/进程数量。1 表示单卡；大于 1 时，每个 rank 持有
    # 一部分模型权重，并通过 NCCL collective 合并计算结果。
    tensor_parallel_size: int = 1

    # True：始终使用普通 PyTorch eager 路径；False：符合条件的 decode 会使用
    # CUDA Graph。学习和调试阶段推荐设为 True，使实际执行链路更加直观。
    enforce_eager: bool = False

    # 从模型目录的 config.json 解析出的 Hugging Face 配置，例如 hidden_size、
    # num_hidden_layers 和 num_attention_heads。该字段由 __post_init__ 自动填写。
    hf_config: AutoConfig | None = None

    # 序列结束 token 的 ID。这里先用 -1 作为“尚未初始化”的占位值，之后由
    # LLMEngine 从 tokenizer.eos_token_id 读取，再交给 Scheduler 做停止判断。
    eos: int = -1

    # 一个物理 KV Cache block 能保存多少个连续 token 的 K/V。本实现要求它是
    # 256 的整数倍；Sequence 的逻辑块划分和 BlockManager 都使用同一个值。
    kvcache_block_size: int = 256

    # GPU 上实际能够分配的 KV Cache block 数。初始值 -1 是占位符；模型预热后，
    # ModelRunner.allocate_kv_cache() 会根据剩余显存计算并覆盖它。
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """完成自动生成的 __init__() 之后，校验参数并补全模型配置。"""

        # 当前加载器只遍历本地目录中的权重文件，因此 model 必须是有效目录。
        assert os.path.isdir(self.model)

        # 保证 block 大小符合当前 KV Cache 和 FlashAttention 路径的约束。
        assert self.kvcache_block_size % 256 == 0

        # 限制张量并行规模，避免创建当前实现未计划支持的进程数量。
        assert 1 <= self.tensor_parallel_size <= 8

        # 读取模型目录中的 config.json。AutoConfig 会根据 model_type 返回具体
        # 配置对象；对本项目使用的模型而言，通常会得到 Qwen3Config。
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # 用户可以主动设置更短的上下文来节省 KV Cache，但不能请求超过模型
        # 位置编码所支持的最大长度。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
