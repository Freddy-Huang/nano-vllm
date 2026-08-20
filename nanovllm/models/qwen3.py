import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 的自注意力模块。

    为了提高推理吞吐，nano-vLLM 不使用常见的
    ``[batch_size, sequence_length, hidden_size]`` 三维输入，而是把一个批次中
    所有有效 token 压平为 ``[num_tokens, hidden_size]``。不同请求的序列边界由
    Attention 内部的运行时 context 记录。

    本模块还同时实现了两种常见优化：

    1. GQA（Grouped-Query Attention）：Q 的头数可以多于 K/V 的头数，多组 Q
       共享同一组 K/V，从而减少 KV Cache 占用。
    2. TP（Tensor Parallelism，张量并行）：不同 GPU 各自计算一部分注意力头，
       最后由输出投影层汇总结果。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        # TP 进程数通常也就是参与张量并行的 GPU 数量。
        # 注意：进程组必须由模型运行器在构造模型之前初始化。
        tp_size = dist.get_world_size()

        # num_heads 是整个模型的 Q 头总数；每个 TP rank 只保存并计算其中一份。
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        # K/V 头也按 TP rank 切分。GQA 中 num_kv_heads 通常小于 num_heads。
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # 单个注意力头的维度。部分配置会显式提供 head_dim；没有时再用传统公式计算。
        self.head_dim = head_dim or hidden_size // self.total_num_heads

        # 以下尺寸都是“单个 TP rank”上的最后一维大小，用于稍后拆分融合后的 QKV。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # 点积注意力需要除以 sqrt(head_dim)，防止维度增大后点积数值过大。
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # 一次线性变换同时得到 Q、K、V，比依次执行三个小投影更高效。
        # QKVParallelLinear 会沿输出维切分权重，因此每个 TP rank 只持有本地头。
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 各 rank 先用本地注意力头做投影，RowParallelLinear 随后通过 all-reduce
        # 把它们的贡献相加，恢复为完整的 hidden_size 表示。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 某些 Qwen3 配置把实际使用的 RoPE base 放在 rope_scaling 字典中。
        # 当前轻量实现只读取 rope_theta，并没有实现其他复杂的 RoPE 缩放策略。
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # RoPE（旋转位置编码）直接旋转 Q/K，不会给 hidden_states 加位置向量。
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 底层 Attention 根据运行时 context 自动区分：
        # - prefill：并行处理整段 prompt；
        # - decode：读取 KV Cache，一次为每个请求计算一个新 token。
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 标准 Qwen3（attention_bias=False）会对每个 Q/K 头单独做 RMSNorm，
        # 归一化的维度是 head_dim。保留带 bias 的分支是为了兼容其他配置。
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, hidden_size]
        # qkv: [num_tokens, q_size + kv_size + kv_size]（这里均为本 rank 的尺寸）
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 把拼在最后一维的多个头恢复出来：
        # q -> [num_tokens, num_heads, head_dim]
        # k/v -> [num_tokens, num_kv_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # positions: [num_tokens]，其中每个元素是对应 token 在自己序列中的位置。
        # RoPE 只作用于 Q/K，因为注意力权重由 Q 与 K 的相对角度决定。
        q, k = self.rotary_emb(positions, q, k)

        # o: [num_tokens, num_heads, head_dim]。
        # 底层会把本轮 K/V 写入 KV Cache；decode 时会复用历史 K/V。
        o = self.attn(q, k, v)

        # 合并本 rank 的所有 Q 头，再投影回模型隐藏维度。
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Transformer 层中的门控前馈网络（SwiGLU 形式）。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # gate_proj 和 up_proj 拥有相同输入，把二者合并为一次矩阵乘法。
        # 输出最后一维为 2 * intermediate_size（在 TP 中由各 rank 分片持有）。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 将 intermediate_size 再投影回 hidden_size；RowParallelLinear 会汇总
        # 不同 TP rank 的部分结果。
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # 此处只实现 Qwen3 使用的 SiLU 门控：SiLU(gate) * up。
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # gate_up 的最后一维由 [gate, up] 两半组成。
        gate_up = self.gate_up_proj(x)
        # SiluAndMul 在内部一分为二并计算 SiLU(gate) * up，
        # 所以张量最后一维从 2 * intermediate_size 变回 intermediate_size。
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """一个 Qwen3 解码器层：自注意力子层 + MLP 子层。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # Qwen3 是 decoder-only 模型，因此这里只需要带因果遮罩的自注意力，
        # 不包含 encoder-decoder 结构中的交叉注意力。
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 采用 Pre-Norm：先归一化，再进入 Attention/MLP。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 这里用 (hidden_states, residual) 两个张量延迟执行残差相加，以减少一次
        # 独立的逐元素 kernel：RMSNorm(x, residual) 会先计算 x + residual，
        # 返回“归一化后的输入”和“更新后的残差主干”。
        if residual is None:
            # 第一层尚无旧残差：保存 embedding，同时归一化后送入 Attention。
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 从第二层开始，先把上一层 MLP 输出加入残差，再做 input RMSNorm。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)

        # 把 Attention 输出加入残差并归一化，归一化结果送入 MLP；更新后的
        # residual 则绕过 MLP，留到下一层（或模型末尾）再相加。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """不含词表输出头的 Qwen3 Transformer 主干。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入把离散 token id 映射为 hidden_size 维向量。
        # VocabParallelEmbedding 会在 TP rank 之间按词表维度切分大矩阵。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 每一层结构相同但参数独立，层数由模型配置决定。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 所有解码器层结束后还需要一次最终 RMSNorm。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # input_ids/positions 的形状通常都是 [num_tokens]；输出为
        # [num_tokens, hidden_size]。批次内各序列已经由调度器压平。
        hidden_states = self.embed_tokens(input_ids)

        # residual 在第一层中初始化，随后作为贯穿各层的残差主干传递。
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 循环结束时 hidden_states 仍是最后一层 MLP 的输出；最终 RMSNorm 会先
        # 将它加到 residual 上，再归一化。下划线表示更新后的 residual 不再使用。
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """用于自回归文本生成的完整 Qwen3：Transformer 主干 + 语言模型头。"""

    # Hugging Face checkpoint 分别保存 q/k/v 和 gate/up 的权重，而本实现为了
    # 提升推理效率使用融合层。该映射告诉权重加载器每块原始权重应写到哪个
    # 融合参数中；例如 q_proj 会写入 qkv_proj 的 q 分片。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # lm_head 将 hidden_size 维隐藏状态投影到 vocab_size 维 logits。
        # 该权重同样沿词表维度做张量并行切分。
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 权重绑定：输入 embedding 与输出 lm_head 共享同一份参数存储，
            # 既符合部分模型的训练方式，也能节省一份词表矩阵的显存。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # forward 只返回隐藏状态，让推理引擎可以按需决定何时计算 logits。
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 将隐藏状态转换为“每个候选 token 的未归一化分数”。采样器随后才会
        # 应用 temperature、top-k/top-p 和 softmax 等策略。
        # prefill 阶段 ParallelLMHead 只选每条请求的最后一个 token 来算 logits，
        # 避免为 prompt 中无需采样的位置做无用计算。
        return self.lm_head(hidden_states)
