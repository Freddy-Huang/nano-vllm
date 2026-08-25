"""控制单条请求如何采样和何时停止生成的参数。

参数流转路径如下：

``LLM.generate()``
    -> 为每条 prompt 创建 ``Sequence``
    -> Sequence 保存本对象中的三个字段
    -> ModelRunner/Sampler 使用 temperature 采样
    -> Scheduler 使用 max_tokens 和 ignore_eos 判断是否结束

正式推理框架通常还支持 top-k、top-p、重复惩罚等选项；Nano-vLLM 为了保持
实现精简，目前只提供这里的三个参数。
"""

from dataclasses import dataclass


# dataclass 根据字段声明自动生成 __init__()，所以可以直接写：
# SamplingParams(temperature=0.6, max_tokens=128)。
# slots=True 限制实例只能拥有下面这些字段，并减少对象的额外内存开销。
@dataclass(slots=True)
class SamplingParams:
    """一条生成请求所使用的采样与停止参数。"""

    # 采样温度。Sampler 会先计算 logits / temperature，再执行 softmax 和随机采样：
    # - temperature = 1：不改变 logits 的相对尺度；
    # - 0 < temperature < 1：概率分布更尖锐，输出更稳定；
    # - temperature > 1：概率分布更平坦，输出更随机。
    temperature: float = 1.0

    # 最多新生成多少个 token，不包含 prompt 本身。Scheduler 每次追加 token 后
    # 都会检查 completion token 数，达到该值就把 Sequence 标记为 FINISHED。
    max_tokens: int = 64

    # False：采样到 tokenizer 的 EOS（结束符）时立即结束；
    # True：忽略 EOS，继续生成直到达到 max_tokens。基准测试常将它设为 True，
    # 从而让每条请求生成固定数量的 token，便于比较吞吐量。
    ignore_eos: bool = False

    def __post_init__(self):
        """在 dataclass 自动生成的 __init__() 结束后校验采样温度。"""

        # temperature=0 无法用于 logits / temperature，也通常代表贪心解码。
        # 当前 Sampler 只实现随机采样，没有单独的 argmax 贪心分支，所以拒绝
        # 零温度以及非常接近零的值。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
