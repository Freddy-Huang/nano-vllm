"""Nano-vLLM 的最小文本生成示例。

建议初学者先从这个文件进入项目。它展示了最外层的完整使用流程：

1. 指定本地 Qwen3 模型目录；
2. 使用 tokenizer 把普通对话转换成 Qwen3 需要的提示词格式；
3. 创建 Nano-vLLM 推理引擎；
4. 设置采样参数并调用 ``generate()``；
5. 打印模型新生成的文本。

真正的请求调度和生成循环位于 ``nanovllm/engine/llm_engine.py``，模型计算
位于 ``nanovllm/models/qwen3.py``。本文件只是调用这些功能的公开入口示例。
"""

import os

# LLM 是 Nano-vLLM 对外提供的推理入口；SamplingParams 用于控制生成过程。
from nanovllm import LLM, SamplingParams
# 这里额外创建 Hugging Face tokenizer，是为了调用 apply_chat_template()。
# LLM 内部也会创建自己的 tokenizer，并在 generate() 中负责 encode/decode。
from transformers import AutoTokenizer


def main():
    # expanduser() 会把路径开头的 "~" 展开为当前用户的主目录。
    # 这个目录必须是已经下载好的本地模型目录，通常至少包含 config.json、
    # tokenizer 配置和若干 .safetensors 权重文件。请按自己的环境修改此路径。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # tokenizer 负责在“人类可读文本”和“模型使用的 token ID”之间转换。
    # from_pretrained() 会从上面的本地模型目录读取 Qwen3 的 tokenizer 配置。
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 创建推理引擎时会读取模型配置、初始化 GPU/NCCL、构造 Qwen3 网络、
    # 加载权重、预热模型，并分配 KV Cache。因此这一行通常耗时和占显存最多。
    #
    # enforce_eager=True：直接执行 PyTorch 运算，不使用 CUDA Graph。速度可能
    # 略慢，但执行链路更直观，适合刚开始阅读和调试代码。
    # tensor_parallel_size=1：只使用一张 GPU，不进行多卡张量并行。
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # temperature 控制采样随机性：越小越倾向于高概率 token，越大越随机。
    # max_tokens 表示每条请求最多新生成 256 个 token，不包含输入 prompt。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    # Nano-vLLM 的 generate() 可以一次接收多条 prompt，并通过调度器将它们
    # 动态组成 batch。这里先准备两条普通的用户文本。
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]

    # Qwen3 是聊天模型，通常不能只把用户原文直接送进模型，而要使用它训练时
    # 约定的 Chat Template，加入角色标记和生成起始标记。
    #
    # 转换结果仍然是字符串，因为 tokenize=False。真正的字符串 -> token ID
    # 转换会在 LLMEngine.add_request() 中完成。
    # add_generation_prompt=True 会在末尾加入 assistant 开始回答所需的标记。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # generate() 是同步接口：它会先处理所有 prompt（Prefill），然后循环执行
    # Decode，每轮为运行中的请求生成一个新 token，直到遇到 EOS 或 max_tokens。
    # 返回值和 prompts 一一对应，每项形如：
    # {"text": "模型生成的文本", "token_ids": [生成的 token ID, ...]}
    outputs = llm.generate(prompts, sampling_params)

    # zip() 将每条格式化后的 prompt 与对应输出配对。
    for prompt, output in zip(prompts, outputs):
        print("\n")
        # !r 使用 repr() 形式打印字符串，便于看到换行符和 Chat Template 标记。
        print(f"Prompt: {prompt!r}")
        # output["text"] 只包含模型新生成的内容，不包含输入 prompt。
        print(f"Completion: {output['text']!r}")


# 只有直接执行 `python example.py` 时才调用 main()；如果该文件被其他模块
# import，则不会自动加载模型和开始推理。这是 Python 脚本的标准主入口写法。
if __name__ == "__main__":
    main()
