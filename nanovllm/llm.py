"""Nano-vLLM 面向使用者的顶层接口。

这个文件刻意保持得非常简单：公开类 ``LLM`` 继承真正实现推理流程的
``LLMEngine``，对外提供一个简短、稳定，并且接近 vLLM 使用习惯的类名。

调用关系如下：

``from nanovllm import LLM``
    -> ``nanovllm/__init__.py`` 重新导出本文件的 LLM
    -> ``LLM`` 继承 ``LLMEngine``
    -> 实际执行 ``LLMEngine.__init__()`` 和 ``LLMEngine.generate()``
"""

# 真正的模型初始化、请求调度和生成循环都实现在 LLMEngine 中。
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """Nano-vLLM 提供给用户的推理入口类。

    这里没有重写任何方法，所以 LLM 会直接继承 LLMEngine 的全部行为，包括：

    - ``__init__()``：读取配置、创建 ModelRunner、加载模型并创建 Scheduler；
    - ``generate()``：加入请求并循环执行 Prefill 和 Decode；
    - ``add_request()`` / ``step()``：管理一次生成中的请求和执行轮次；
    - ``exit()``：释放模型进程及相关资源。

    因此执行 ``LLM(model_path, ...)`` 时，Python 实际调用的是继承而来的
    ``LLMEngine.__init__()``；执行 ``llm.generate(...)`` 时，实际调用的是
    ``LLMEngine.generate()``。

    单独保留这个空子类的主要目的不是增加功能，而是把“用户使用的公开 API”
    与“内部引擎实现”分开。未来即使 LLMEngine 的组织方式变化，项目仍可以
    尽量维持 ``LLM(...)`` 这个简洁入口。
    """

    # pass 表示当前类体中没有新增属性或方法。这不代表 LLM 什么都不能做：
    # 通过 Python 继承，它已经拥有父类 LLMEngine 定义的所有方法。
    pass
