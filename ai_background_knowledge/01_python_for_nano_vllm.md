# 01｜面向 Nano-vLLM 的 Python 入门

> 目标不是背完 Python，而是能顺畅阅读 Nano-vLLM。建议边看边运行示例，用时约 2～3 小时，也可以拆成两次学习。

## 本章目标

学完后，你应该能解释：

- 列表、字典、元组分别适合保存什么数据；
- 函数的参数、返回值、类型提示和 `**kwargs` 是什么；
- 类、实例、继承、`super()` 和特殊方法如何工作；
- `dataclass`、`property`、枚举和迭代器解决了什么问题；
- Nano-vLLM 为什么创建子进程，以及对象如何在进程间传递。

暂时不必学习装饰器实现原理、元类、异步编程和复杂的并发控制。遇到 `@dataclass`、`@property` 时，先把它们当成 Python 提供的“代码生成/行为声明”即可。

## 重点地图

| 优先级 | 知识 | 在项目中的用途 |
| --- | --- | --- |
| 必会 | 变量、列表、字典、切片、循环、函数 | 几乎每个文件都会出现 |
| 必会 | 类、实例属性、继承、特殊方法 | `Sequence`、`LLMEngine`、模型层 |
| 必会 | `dataclass`、`property`、类型提示 | 配置、采样参数、序列状态 |
| 看懂即可 | 枚举、迭代器、生成器 | 请求状态、请求 ID |
| 先理解概念 | 多进程、序列化、Event、共享内存 | Tensor Parallel 的多 GPU 执行 |

---

## 1. 变量保存的是“对象的引用”

先看一个最容易踩坑的例子：

```python
tokens = [10, 20]
same_tokens = tokens
copied_tokens = tokens.copy()

tokens.append(30)

print(same_tokens)    # [10, 20, 30]，指向同一个列表
print(copied_tokens)  # [10, 20]，是一个浅拷贝
```

可以把变量理解成贴在对象上的标签，而不是装对象的盒子。`same_tokens = tokens` 只增加一个标签，没有复制列表。

这正是 `Sequence` 构造函数使用 `copy(token_ids)` 的原因：请求内部要保存自己的 token 列表，调用者随后修改原列表时，不应影响请求。

### 可变与不可变

- 常见不可变对象：`int`、`float`、`bool`、`str`、`tuple`、`None`。
- 常见可变对象：`list`、`dict`、`set`，以及大多数自定义类实例。

```python
temperature = 0.8
temperature = 0.6  # 让名称指向新 float，不会修改原来的 0.8

token_ids = [1, 2]
token_ids.append(3)  # 直接修改原列表
```

**重点：** 看到函数修改列表、字典或对象属性时，要问“其他地方是否也持有同一个对象”。这对理解请求状态变化很重要。

---

## 2. 四种常用容器

### 2.1 `list`：有顺序、可修改

```python
token_ids = [101, 2048, 13, 102]

first = token_ids[0]       # 101
last = token_ids[-1]       # 102
prompt = token_ids[:3]     # [101, 2048, 13]，左闭右开
token_ids.append(999)      # 在末尾增加元素
length = len(token_ids)    # 5
```

切片 `items[start:stop]` 包含 `start`，不包含 `stop`。Nano-vLLM 用它分离 prompt token 与新生成的 completion token。

### 2.2 `dict`：用键查找值

```python
outputs = {}
outputs[2] = [31, 32]
outputs[0] = [11, 12]

ordered = [outputs[seq_id] for seq_id in sorted(outputs)]
print(ordered)  # [[11, 12], [31, 32]]
```

生成过程中，请求完成顺序不固定。因此 `LLMEngine.generate()` 先用 `seq_id` 作为字典的键收集结果，最后排序恢复输入顺序。

### 2.3 `tuple`：有顺序、不可修改

```python
result = (7, [101, 102])
seq_id, token_ids = result  # 解包
```

元组常用于返回一组含义固定的数据。项目中的 `step()` 会返回 `(outputs, num_tokens)`。

### 2.4 `set`：元素唯一、适合成员判断

```python
valid_fields = {"max_num_seqs", "enforce_eager"}
print("max_num_seqs" in valid_fields)  # True
```

### 一句话选择

- 一串有顺序的数据：`list`
- 名称到数据的映射：`dict`
- 固定的一组返回值：`tuple`
- 去重或快速判断“是否存在”：`set`

---

## 3. 条件、循环与推导式

### 3.1 条件判断

```python
prompt = "hello"

if isinstance(prompt, str):
    print("需要先 tokenizer.encode")
else:
    print("已经是 token ID 列表")
```

`isinstance(value, SomeType)` 比较的是对象类型。项目用它让公开 API 同时接受字符串和 token ID 列表。

### 3.2 `for`、`enumerate` 与 `zip`

```python
prompts = ["hello", "world"]
temperatures = [0.6, 0.8]

for index, (prompt, temperature) in enumerate(zip(prompts, temperatures)):
    print(index, prompt, temperature)
```

- `zip(a, b)` 把两组数据按位置配对；长度不同时会在较短的一组结束。
- `enumerate(items)` 同时产生下标和元素。

### 3.3 列表、集合和字典推导式

推导式是“循环 + 可选条件”的紧凑写法：

```python
numbers = [1, 2, 3, 4]
squares = [n * n for n in numbers]             # list
even_numbers = {n for n in numbers if n % 2 == 0}  # set
id_to_square = {n: n * n for n in numbers}     # dict
```

把下面这段项目风格代码从右向左读：

```python
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
```

意思是：“遍历 `kwargs` 中的键值对，只保留名字属于 `config_fields` 的项，组成新字典。”

**阅读技巧：** 推导式看不懂时，先还原成普通 `for`：

```python
config_kwargs = {}
for k, v in kwargs.items():
    if k in config_fields:
        config_kwargs[k] = v
```

---

## 4. 函数：输入、输出和副作用

```python
def count_blocks(num_tokens: int, block_size: int = 256) -> int:
    """计算容纳这些 token 所需的 block 数。"""
    assert num_tokens >= 0
    return (num_tokens + block_size - 1) // block_size
```

- `num_tokens: int` 是参数的类型提示。
- `block_size=256` 是默认参数。
- `-> int` 是返回值的类型提示。
- `assert` 表示程序必须满足的内部条件；不满足会抛出 `AssertionError`。
- `return` 把结果交给调用者。

类型提示主要帮助人和编辑器理解代码，Python 运行时通常不会自动阻止错误类型：

```python
def double(x: int) -> int:
    return x * 2

double("ha")  # 仍能运行，结果是 "haha"
```

### 4.1 位置参数、关键字参数与 `**kwargs`

```python
def create_engine(model, max_num_seqs=512, enforce_eager=False):
    print(model, max_num_seqs, enforce_eager)

create_engine("/model", max_num_seqs=8, enforce_eager=True)

options = {"max_num_seqs": 8, "enforce_eager": True}
create_engine("/model", **options)  # 把字典展开为关键字参数
```

在函数定义中，`**kwargs` 会收集未明确列出的关键字参数：

```python
def show_options(**kwargs):
    print(kwargs)

show_options(a=1, b=2)  # {'a': 1, 'b': 2}
```

`LLMEngine.__init__(self, model, **kwargs)` 就用这种方式接收各种引擎配置。

### 4.2 副作用

函数除了返回值，还可能修改外部可见状态：

```python
def append_token(token_ids: list[int], token_id: int) -> None:
    token_ids.append(token_id)  # 修改调用者传入的列表，这就是副作用
```

阅读项目函数时固定问三件事：

1. 输入是什么？
2. 返回什么？
3. 修改了哪个对象的状态？

### 4.3 默认参数陷阱

不要把可变对象作为默认参数：

```python
# 错误示范：多次调用会共享同一个列表
def add_bad(token, tokens=[]):
    tokens.append(token)
    return tokens

# 推荐：每次调用创建新列表
def add_good(token, tokens=None):
    if tokens is None:
        tokens = []
    tokens.append(token)
    return tokens
```

默认值会在函数定义时创建一次，而不是每次调用时创建。即使某个项目中的默认对象当前没有被修改，也不要在自己的代码里模仿这种写法。

---

## 5. 类与实例：把数据和行为放在一起

下面是一个缩小版请求对象：

```python
class Request:
    next_id = 0  # 类属性：所有实例共享

    def __init__(self, token_ids: list[int]):
        self.request_id = Request.next_id
        Request.next_id += 1
        self.token_ids = token_ids.copy()  # 实例属性：每个请求各自拥有
        self.finished = False

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

request = Request([10, 20])
request.append_token(30)
print(request.request_id, request.token_ids)
```

- 类是创建对象的模板，实例是实际创建出的对象。
- `__init__` 在实例创建时初始化状态。
- `self` 表示“当前这个实例”；调用 `request.append_token(30)` 时，Python 自动把 `request` 作为 `self`。
- `self.xxx` 通常是实例属性，`Request.xxx` 通常是类属性。

### 5.1 特殊方法让对象表现得像内置类型

名字两边带双下划线的方法通常称为特殊方法：

```python
class Tokens:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

tokens = Tokens([11, 22, 33])
print(len(tokens))  # 自动调用 __len__
print(tokens[1])    # 自动调用 __getitem__
```

`Sequence` 实现了这两个方法，所以它可以使用 `len(seq)`、`seq[i]` 和切片。因为从 `0` 开始连续调用 `__getitem__` 能取值，它也可以被 `for` 遍历；这是一种旧式的迭代协议。

### 5.2 继承与 `super()`

```python
class Engine:
    def __init__(self, name):
        self.name = name

    def run(self):
        return "running"

class LLM(Engine):
    def __init__(self, name):
        super().__init__(name)  # 调用父类初始化

llm = LLM("nano")
print(llm.run())  # 子类继承了父类方法
```

项目里的 `LLM` 是 `LLMEngine` 的空子类，因此会直接继承后者的初始化和生成方法。PyTorch 模型层继承 `nn.Module` 时，也必须调用 `super().__init__()`，让 PyTorch 建立参数和子模块管理机制。

**重点：** 看到一个类里找不到某个方法时，沿着括号中的父类继续找。

---

## 6. `dataclass`：少写重复的配置类代码

普通配置类要手写 `__init__`。`@dataclass` 会根据字段声明自动生成初始化、打印和比较等方法：

```python
from dataclasses import dataclass

@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 0

params = SamplingParams(temperature=0.6, max_tokens=8)
print(params.max_tokens)  # 8
```

执行顺序可以简化为：

```text
SamplingParams(...)
  -> dataclass 自动生成的 __init__ 给字段赋值
  -> 自动调用 __post_init__ 做补充校验或计算
```

`slots=True` 限制实例只能拥有声明过的字段，通常还能减少内存占用。写错属性名时会更早暴露：

```python
params.max_token = 9  # AttributeError，字段名应为 max_tokens
```

项目的 `Config` 用 `__post_init__` 校验模型目录和并行规模，并根据模型配置修正最大长度。

---

## 7. `property`：把无参数计算写成属性

```python
class Sequence:
    def __init__(self, token_ids, num_prompt_tokens):
        self.token_ids = token_ids
        self.num_prompt_tokens = num_prompt_tokens

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

seq = Sequence([1, 2, 3, 4], num_prompt_tokens=3)
print(seq.num_completion_tokens)  # 注意：不是 seq.num_completion_tokens()
```

`property` 适合表达“由当前状态计算出的属性”。调用者读起来像访问字段，但每次访问都会执行方法。

在 `Sequence` 中：

- `num_tokens` 是会被显式更新的实例属性；
- `num_completion_tokens` 是根据已有状态动态计算的 property；
- `num_blocks` 也是动态计算值。

**阅读技巧：** 一个属性不知道从哪里赋值时，搜索同名的 `def` 和 `@property`。

---

## 8. 枚举：让状态比数字更有意义

```python
from enum import Enum, auto

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

status = SequenceStatus.WAITING
if status == SequenceStatus.WAITING:
    status = SequenceStatus.RUNNING
```

`auto()` 自动生成互不相同的值。业务代码只关心有意义的名称，不依赖值究竟是 1、2 还是 3。

不要写 `status == "WAITING"`，因为左边是枚举成员，不是字符串。

---

## 9. 迭代器、生成器与 `itertools.count`

可迭代对象能被 `for` 消费：

```python
for token_id in [10, 20, 30]:
    print(token_id)
```

迭代器会记住“走到哪里了”，`next()` 每次取下一个值：

```python
from itertools import count

counter = count()  # 0, 1, 2, ...，按需产生，不会先创建无限列表
print(next(counter))  # 0
print(next(counter))  # 1
```

`Sequence.counter = count()` 是类属性，所有请求共享同一个计数器，因此每次 `next(Sequence.counter)` 都能得到新的序列 ID。

生成器函数使用 `yield` 按需产生值：

```python
def unfinished(requests):
    for request in requests:
        if not request.finished:
            yield request
```

本项目暂时很少直接写生成器。你只需记住：生成器通常只能逐步消费一次，不是已经装满数据的列表。

---

## 10. 类型提示：它是阅读地图，不是运行保证

Nano-vLLM 使用了现代 Python 类型语法：

```python
def add_request(prompt: str | list[int]) -> None:
    ...
```

含义是 `prompt` 可以是字符串，或“元素为整数的列表”。常见写法：

| 写法 | 含义 |
| --- | --- |
| `list[int]` | 整数列表 |
| `dict[int, str]` | 键为整数、值为字符串的字典 |
| `str \| None` | 字符串或空值 |
| `tuple[int, list[int]]` | 一个整数和一个整数列表组成的元组 |
| `-> None` | 函数主要做操作，不返回有用值 |

`None` 表示“没有值”。判断时通常写 `value is None`，不要写 `value == None`。

类型提示中出现 `torch.Tensor` 时，还不够描述张量。阅读 AI 代码要额外记录：

```text
shape + dtype + device + 各维含义
```

例如：`input_ids: [T], int64, CUDA, T 是本轮 token 总数`。这是下一篇 PyTorch 知识的重点。

---

## 11. 模块、包与导入

一个 `.py` 文件通常是一个模块，包含多个模块的目录可以构成包：

```python
from nanovllm.config import Config
```

可以从右向左读：从 `nanovllm` 包的 `config` 模块中导入 `Config` 名称。

当脚本需要创建多进程时，应使用入口保护：

```python
def main():
    print("program starts")

if __name__ == "__main__":
    main()
```

文件被直接运行时，`__name__` 是 `"__main__"`；被其他文件导入时不是。入口保护可防止使用 `spawn` 创建的子进程再次执行主程序。

---

## 12. 多进程：先建立正确的心智模型

### 12.1 进程不是普通函数调用

```python
import multiprocessing as mp

def worker(rank: int):
    print(f"worker {rank}")

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=worker, args=(1,))
    process.start()  # 启动新进程
    process.join()   # 等待新进程结束
```

新进程拥有独立的 Python 解释器和大部分独立内存。不能假设“父进程改了普通列表，子进程就自动看到”。进程之间必须用明确的通信机制。

Nano-vLLM 的思路是：

```text
rank 0 主进程
  ├─ 执行调度和第 0 张 GPU 的模型计算
  ├─ 通过共享内存写入命令/参数
  └─ 通过 Event 唤醒其他 rank
       ├─ rank 1 子进程 -> GPU 1
       ├─ rank 2 子进程 -> GPU 2
       └─ ...
```

- `spawn`：启动全新的 Python 子进程，更适合 CUDA 场景。
- `Process.start()`：真正启动子进程。
- `Process.join()`：等待子进程退出，避免资源悬空。
- `Event`：进程间的“信号灯”，用于通知而不是承载大块数据。
- `SharedMemory`：多个进程可访问的一块共享字节区域。

第一次阅读时理解这张图即可，不要立即深挖锁、内存一致性和共享内存编码细节。

### 12.2 序列化与 `__getstate__` / `__setstate__`

进程间传递 Python 对象时，通常要先序列化，也就是把对象状态转换为可传输形式。`pickle` 在序列化对象时可以调用：

- `__getstate__()`：决定“发送哪些状态”；
- `__setstate__(state)`：收到后如何重建状态。

Nano-vLLM 为 `Sequence` 自定义这两个方法，是为了减少进程通信数据：prefill 需要 prompt token，而 decode 通常只需要最后一个 token 和长度等元数据。

**重点：** 子进程拿到的通常是重建后的对象，不是与父进程共享的同一个普通 Python 对象。

---

## 13. 把知识映射回 Nano-vLLM

建议按顺序打开以下源码，并完成右侧任务：

| 源码 | 重点观察 | 你应该能说出的结论 |
| --- | --- | --- |
| [`sampling_params.py`](../nanovllm/sampling_params.py) | `@dataclass`、默认值、`__post_init__` | 参数如何自动初始化和校验 |
| [`config.py`](../nanovllm/config.py) | union 类型、`slots`、状态修正 | 配置哪些是输入，哪些在初始化后补齐 |
| [`sequence.py`](../nanovllm/engine/sequence.py) | 类属性、property、切片、特殊方法 | 请求数据怎样随生成过程变化 |
| [`llm.py`](../nanovllm/llm.py) | 空子类 | `LLM` 为什么仍有 `generate()` |
| [`llm_engine.py`](../nanovllm/engine/llm_engine.py) | `**kwargs`、推导式、`zip`、多进程 | 一批请求如何加入并循环执行 |
| [`linear.py`](../nanovllm/layers/linear.py) | 继承、`super()`、方法重写 | 各线性层共享什么，又改变什么 |

### 精读一个真实片段

```python
if not isinstance(sampling_params, list):
    sampling_params = [sampling_params] * len(prompts)
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

逐句翻译：

1. 如果调用者只传入一个采样配置，就构造一个与 prompts 等长的列表；这里重复的是同一个对象引用，但后续只读取它，所以当前实现没有问题。
2. `zip` 把每个 prompt 与对应配置配对。
3. `self.add_request(...)` 把请求交给当前引擎实例。

这就是推荐的源码阅读方式：先确定对象类型，再解释控制流，最后记录状态变化。

---

## 14. 动手练习

先运行配套练习：

```bash
python ai_background_knowledge/examples/01_python_basics.py
```

然后不要看答案，尝试完成：

### 练习 1：手算 block 数

block size 为 4 时，token 数分别为 1、4、5、8、9，需要多少个 block？再用 `count_blocks()` 验证。

### 练习 2：解释请求状态

阅读 `MiniSequence.append_token()`，回答它修改了哪些字段。为什么 `num_completion_tokens` 不需要手动加一？

### 练习 3：还原推导式

把下面代码改写成普通 `for` 循环：

```python
finished_ids = [seq.seq_id for seq in sequences if seq.is_finished]
```

### 练习 4：追踪继承

打开 `nanovllm/llm.py`。它没有实现 `generate()`，为什么 `LLM(...).generate(...)` 合法？真正的方法在哪里？

### 练习 5：找出共享引用

预测输出，再运行验证：

```python
params = SamplingParams(max_tokens=8)
items = [params] * 3
items[0].max_tokens = 2
print([item.max_tokens for item in items])
```

思考：如果希望三个元素完全独立，应如何创建？

<details>
<summary>点击查看答案</summary>

1. 分别需要 1、1、2、2、3 个 block。
2. 修改 `token_ids`、`last_token`、`num_tokens`；completion 数由总 token 数减 prompt token 数动态计算。
3. 先创建空列表，遍历 `sequences`，满足 `seq.is_finished` 时 append `seq.seq_id`。
4. `LLM` 继承 `LLMEngine`，方法定义在 `nanovllm/engine/llm_engine.py`。
5. 输出 `[2, 2, 2]`，因为列表中是同一个对象的三个引用。可用 `[SamplingParams(max_tokens=8) for _ in range(3)]` 创建三个对象。

</details>

---

## 15. 自测清单

不看正文，确认自己能够做到：

- [ ] 解释赋值与浅拷贝的区别。
- [ ] 用列表、字典和元组分别表示合适的数据。
- [ ] 看懂切片、`zip`、`enumerate` 和简单推导式。
- [ ] 从函数签名中找出输入、默认值、类型和返回值。
- [ ] 区分类属性与实例属性。
- [ ] 解释 `__init__`、`__len__`、`__getitem__` 的触发时机。
- [ ] 沿继承关系找到真正执行的方法。
- [ ] 解释 `dataclass`、`__post_init__`、`property` 和枚举的用途。
- [ ] 说明 `spawn`、`start()`、`join()`、`Event` 各自的大意。
- [ ] 说出进程间传对象为什么涉及序列化。

如果前 8 项可以独立回答，就已经具备阅读 Nano-vLLM 前半部分代码的 Python 基础。多进程暂时说出整体流程即可，读到 Tensor Parallel 时再回来复习。

## 下一步

下一章建议学习 PyTorch 与 Tensor，重点是：shape、dtype、device、广播、矩阵乘法、`nn.Module`、参数，以及 `torch.distributed` 的 collective。开始前可以先进入学习路线的“阶段 1”，阅读 `example.py` 到 `sampling_params.py`，用本章方法标出每个对象的类型和状态变化。

