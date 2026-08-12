"""第一章配套练习：不依赖 GPU 或第三方库，可直接用 Python 运行。"""

from dataclasses import dataclass
from enum import Enum, auto
from itertools import count


def count_blocks(num_tokens: int, block_size: int = 4) -> int:
    """返回容纳 num_tokens 所需的 block 数。"""
    assert num_tokens > 0, "本练习假设序列至少包含一个 token"
    assert block_size > 0
    return (num_tokens + block_size - 1) // block_size


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(slots=True)
class MiniSamplingParams:
    temperature: float = 1.0
    max_tokens: int = 4

    def __post_init__(self) -> None:
        assert self.temperature > 0
        assert self.max_tokens > 0


class MiniSequence:
    """只保留理解 Python 所需字段的缩小版 Sequence。"""

    counter = count()
    block_size = 4

    def __init__(self, token_ids: list[int], params: MiniSamplingParams):
        assert token_ids, "prompt 不能为空"
        self.seq_id = next(MiniSequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = token_ids.copy()
        self.num_prompt_tokens = len(token_ids)
        self.max_tokens = params.max_tokens

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, index: int) -> int:
        return self.token_ids[index]

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        return count_blocks(len(self), self.block_size)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        if self.num_completion_tokens >= self.max_tokens:
            self.status = SequenceStatus.FINISHED
        else:
            self.status = SequenceStatus.RUNNING


def check_copy_and_containers() -> None:
    original = [10, 20]
    copied = original.copy()
    original.append(30)

    assert original == [10, 20, 30]
    assert copied == [10, 20]

    outputs = {2: [22], 0: [11]}
    ordered = [outputs[seq_id] for seq_id in sorted(outputs)]
    assert ordered == [[11], [22]]


def check_blocks() -> None:
    expected = {1: 1, 4: 1, 5: 2, 8: 2, 9: 3}
    actual = {num_tokens: count_blocks(num_tokens) for num_tokens in expected}
    assert actual == expected


def check_sequence() -> None:
    caller_tokens = [101, 102, 103]
    params = MiniSamplingParams(temperature=0.6, max_tokens=2)
    seq = MiniSequence(caller_tokens, params)

    caller_tokens.append(999)
    assert seq.token_ids == [101, 102, 103], "Sequence 应保存自己的列表副本"
    assert len(seq) == 3
    assert seq[-1] == 103
    assert seq.num_blocks == 1
    assert seq.num_completion_tokens == 0

    seq.append_token(201)
    assert seq.status == SequenceStatus.RUNNING
    assert seq.num_completion_tokens == 1

    seq.append_token(202)
    assert seq.is_finished
    assert seq.num_completion_tokens == 2
    assert seq.num_blocks == 2


def main() -> None:
    check_copy_and_containers()
    check_blocks()
    check_sequence()
    print("All Python basics checks passed!")


if __name__ == "__main__":
    main()

