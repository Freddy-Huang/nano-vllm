"""第四章配套练习：纯 Python 模拟 paged KV Cache 的地址与调度元数据。"""

from dataclasses import dataclass, field


def num_blocks(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


def token_slots(num_tokens: int, block_table: list[int], block_size: int) -> list[int]:
    """返回逻辑序列中每个 token 对应的物理线性 slot。"""
    assert len(block_table) >= num_blocks(num_tokens, block_size)
    return [
        block_table[position // block_size] * block_size + position % block_size
        for position in range(num_tokens)
    ]


@dataclass
class MiniSequence:
    token_ids: list[int]
    block_table: list[int]
    block_size: int = 4
    num_cached_tokens: int = 0
    num_scheduled_tokens: int = 0
    prompt_length: int = field(init=False)

    def __post_init__(self) -> None:
        self.prompt_length = len(self.token_ids)

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    def schedule_prefill(self, budget: int) -> list[int]:
        remaining = len(self.token_ids) - self.num_cached_tokens
        self.num_scheduled_tokens = min(remaining, budget)
        start = self.num_cached_tokens
        return self.token_ids[start : start + self.num_scheduled_tokens]

    def finish_prefill_step(self) -> None:
        self.num_cached_tokens += self.num_scheduled_tokens
        self.num_scheduled_tokens = 0

    def append_sampled_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def decode_metadata(self) -> dict[str, int]:
        length = len(self.token_ids)
        logical_block = (length - 1) // self.block_size
        offset = (length - 1) % self.block_size
        return {
            "input_id": self.last_token,
            "position": length - 1,
            "context_len": length,
            "slot": self.block_table[logical_block] * self.block_size + offset,
        }


def check_paged_slots() -> None:
    assert num_blocks(1, 4) == 1
    assert num_blocks(4, 4) == 1
    assert num_blocks(5, 4) == 2
    assert token_slots(6, [3, 7], 4) == [12, 13, 14, 15, 28, 29]


def check_chunked_prefill() -> None:
    seq = MiniSequence(list(range(10)), block_table=[4, 1, 8])
    assert seq.schedule_prefill(budget=6) == [0, 1, 2, 3, 4, 5]
    seq.finish_prefill_step()
    assert seq.num_cached_tokens == 6

    assert seq.schedule_prefill(budget=6) == [6, 7, 8, 9]
    seq.finish_prefill_step()
    assert seq.num_cached_tokens == 10


def check_decode_metadata() -> None:
    seq = MiniSequence(list(range(6)), block_table=[3, 7, 2])
    seq.num_cached_tokens = 6
    seq.append_sampled_token(100)
    assert seq.decode_metadata() == {
        "input_id": 100,
        "position": 6,
        "context_len": 7,
        "slot": 30,
    }

    seq.append_sampled_token(101)
    assert seq.decode_metadata()["slot"] == 31
    seq.append_sampled_token(102)
    assert seq.decode_metadata()["slot"] == 8  # 第三个物理 block 是 2。


def main() -> None:
    check_paged_slots()
    check_chunked_prefill()
    check_decode_metadata()
    print("All LLM inference checks passed!")


if __name__ == "__main__":
    main()

