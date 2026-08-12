"""第五章配套练习：纯 Python 计算显存和 Tensor Parallel shape。"""


def kv_block_bytes(
    num_layers: int,
    block_size: int,
    num_kv_heads_per_gpu: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    return 2 * num_layers * block_size * num_kv_heads_per_gpu * head_dim * dtype_bytes


def column_parallel_weight_shape(
    output_size: int, input_size: int, world_size: int
) -> tuple[int, int]:
    assert output_size % world_size == 0
    return output_size // world_size, input_size


def row_parallel_weight_shape(
    output_size: int, input_size: int, world_size: int
) -> tuple[int, int]:
    assert input_size % world_size == 0
    return output_size, input_size // world_size


def check_kv_memory() -> None:
    size = kv_block_bytes(
        num_layers=28,
        block_size=256,
        num_kv_heads_per_gpu=4,
        head_dim=128,
        dtype_bytes=2,
    )
    assert size == 14_680_064
    assert size / 1024**2 == 14

    # GQA 减半 KV head 数，也会把 KV Cache 大小减半。
    half_heads = kv_block_bytes(28, 256, 2, 128, 2)
    assert half_heads == size // 2


def check_tensor_parallel_shapes() -> None:
    output_size, input_size, world_size = 4096, 2048, 4
    assert column_parallel_weight_shape(output_size, input_size, world_size) == (
        1024,
        2048,
    )
    assert row_parallel_weight_shape(output_size, input_size, world_size) == (
        4096,
        512,
    )

    # Column 分片沿输出维拼接；Row 分片的局部输出逐元素相加。
    column_outputs = [[1, 2], [3, 4]]
    gathered = column_outputs[0] + column_outputs[1]
    assert gathered == [1, 2, 3, 4]

    row_partial_0 = [1, 3]
    row_partial_1 = [2, 4]
    reduced = [a + b for a, b in zip(row_partial_0, row_partial_1)]
    assert reduced == [3, 7]


def main() -> None:
    check_kv_memory()
    check_tensor_parallel_shapes()
    print("All GPU and parallelism checks passed!")


if __name__ == "__main__":
    main()

