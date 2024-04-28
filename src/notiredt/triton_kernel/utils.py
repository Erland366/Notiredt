import triton


def element_wise_kernel_config(
    block_name: str = "BLOCK_SIZE",
) -> list[triton.Config]:
    """Returns kernel configuration for element-wise operations.

    Args:
        block_name (str, optional): Name of block arguments row are distributed over. Defaults to "BLOCK_SIZE".

    Returns:
        list[triton.Config]
    """
    return [
        triton.Config({block_name: 64}, num_warps=2),
        triton.Config({block_name: 128}, num_warps=2),
        triton.Config({block_name: 256}, num_warps=4),
        triton.Config({block_name: 512}, num_warps=4),
        triton.Config({block_name: 1024}, num_warps=4),
    ]
