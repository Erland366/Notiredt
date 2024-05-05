import torch
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


def allow_tf32() -> bool:
    """
    Returns whether the current GPU architecture
    """
    return torch.cuda.get_device_capability()[0] >= 8


def get_n_stages(n_stages: int = 2) -> int:
    """
    Receives number of stages for software pipelining and returns it as-is
    if the GPU architecture is Ampere or newer and 2 otherwise.
    """
    return 2 if torch.cuda.get_device_capability()[0] < 8 else n_stages


def get_output_dtype(
    input_dtype: torch.dtype = torch.float32, autocast: str | None = None
) -> torch.dtype:
    """

    Args:
        input_dtype: Input dtype
        autocast: The relevant operation's autocast behavior.
            None signifies the input dtype should flow through,
            'fp16' signifies autocasting to FP16 when AMP is enabled.

    Raises:
        RuntimeError:

    Returns:

    """
    assert torch.get_autocast_gpu_dtype(), f"Only autocast to float16 is supported, received {torch.get_autocast_gpu_dtype()}"

    if torch.is_autocast_enabled():
        if autocast is None:
            return input_dtype

        elif autocast == "fp16":
            return torch.float16

        elif autocast == "fp32":
            return torch.float16

        else:
            raise RuntimeError(
                f"Autocast type {autocast} is invalid. "
                "Options are None, fp16, and fp32"
            )

    else:
        return input_dtype


def warps_kernel_configs() -> list[triton.Config]:
    """

    Returns: Kernel configurations with all possible number of warps

    """
    return [triton.Config({}, num_warps=2**i) for i in range(6)]
