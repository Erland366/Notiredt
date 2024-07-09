import torch
import triton
from torch import Tensor


def element_wise_kernel_config(
    block_name: str = "BLOCK_SIZE",
) -> list[triton.Config]:
    """
    Returns a list of Triton configurations for element-wise kernels.

    Args:
        block_name (str): The name of the block size parameter in the configuration.

    Returns:
        list[triton.Config]: A list of Triton configurations.
    """
    # Define the configurations for different block sizes.
    return [
        triton.Config({block_name: 64}, num_warps=2),  # Block size = 64, num warps = 2
        triton.Config(
            {block_name: 128}, num_warps=2
        ),  # Block size = 128, num warps = 2
        triton.Config(
            {block_name: 256}, num_warps=4
        ),  # Block size = 256, num warps = 4
        triton.Config(
            {block_name: 512}, num_warps=4
        ),  # Block size = 512, num warps = 4
        triton.Config(
            {block_name: 1024}, num_warps=4
        ),  # Block size = 1024, num warps = 4
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


def is_valid(x: Tensor) -> Tensor:
    """
    Check if the input tensor is valid for GPU computations.

    Args:
        x (torch.Tensor): Input tensor.

    Raises:
        AssertionError: If the tensor is not on the GPU.
        AssertionError: If the tensor is not contiguous.

    Returns:
        torch.Tensor: The input tensor.
    """
    # Check if the tensor is on the GPU
    assert x.is_cuda, "Tensor is not in GPU, use `.to('cuda')` on this Tensor!"

    # Check if the tensor is contiguous
    assert (
        x.is_contiguous()
    ), "Tensor is not contiguous, use `.contiguous()` on this Tensor!"


def set_seed(seed: int) -> None:
    """
    Set seed for all random number generators.

    Args:
        seed (int): Seed value.
    """

    # Set seed for Python's random module
    import random

    random.seed(seed)

    # Set seed for PyTorch
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Set seed for NumPy
    import numpy as np

    np.random.seed(seed)
