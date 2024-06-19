import triton
import triton.language as tl

from notiredt.triton_kernel.utils import element_wise_kernel_config


@triton.jit
def apply_dropout(input_tensor, drop_p, seed, offset):
    """
    Randomly zeroes elements in the input.

    Args:
        input_tensor (tl.tensor): Which is our tensor
        drop_p (float): The threshold for the dropped out value
        seed ():
        offset ():
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input_tensor / (1 - drop_p))


@triton.jit
def apply_dropout_grad(output_grad, drop_p, seed, offset):
    """

    Args:
        output_grad ():
        drop_p ():
        seed ():
        offset ():
    """
    random = tl.randn(seed, offset)
    return tl.where(random < drop_p, 0, output_grad / (1 - drop_p))


@triton.autotune(
    configs=element_wise_kernel_config(),
    key=["size"],
)
@triton.jit
def dropout_forward_kernel(
    input_pointer, output_pointer, size, drop_p, seed, BLOCK_SIZE: tl.constexpr
):
    """

    Args:
        input_pointer ():
        output_pointer ():
        size ():
        drop_p ():
        seed ():
        BLOCK_SIZE:
    """
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input_tensor = tl.load(input_pointer + offset, mask=mask)
    output_tensor = apply_dropout(input_tensor, drop_p, seed, offset)
    tl.store(output_pointer + offset, output_tensor, mask=mask)
