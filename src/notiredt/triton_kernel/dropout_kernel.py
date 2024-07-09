import triton
import triton.language as tl
from notiredt.triton_kernel.utils import element_wise_kernel_config


@triton.jit
def apply_dropout(inp, drop_p, seed, offset):
    # dropout in neural network turns out scale the rest
    # of the tensor that's not dropped out by 1 / (1 - drop_p)
    # so the total value across this network is stays the same
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, inp / (1 - drop_p))


@triton.jit
def apply_dropout_grad(out_grad, drop_p, seed, offset):
    # grad dropout is out_grad * (1 / (1 - drop_p))
    # basically the same as forward pass, but now we use
    # out_grad instead of inp
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0.0, out_grad * (1 / (1 - drop_p)))


@triton.autotune(
    configs=element_wise_kernel_config(),
    key=["size"],
)
@triton.jit
def dropout_forward_kernel(
    inp_ptr, out_ptr, size, drop_p, seed, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input_tensor = tl.load(inp_ptr + offset, mask=mask)
    output_tensor = apply_dropout(input_tensor, drop_p, seed, offset)
    tl.store(out_ptr + offset, output_tensor, mask=mask)


# fmt:off
@triton.autotune(
    configs=element_wise_kernel_config(),
    key=["size"],
)
@triton.jit
def dropout_backward_kernel(
    out_grad_ptr, 
    inp_grad_ptr, 
    size,
    drop_p,
    seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < size

    out_grad = tl.load(out_grad_ptr + offs, mask=mask)
    inp_grad = apply_dropout_grad(out_grad, drop_p, seed, offs)

    tl.store(inp_grad_ptr + offs, inp_grad, mask=mask)
