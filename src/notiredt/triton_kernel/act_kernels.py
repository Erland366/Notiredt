import triton
import triton.language as tl
from notiredt.triton_kernel.dropout_kernel import apply_dropout, apply_dropout_grad
from notiredt.triton_kernel.utils import element_wise_kernel_config


@triton.jit
def sigmoid(inp):
    """Applies sigmoid to the input

    Parameters
    ----------
    input : SequenceOrTensor. The input must be loaded and cannot be a pointer

    Returns
    -------
        Input transformed by sigmoid.
    """
    return 1 / (1 + tl.exp(-inp))


@triton.jit
def sigmoid_grad(inp):
    out = sigmoid(inp)
    return out * (1 - out)


@triton.jit
def apply_act_func(inp, drop_p, seed, offset, act_func, dropout):
    if act_func != "relu":
        input_tensor = inp.to(tl.float32)

    if act_func == "sigmoid":
        output = sigmoid(input_tensor)

    if dropout:
        output = apply_dropout(input_tensor, drop_p, seed, offset)
    return output


@triton.autotune(configs=element_wise_kernel_config(), key=["size"])
@triton.jit
def act_func_forward_kernel(
    input_pointer,
    output_pointer,
    size,
    drop_p,
    seed,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    input_tensor = tl.load(input_pointer + offset, mask=mask)
    tl.store(
        output_pointer + offset,
        apply_act_func(input_tensor, drop_p, seed, offset, act_func, dropout),
        mask=mask,
    )


@triton.jit
def apply_act_grad_func(
    out_grad, inp, drop_p, seed, offset, act_func: tl.constexpr, dropout: tl.constexpr
):
    if act_func != "relu":
        inp = inp.to(tl.float32)

    if act_func == "sigmoid":
        out = sigmoid_grad(inp)

    if dropout:
        out_grad = apply_dropout_grad(out_grad, drop_p, seed, offset)

    return out_grad * out


@triton.autotune(configs=element_wise_kernel_config(), key=["size"])
@triton.jit
def act_func_backward_kernel(
    out_grad_ptr,
    inp_ptr,
    out_ptr,
    size,
    drop_p,
    seed,
    act_func: tl.constexpr,
    dropout: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    out_grad = tl.load(out_grad_ptr + offset, mask=mask)
    inp = tl.load(inp_ptr, mask=mask)

    tl.store(
        out_ptr + offset,
        apply_act_grad_func(out_grad, inp, drop_p, seed, offset, act_func, dropout),
        mask=mask,
    )
