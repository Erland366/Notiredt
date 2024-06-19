import triton
import triton.language as tl

from notiredt.aliases import SequenceOrTensor

from .utils import element_wise_kernel_config


@triton.jit
def sigmoid(input: SequenceOrTensor) -> SequenceOrTensor:
    """Applies sigmoid to the input

    Parameters
    ----------
    input : SequenceOrTensor. The input must be loaded and cannot be a pointer

    Returns
    -------
        Input transformed by sigmoid.
    """
    return 1 / (1 + tl.exp(-input))


@triton.jit
def apply_act_func(input_tensor, drop_p, seed, offset, act_func, dropout):
    if act_func != "relu":
        input_tensor = input_tensor.to(tl.float32)

    if act_func == "sigmoid":
        output = sigmoid(input_tensor)

    if dropout:
        output = apply_dropout()
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
