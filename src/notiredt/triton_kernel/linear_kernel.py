import triton
import triton.language as tl

from .act_kernels import apply_act_func
from .utils import allow_tf32, get_n_stages


def linear_forward_config(
    BLOCK_SIZE_BATCH: int,
    BLOCK_SIZE_IN_FEAT: int,
    BLOCK_SIZE_OUT_FEAT: int,
    GROUP_SIZE_BATCH: int = 8,
    n_warps: int = 4,
    n_stages: int = 2,
) -> triton.Config:
    """

    Args:
        BLOCK_SIZE_BATCH:
        BLOCK_SIZE_IN_FEAT:
        BLOCK_SIZE_OUT_FEAT:
        GROUP_SIZE_BATCH:
        n_warps:
        n_stages:

    Returns:

    """
    return triton.Config(
        {
            "BLOCK_SIZE_BATCH": BLOCK_SIZE_BATCH,
            "BLOCK_SIZE_IN_FEAT": BLOCK_SIZE_IN_FEAT,
            "BLOCK_SIZE_OUT_FEAT": BLOCK_SIZE_OUT_FEAT,
            "GROUP_SIZE_BATCH": GROUP_SIZE_BATCH,
        },
        num_warps=n_warps,
        num_stages=get_n_stages(n_stages),
    )


@triton.autotune(
    configs=[
        linear_forward_config(32, 32, 32, n_warps=2, n_stages=2),
        linear_forward_config(64, 32, 32, n_warps=2, n_stages=5),
        linear_forward_config(64, 32, 128, n_warps=4, n_stages=4),
        linear_forward_config(64, 32, 256, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 32, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 64, n_warps=4, n_stages=4),
        linear_forward_config(128, 32, 128, n_warps=4, n_stages=4),
        linear_forward_config(128, 64, 256, n_warps=8, n_stages=3),
    ],
    key=["batch_dim", "in_feat_dim", "out_feat_dim", "fp16"],
)
@triton.heuristics({"tf32": lambda _: allow_tf32()})
@triton.jit
def linear_forward_kernel(
    # Pointers
    input_pointer,
    weight_pointer,
    bias_pointer,
    pre_act_pointer,
    output_pointer,
    # dims
    batch_dim,
    in_feat_dim,
    out_feat_dim,
    # strides
    input_batch_stride,
    input_in_batch_stride,
    weight_in_feat_stride,
    weight_out_feat_stride,
    pre_act_batch_stride,
    pre_act_out_feat_stride,
    output_batch_stride,
    output_out_feat_stride,
    # constexprs
    add_bias: tl.constexpr,
    act_func: tl.constexpr,
    save_pre_act: tl.constexpr,
    fp16: tl.constexpr,
    tf32: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    GROUP_SIZE_BATCH: tl.constexpr,
):
    """

    Args:
        input_pointer ():
        weight_pointer ():
        bias_pointer ():
        pre_act_pointer ():
        output_pointer ():
        batch_dim ():
        in_feat_dim ():
        out_feat_dim ():
        input_batch_stride ():
        input_in_batch_stride ():
        weight_in_feat_stride ():
        weight_out_feat_stride ():
        pre_act_batch_stride ():
        pre_act_out_feat_stride ():
        output_batch_stride ():
        output_out_feat_stride ():
        add_bias:
        act_func:
        save_pre_act:
        fp16:
        tf32:
        BLOCK_SIZE_BATCH:
        BLOCK_SIZE_IN_FEAT:
        BLOCK_SIZE_OUT_FEAT:
        GROUP_SIZE_BATCH:
    """
    # Programs are blocked together, GROUP_SIZE_BATCH at at time
    # to alleviate L2 Miss rates
    pid = tl.program_id(axis=0)
