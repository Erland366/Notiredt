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
    configs=[linear_forward_config(32, 32, 32, n_warps=2, n_stages=2)],
    key=["batch_dim", "in_feat_dim", "out_feat_dim", "fp16"],
)
@triton.heuristics({"tf32": lambda _: allow_tf32()})
@triton.jit
def linear_forward_kernel():
    pass
