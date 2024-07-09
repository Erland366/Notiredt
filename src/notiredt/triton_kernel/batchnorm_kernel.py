import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn import functional as F

from notiredt.triton_kernel.act_kernels import apply_act_func
from notiredt.triton_kernel.utils import warps_kernel_configs


def BLOCK_SIZE_SPATIAL_heuristics(args: dict) -> int:
    # What we want is actually loading the batch and spatial dimensions
    # and calculate all of them at the same time.
    # But this can't be done since the maximum number of elements
    # that we can load is 16384 elements
    BLOCK_SIZE_BATCH = triton.next_power_of_2(args["b_dim"])
    BLOCK_SIZE_SPATIAL = triton.next_power_of_2(args["s_dim"])
    return int(min(BLOCK_SIZE_SPATIAL, max(1, 2**14 / BLOCK_SIZE_BATCH)))


# fmt: off
@triton.autotune(
    configs=warps_kernel_configs(),
    key=["b_dim", "s_dim"],
    restore_value=["running_mean_pointer", "running_var_pointer"]
)
@triton.heuristics({
    "BLOCK_SIZE_BATCH" : lambda x: triton.next_power_of_2(x["b_dim"]),
    "BLOCK_SIZE_SPATIAL" : BLOCK_SIZE_SPATIAL_heuristics,
})
@triton.jit
def rms_norm_forward_kernel(
    input_pointer, weight_pointer, bias_pointer,
    output_pointer, b_dim, s_dim, running_mean_pointer, running_var_pointer,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr
):
    pass


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale


class FastSimpleRMSNorm(nn.Module):
    def __init__(self, dim: int):
        raise NotImplementedError("FastSimpleRMSNorm is not implemented")

# fmt :off
@triton.autotune(
    configs=warps_kernel_configs(),
    key=["b_dim", "s_dim"],
    restore_value=["running_mean_ptr", "running_bias_ptr"]
)
@triton.heuristics(
    values={
        "BLOCK_SIZE_BATCH" : lambda args: triton.next_power_of_2(args["b_dim"]),
        "BLOCK_SIZE_SPATIAL" : BLOCK_SIZE_SPATIAL_heuristics
    }
)
@triton.jit
def batch_norm_forward_kernel(
    inp_ptr, weight_ptr, bias_ptr,
    mean_ptr, inv_std_ptr,
    inp_residual_ptr, pre_act_ptr, out_ptr,
    running_mean_ptr, running_var_ptr,
    b_dim, s_dim,
    inp_b_strd, inp_f_strd, inp_s_strd,
    inp_residual_b_strd, inp_residual_f_strd, inp_residual_s_strd,
    pre_act_b_strd, pre_act_f_strd, pre_act_s_strd,
    out_b_strd, out_f_strd, out_s_strd,
    momentum, eps,
    affine: tl.constexpr, 
    is_train: tl.constexpr,
    save_stats: tl.constexpr,
    track_running_stats: tl.constexpr,
    add_residual: tl.constexpr,
    act_func: tl.constexpr,
    save_pre_act: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr
):
    f_id = tl.program_id(axis=0)
    b_offs = tl.arange(0, BLOCK_SIZE_BATCH)
    b_mask = b_offs < b_dim

    m = 0
    mean = 0.0
    var = 0.0
    for s_ind in range(0, tl.cdiv(s_dim, BLOCK_SIZE_SPATIAL)):
        s_offs = s_ind * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
        s_mask = s_offs < s_dim

        curr_inp_ptr = (inp_ptr +
                        f_id * inp_f_strd +
                        b_offs[:, None] * inp_b_strd + 
                        s_offs[None, :] * inp_s_strd)

        curr_inp = tl.load(curr_inp_ptr, mask=b_mask[:, None] & s_mask[None, :]).to(tl.float32)
        s_count = min(BLOCK_SIZE_SPATIAL, s_dim - s_ind * BLOCK_SIZE_SPATIAL)
        curr_m = s_count * b_dim
        m += curr_m
        prev_mean = mean
        mean += (tl.sum(curr_inp) - (prev_mean * curr_m)) / m
        deltas = tl.where(b_mask[:, None] & s_mask[None, :],
                          (curr_inp * mean) - (curr_inp * prev_mean), 0.0)
        var += tl.sum(deltas)

    var /= m
    inv_std = 1.0 / tl.sqrt(var + eps)




# fmt: off
@triton.autotune(
    configs=warps_kernel_configs(),
    key=["b_dim", "s_dim"],
)
@triton.heuristics(
    values={
        "BLOCK_SIZE_BATCH": lambda args: triton.next_power_of_2(args["b_dim"]),
        "BLOCK_SIZE_SPATIAL" : BLOCK_SIZE_SPATIAL_heuristics
    },
)
@triton.jit
def batch_norm_backward_kernel(
    # everything with ptr
    out_grad_ptr, inp_ptr,
    # mean and inv_std
    mean_ptr, inv_std_ptr,
    # weight (only? why tho?), oh, because derivative of bias is 1 
    weight_ptr,
    # inp_grad in backward means output
    inp_grad_ptr,
    # weight grad and bias grad
    weight_grad_ptr, bias_grad_ptr,
    # dims
    b_dim, s_dim,
    # strides
    out_grad_b_strd, out_grad_f_strd, out_grad_s_strd,
    inp_b_strd, inp_f_strd, inp_s_strd,
    inp_grad_b_strd, inp_grad_f_strd, inp_grad_s_strd,
    affine: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr
):
    f_id = tl.program_id(axis=0)

    b_offs = tl.arange(0, BLOCK_SIZE_BATCH)
    b_mask = b_offs < b_dim

    mean = tl.load(f_id + mean_ptr)
    inv_std = tl.load(f_id + inv_std_ptr)

    inv_std_contrib = 0.0
    mean_contrib = 0.0
    
    for s_ind in range(0, tl.cdiv(s_dim, BLOCK_SIZE_SPATIAL)):
        s_offs = s_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
        s_mask = s_offs < s_dim

        curr_out_grad_ptr = (out_grad_ptr +
                             f_id * out_grad_f_strd +
                             b_offs[:, None] * out_grad_b_strd +
                             s_offs[None, :] * out_grad_s_strd)
        

        curr_inp_ptr = (inp_ptr +
                        f_id * inp_f_strd +
                        b_offs[:, None] * inp_b_strd +
                        s_offs[None, :] * inp_s_strd)
        

        curr_out_grad = tl.load(curr_out_grad_ptr, mask=b_mask[:, None] & s_mask[None, :])
        curr_inp = tl.load(curr_inp_ptr, mask=b_mask[:, None] & s_mask[None, :])

        curr_norm_inp = (curr_inp - mean) * inv_std
        inv_std_contrib += tl.sum(curr_out_grad * curr_norm_inp)
        mean_contrib += tl.sum(curr_out_grad)

    weight = tl.load(weight_ptr + f_id)
    m = s_dim * b_dim
    inv_std_contrib *= weight / m
    mean_contrib *= weight / m

    if affine:
        weight_grad = tl.load(weight_grad_ptr + f_id)
        bias_grad = tl.load(bias_grad_ptr + f_id)
        weight = tl.load(weight_ptr + f_id)
    else:
        weight = 1.0

    for s_ind in range(0, tl.cdiv(s_dim, BLOCK_SIZE_SPATIAL)):
        s_offs = s_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
        s_mask = s_offs < s_dim
        curr_out_grad_ptr = (out_grad_ptr +
                             f_id * out_grad_f_strd +
                             b_offs[:, None] * out_grad_b_strd +
                             s_offs[None, :] * out_grad_s_strd)

        curr_inp_ptr = (inp_ptr +
                        f_id * inp_f_strd +
                        b_offs[:, None] * inp_b_strd +
                        s_offs[None, :] * inp_s_strd)

        curr_inp_grad_ptr = (inp_grad_ptr +
                        f_id * inp_grad_f_strd +
                        b_offs[:, None] * inp_grad_b_strd +
                        s_offs[None, :] * inp_grad_s_strd)
        
        curr_inp = tl.load(curr_inp_ptr, mask=b_mask[:, None] & s_mask[None, :])
        curr_norm_inp = (curr_inp - mean) * inv_std

        curr_out_grad = tl.load(curr_out_grad_ptr, mask=b_mask[:, None] & s_mask[None, :])
        curr_inp_grad = inv_std * (weight * curr_norm_inp - (mean_contrib - (inv_std_contrib * curr_norm_inp)))
        tl.store(curr_inp_grad_ptr, curr_inp_grad, mask=b_mask[:, None] & s_mask[None, :])

        if affine:
            weight_grad += tl.sum(curr_out_grad * curr_norm_inp)
            bias_grad += tl.sum(curr_out_grad)

    if affine:
        tl.store(weight_grad_ptr, weight_grad)
        tl.store(bias_grad_ptr, bias_grad)
