import torch

import triton
import triton.language as tl


@triton.jit
def mean_kernel_batch_major(
    input_ptr, output_ptr, batch_size, spatial_size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    # Initialize accumulator
    acc = 0.0
    count = 0

    # Iterate over the batch dimension
    for i in range(0, batch_size, BLOCK_SIZE):
        batch_offset = i + tl.arange(0, BLOCK_SIZE)
        batch_mask = batch_offset < batch_size

        # Load and accumulate
        x = tl.load(
            input_ptr + pid * spatial_size * batch_size + batch_offset * spatial_size,
            mask=batch_mask,
        )
        acc += tl.sum(x * batch_mask, axis=0)
        count += tl.sum(batch_mask, axis=0)

    # Compute and store mean
    mean = acc / count
    tl.store(output_ptr + pid, mean)


def mean_triton(x, layout="batch_major"):
    output = torch.empty(
        x.shape[1] if layout == "batch_major" else x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    if layout == "batch_major":
        mean_kernel_batch_major[(x.shape[1],)](
            x, output, x.shape[0], x.shape[1], BLOCK_SIZE=32
        )
    else:
        mean_kernel_spatial_major[(x.shape[0],)](
            x, output, x.shape[0], x.shape[1], BLOCK_SIZE=32
        )

    return output


@triton.jit
def mean_kernel_batch_major_2(
    inp_ptr,
    out_ptr,
    inp_b_strd,
    inp_s_strd,
    out_b_strd,
    s_dim,
    BLOCK_SIZE: tl.constexpr,
):
    b_pid = tl.program_id(0)

    count = 0
    mean = 0.0

    for block_ind in range(0, tl.cdiv(s_dim, BLOCK_SIZE)):
        s_offs = block_ind * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        s_mask = s_offs < s_dim

        curr_inp_ptr = inp_ptr + b_pid * inp_b_strd + s_offs * inp_s_strd
        curr_inp = tl.load(curr_inp_ptr, mask=s_mask, other=0.0)

        s_count = min(BLOCK_SIZE, s_dim - BLOCK_SIZE * block_ind)
        count += s_count

        prev_mean = mean

        mean += (tl.sum(curr_inp) - (s_count * prev_mean)) / count

    tl.store(
        out_ptr + b_pid * out_b_strd,
        mean,
    )


@triton.jit
def mean_kernel_spatial_major(  # [b, s]
    inp_ptr,
    out_ptr,
    inp_b_strd,
    inp_s_strd,
    out_b_strd,
    batch_dim,
    BLOCK_SIZE: tl.constexpr,
):
    s_pid = tl.program_id(0)
    b_offs = tl.arange(0, BLOCK_SIZE)
    b_mask = b_offs < batch_dim

    curr_inp_ptr = inp_ptr + s_pid * inp_s_strd + b_offs * inp_b_strd
    curr_inp = tl.load(curr_inp_ptr, mask=b_mask)
    s_dim = min(batch_dim, tl.cdiv(inp_b_strd, BLOCK_SIZE))
    mean = tl.sum(curr_inp) / s_dim

    curr_out_ptr = out_ptr + s_pid + b_offs * out_b_strd

    tl.store(curr_out_ptr, mean, mask=b_mask)


def mean_kernel(x):
    out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    mean_kernel_batch_major_2[(x.shape[0],)](
        x, out, *x.stride(), *out.stride(), x.shape[1], BLOCK_SIZE=32
    )
    return out


# Test the kernels
batch_size, spatial_size = 1024, 256
x_batch_major = torch.randn(batch_size, spatial_size, device="cuda")
x_spatial_major = x_batch_major.t().contiguous()

# Compute means using Triton kernels
# mean_batch_major = mean_triton(x_batch_major, layout="batch_major")
# mean_spatial_major = mean_triton(x_spatial_major, layout="spatial_major")

# Compute mean using PyTorch for verification
torch_mean = x_batch_major.mean(0)

# print(
#     "Max difference (batch major):", (mean_batch_major - torch_mean).abs().max().item()
# )
# print(f"{mean_spatial_major = }")
# print(
#     "Max difference (spatial major):",
#     (mean_spatial_major - torch_mean).abs().max().item(),
# )

triton_mean = mean_kernel(x_spatial_major)
print(f"{triton_mean = }")
print(f"{torch_mean = }")

print("Max difference (batch major):", (triton_mean - torch_mean).abs().max().item())
