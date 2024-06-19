import torch
import triton
import triton.language as tl
import triton_util as tu


@triton.jit
def load_2d_kernel(
    x_ptr, y_ptr, x_size_0, x_size_1, x_stride_0, x_stride_1, BLOCK_SIZE: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    x = tu.load_2d(
        x_ptr, BLOCK_SIZE, BLOCK_SIZE, pid_0, pid_1, x_size_0, x_size_1, x_stride_0
    )

    x += pid_0 * pid_1

    y_offsets = tu.get_2d_offset(
        tu.get_1d_offset(BLOCK_SIZE, pid_0),
        tu.get_1d_offset(BLOCK_SIZE, pid_1),
        x_stride_0,
        x_stride_1,
    )

    tl.store(y_ptr + y_offsets, x)


def load_2d():
    a = torch.zeros(16, 16, device="cuda", dtype=torch.float16)
    x_size_0, x_size_1 = a.size()
    x_stride_0, x_stride_1 = a.stride()
    b = torch.empty_like(a)

    load_2d_kernel[(tu.cdiv(x_size_0, 4), tu.cdiv(x_size_1, 4))](
        a,
        b,
        x_size_0,
        x_size_1,
        x_stride_0,
        x_stride_1,
        4,  # type: ignore
    )

    print(b)


if __name__ == "__main__":
    load_2d()
