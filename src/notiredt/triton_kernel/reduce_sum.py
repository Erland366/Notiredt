import lovely_tensors as lt
import torch
import triton
import triton.language as tl

lt.monkey_patch()


@triton.jit
def _reduce_sum_naive(A, B, stride_AX, N: tl.constexpr):
    row = tl.program_id(0)

    sum_ = 0.0
    for k in range(N):
        offsets = row * stride_AX + k
        mask = offsets < N
        a = tl.load(A + offsets, mask=mask)
        sum_ += a
    tl.store(B, sum_)


def reduce_sum_naive(A: torch.FloatTensor):
    assert A.is_cuda
    N = A.shape[0]
    B = torch.zeros(1, device="cuda")
    _reduce_sum_naive[(N,)](A, B, *A.stride(), N)
    return B


@triton.jit
def _reduce_sum(A, B, stride_AX, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    thread_idx = block_start + tl.arange(0, BLOCK_SIZE)

    i = thread_idx * 2

    stride = 1
    while stride < BLOCK_SIZE:
        thread_offset = i + stride
        a = tl.load(A + i, mask=(thread_offset < N) and (thread_idx % stride == 0))
        b = tl.load(
            A + thread_offset, mask=(thread_offset < N) and (thread_idx % stride == 0)
        )
        c = a + b
        tl.store(A + i, c, mask=thread_offset < N)

        tl.debug_barrier()

        stride *= 2

    tl.store(B + thread_idx, tl.load(A + thread_idx))


def reduce_sum(A: torch.FloatTensor):
    assert A.is_cuda
    N = A.shape[0]
    B = torch.zeros(1, device="cuda")
    _reduce_sum[(1,)](A, B, *A.stride(), N, BLOCK_SIZE=32)
    return B


print(reduce_sum_naive(torch.arange(10, device="cuda")))
print(reduce_sum(torch.arange(11, device="cuda")))
