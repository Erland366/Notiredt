import lovely_tensors as lt
import torch
import triton
import triton.language as tl

from notiredt.utils.benchmark import TimeIt, equal

lt.monkey_patch()


@triton.jit
def _reduce_sum_naive(A, B, stride_AX, N):
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
def _reduce_sum(A, B, stride_AX, N):
    row = tl.program_id(0)
    i = row * 2

    for stride in range(N):
        a = tl.load(A + i)
        b = tl.load(A + i + stride)
        c = a + b
        tl.device_print("c", c)
        tl.store(A + i, c)
        tl.debug_barrier()

    a = tl.load(A + i)
    tl.store(B, a)


def reduce_sum(A: torch.FloatTensor):
    assert A.is_cuda
    N = A.shape[0]
    B = torch.zeros(1, device="cuda")
    _reduce_sum[(N,)](A, B, *A.stride(), N)
    return B


print(reduce_sum_naive(torch.arange(10, device="cuda")))
print(reduce_sum(torch.arange(10, device="cuda")))
