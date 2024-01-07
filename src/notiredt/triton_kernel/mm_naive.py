import lovely_tensors as lt
import torch
import triton
import triton.language as tl

from notiredt.utils.benchmark import TimeIt, equal

lt.monkey_patch()


@triton.jit
def _mm_naive(
    A, B, C, stride_AX, stride_AY, stride_BX, stride_BY, stride_CX, stride_CY, N
):
    row, col = tl.program_id(0), tl.program_id(1)

    sum_ = 0.0
    for k in range(N):
        a = tl.load(A + row * stride_AX + k)
        b = tl.load(B + k * stride_BX + col)
        sum_ += a * b
    c = tl.load(C + row * stride_CX + col)
    c += sum_
    tl.store(C + row * stride_CX + col, c)


def mm_naive_triton(A: torch.FloatTensor, B: torch.FloatTensor):
    assert (
        A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]
    ), "Shape must be the same for all matrix"
    assert A.is_cuda and B.is_cuda
    N = A.shape[1]
    C = torch.zeros_like(A)
    _mm_naive[(N, N)](A, B, C, *A.stride(), *B.stride(), *C.stride(), A.shape[0])
    return C


# ========================
def mm_naive_cpu(A: torch.FloatTensor, B: torch.FloatTensor):
    assert (
        A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]
    ), "Shape must be the same for all matrix"
    assert not A.is_cuda and not B.is_cuda
    C = torch.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C


def mm_torch_cpu(A: torch.FloatTensor, B: torch.FloatTensor):
    assert (
        A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]
    ), "Shape must be the same for all matrix"
    assert not A.is_cuda and not B.is_cuda
    return torch.mm(A, B)


def mm_torch_gpu(A: torch.FloatTensor, B: torch.FloatTensor):
    assert (
        A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1]
    ), "Shape must be the same for all matrix"
    assert A.is_cuda and B.is_cuda
    return torch.matmul(A, B)


def benchmark_mm_naive():
    A = torch.rand(16, 16)
    B = torch.rand(16, 16)

    A_cuda = A.cuda()
    B_cuda = A.cuda()

    with TimeIt("mm_naive_cpu", "yellow"):
        print("mm_naive_cpu", mm_naive_cpu(A, B))

    with TimeIt("mm_naive", "bold green"):
        print("mm_naive", mm_naive_triton(A_cuda, B_cuda))

    with TimeIt("mm_torch_cpu", "bold yellow"):
        print("mm_torch_cpu", mm_torch_cpu(A, B))

    with TimeIt("mm_torch_gpu", "green"):
        print("mm_torch_gpu", mm_torch_gpu(A_cuda, B_cuda))

    print(
        "triton_right",
        equal(mm_torch_gpu(A_cuda, B_cuda), mm_naive_triton(A_cuda, B_cuda)),
    )


benchmark_mm_naive()
