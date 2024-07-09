import lovely_tensors as lt
import torch
from loguru import logger

import triton
import triton.language as tl

lt.monkey_patch()
logger.add("debugging_triton.log", filter="triton", level="DEBUG")


@triton.jit
def _vector_addition(A, B, C, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    block_start = row * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + offsets, mask=mask)
    c = a + b
    tl.store(C + offsets, c, mask=mask)


def vector_addition(A: torch.FloatTensor, B: torch.FloatTensor) -> torch.FloatTensor:
    assert A.is_cuda and B.is_cuda
    N = A.shape[0]
    assert N == B.shape[0]
    C = torch.zeros_like(A)

    block_size = 128
    grid_size = triton.cdiv(N, block_size)
    grid = (grid_size,)

    k2 = _vector_addition[grid](
        A,
        B,
        C,  #
        N,  #
        block_size,
    )
    return C


def main():
    # verify numerical fidelity
    torch.manual_seed(2020)
    vec_size = 8192
    a = torch.randn(vec_size, device="cuda")
    b = torch.rand_like(a)
    torch_res = a + b
    triton_res = vector_addition(a, b)
    fidelity_correct = torch.allclose(torch_res, triton_res)
    print(f"{fidelity_correct = }")
    print("VectorAdd complete")


if __name__ == "__main__":
    main()
