import torch
import triton
import triton.language as tl


@triton.jit
def _add(X, Y, Z, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    X = tl.load(X + offsets, mask=mask)
    Y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, X + Y, mask=mask)


def add(X, Y):
    N = X.shape[0]
    assert X.is_cuda and Y.is_cuda
    Z = torch.empty_like(X)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _add[grid](X, Y, Z, N, BLOCK_SIZE=BLOCK_SIZE)
    return Z


def add_torch(x, y):
    return x + y


def main():
    x = torch.randn(1000, device="cuda")
    y = torch.randn(1000, device="cuda")
    z = add(x, y)
    z_torch = add_torch(x, y)
    print(f"{x = }")
    print(f"{y = }")
    print(f"{z = }")
    assert torch.allclose(z, z_torch)
    print("Success! Triton add works correctly")


if __name__ == "__main__":
    main()
