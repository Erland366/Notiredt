import torch
from fastcore.script import call_parse

import triton
import triton.language as tl


@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)


def load_triton_kernel() -> None:
    x = torch.randn(1000, device="cuda")
    y = torch.randn(1000, device="cuda")
    z = torch.empty_like(y)

    BLOCK_SIZE = 256
    triton_[(triton.cdiv(1000, 32),)](x, y, z, 1000, XBLOCK=BLOCK_SIZE)

    assert torch.allclose(z, x * x + y * y)


@call_parse
def main(
    b=32,
) -> None:
    load_triton_kernel()
