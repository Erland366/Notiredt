import notiredt
import pytest
import torch
from notiredt.triton_kernel.dropout_kernel import (
    dropout_backward_kernel,
    dropout_forward_kernel,
)
from notiredt.triton_kernel.utils import is_valid, set_seed
from torch import Tensor

import triton

from .utils import (
    assert_close,
    create_input,
    create_input_like,
    create_zeros_like,
    default_shapes,
)

# @pytest.mark.parametrize("shape", default_shapes())
# @pytest.mark.parametrize("drop_p", [0.0, 0.15, 0.3, 0.5, 0.75, 0.9, 1.0])
# def test_dropout_layer(shape: Tuple[int, ...], drop_p: float) -> None:
#     input = create_input(shape)
#     dropout = attorch.Dropout(drop_p)
#     output = dropout(input)
#     n_zeroed = (torch.count_nonzero(input) - torch.count_nonzero(output)).item()

#     if drop_p == 0:
#         assert n_zeroed == 0


def use_dropout_forward_kernel(inp: Tensor, drop_p: float, seed: int) -> Tensor:
    is_valid(inp)

    flat_inp = inp.flatten()
    size = len(flat_inp)
    out = torch.empty_like(flat_inp)

    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    dropout_forward_kernel[grid](flat_inp, out, size, drop_p, seed)

    return out.view_as(inp)


def use_dropout_backward_kernel(inp: Tensor, drop_p: float, seed: int) -> Tensor:
    is_valid(inp)

    assert inp.requires_grad, "Backward means that it requires to have gradient vector!"

    flat_inp = inp.flatten()
    size = len(flat_inp)
    out = torch.empty_like(flat_inp)

    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    dropout_backward_kernel[grid](inp.grad, out, size, drop_p, seed)

    return out.view_as(inp)


SEED_GLOBAL = 42
set_seed(SEED_GLOBAL)


@pytest.mark.parametrize("shape", default_shapes())
@pytest.mark.parametrize("p", [0.0, 0.15, 0.3, 0.5, 0.75, 0.9, 1.0])
@pytest.mark.kernel
def test_dropout_forward_kernel(shape, p) -> None:
    inp = create_input(shape)

    notiredt_res = use_dropout_forward_kernel(inp, p, SEED_GLOBAL)

    n_zeroed = (torch.count_nonzero(inp) - torch.count_nonzero(notiredt_res)).item()

    assert_close((n_zeroed / inp.numel(), p), rtol=1e-1, atol=5e-2)


@pytest.mark.parametrize("shape", default_shapes())
@pytest.mark.parametrize("p", [0.0, 0.15, 0.3, 0.5, 0.75, 0.9, 1.0])
@pytest.mark.kernel
def test_dropout_backward_kernel(shape, p) -> None:
    inp = create_input(shape)

    out_grad = create_input_like(inp)
    inp.backward(out_grad)

    inp_grad = use_dropout_backward_kernel(inp, p, SEED_GLOBAL)

    n_zeroed = (torch.count_nonzero(inp.grad) - torch.count_nonzero(inp_grad)).item()

    assert_close((n_zeroed / inp.numel(), p), rtol=1e-1, atol=5e-2)


@pytest.mark.parametrize("shape", default_shapes())
@pytest.mark.parametrize("p", [0.0, 0.15, 0.3, 0.5, 0.75, 0.9, 1.0])
def test_dropout_layer(shape, p) -> None:
    inp = create_input(shape)
    dropout = notiredt.Dropout(p)
    out = dropout(inp)

    n_zeroed = (torch.count_nonzero(inp) - torch.count_nonzero(out)).item()

    if p == 0:
        assert n_zeroed == 0

    elif p == 1:
        assert torch.count_nonzero(out).item() == 0

    else:
        assert_close((out, torch.where(out == 0, out, inp / (1 - p))))
        assert_close((n_zeroed / inp.numel(), p), rtol=1e-1, atol=5e-2)

    out_grad = create_input_like(out)
    out.backward(out_grad)
    inp_grad = torch.where(out == 0, out, out_grad / (1 - p))

    assert_close((inp.grad, inp_grad))
