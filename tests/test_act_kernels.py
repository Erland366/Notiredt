import notiredt
import pytest
import torch
from torch import nn
from torch.cuda.amp import autocast

from .utils import assert_close, create_input, create_input_like, default_shapes


@pytest.mark.parametrize("shape", default_shapes())
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("act_func", ["Sigmoid"])
def test_sigmoid(
    amp: bool, input_dtype: torch.dtype, shape: tuple[int, ...], act_func: str
):
    if input_dtype is torch.float16 and not amp:
        return

    notiredt_input = create_input(shape, dtype=input_dtype)
    torch_input = create_input(shape, dtype=input_dtype)

    notiredt_act_func = getattr(notiredt, act_func)()
    torch_act_func = getattr(nn, act_func)()

    with autocast(enabled=amp):
        notiredt_output = notiredt_act_func(notiredt_input)
        torch_output = torch_act_func(torch_input)

    assert_close((notiredt_output, torch_output), rtol=1e-3, atol=1e-3)

    # notiredt_output.backward(create_input_like(notiredt_input))
    # torch_output.backward(create_input_like(torch_input))

    # assert_close((notiredt_input.grad, torch_input.grad), rtol=1e-3, atol=1e-3)
