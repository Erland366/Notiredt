"""
Utilities for Testing
"""

import pytest
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

import notiredt

from .utils import assert_close, create_input, create_input_like, default_shapes

GLOBAL_SEED = 3407


@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "act_func", [None, "relu", "sigmoid", "tanh", "relu", "gelu", "silu"]
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("out_dim", [16, 96, 128, 196, 384, 512, 768, 1024])
@pytest.mark.parametrize("input_shape", default_shapes(min_dim=2, max_dim=3))
def test_linear_layer(
    input_shape: tuple[int, ...],
    out_dim: int,
    bias: bool,
    act_func: str | None,
    input_dtype: bool,
    amp: bool,
) -> None:
    if input_dtype is torch.float16 and not amp:
        return
    notiredt_input = create_input(input_shape, dtype=input_dtype)
    pytorch_input = create_input(input_shape, dtype=input_dtype)

    torch.manual_seed(GLOBAL_SEED)
    notiredt_linear = notiredt.Linear(
        input_shape[-1], out_dim, bias=bias, device="cuda"
    )
    torch.manual_seed(GLOBAL_SEED)
    pytorch_linear = nn.Linear(input_shape[-1], out_dim, bias=bias, device="cuda")
    pytorch_act = nn.Identity() if act_func is None else getattr(F, act_func)

    with autocast(enabled=amp):
        notiredt_output = notiredt_linear(notiredt_input)
        pytorch_output = pytorch_act(pytorch_linear(notiredt_input))

    assert_close((notiredt_output, pytorch_output), rtol=1e-3, atol=1e-3)

    notiredt_output.backward(create_input_like(notiredt_output))
    pytorch_output.backward(create_input_like(pytorch_output))

    bias_grad_pair = (
        (notiredt_linear.bias.grad, pytorch_output.bias.grad) if bias else (None, None)
    )
    assert_close(
        (notiredt_input.grad, pytorch_input.grad),
        (notiredt_linear.weight.grad, pytorch_linear.weight.grad.T.contiguous()),
        bias_grad_pair,
        rtol=1e-3,
        atol=1e-3,
    )
