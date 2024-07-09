import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import lovely_tensors as lt
import notiredt
import pytest
import torch
from notiredt.triton_kernel.batchnorm_kernel import (
    batch_norm_backward_kernel,
    batch_norm_forward_kernel,
)
from notiredt.triton_kernel.utils import is_valid, set_seed
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import init

lt.monkey_patch()

from .utils import assert_close, create_input, default_act_func, default_shapes


def use_batch_norm_forward_kernel(inp: Tensor) -> Tensor:
    is_valid(inp)

    momentum = 1e-1
    eps = 1e-5

    out = torch.empty_like(inp)
    b_dim, f_dim, s_dim = inp.shape

    weight = create_input(f_dim)
    bias = create_input(f_dim)

    running_mean = create_input(f_dim, requires_grad=False)
    running_var = create_input(f_dim, requires_grad=False)

    mean = create_input(f_dim)
    var = create_input(f_dim)

    pre_act = torch.empty_like(inp)
    pre_act_add = torch.empty_like(inp)

    grid = lambda _: (f_dim,)

    # fmt: off
    batch_norm_forward_kernel[grid](
        inp, weight, bias,
        mean, var,
        pre_act_add,
        pre_act,
        out,
        running_mean,
        running_var,
        b_dim, s_dim,
        *inp.stride(),
        *pre_act_add.stride(),
        *pre_act.stride(),
        *out.stride(),
        momentum=momentum,
        eps=eps,
        affine=True,
        act_func="sigmoid",
        save_stats=True,
        track_running_stats=True,
        is_train=True,
        add_pre_act=True,
        save_pre_act=True,
    )


def use_batch_norm_backward_kernel(inp: Tensor) -> Tensor:
    is_valid(inp)

    b_dim, f_dim, s_dim = inp.shape
    out_grad = create_input((b_dim, f_dim, s_dim))
    inp_grad = torch.empty_like(inp)
    weight = create_input(f_dim)

    weight_grad = create_input(f_dim)
    bias_grad = create_input(f_dim)

    mean = create_input(f_dim)
    inv_std = create_input(f_dim)

    affine = True

    grid = lambda _: (f_dim,)

    # fmt: off
    batch_norm_backward_kernel[grid](
        out_grad, inp,
        mean, inv_std,
        weight,
        inp_grad,
        weight_grad,
        bias_grad,
        b_dim, s_dim,
        *out_grad.stride(),
        *inp.stride(),
        *inp_grad.stride(),
        affine=affine,
    )

    return inp_grad


@pytest.mark.parametrize("shape", ((10, 20, 30, 40),))
@pytest.mark.experiment
def test_batch_norm_forward_2d_shape(shape):
    inp = create_input(shape)

    torch_bn = notiredt.BatchNorm2d(inp.shape[1], device="cuda", affine=True)

    torch_res = torch_bn(inp)

    print()
    print(torch_res)
    print(f"{torch_res.shape = }")

    print(f"{torch_bn.weight.shape = }")
    print(f"{torch_bn.bias.shape = }")

    print(f"{torch_bn.running_mean.shape = }")
    print(f"{torch_bn.running_var.shape = }")


@pytest.mark.parametrize("shape", ((10, 20),))
@pytest.mark.experiment
def test_batch_norm_forward_1d_shape(shape):
    inp = create_input(shape)

    torch_bn = notiredt.BatchNorm1d(inp.shape[1], device="cuda", affine=True)

    torch_res = torch_bn(inp)

    breakpoint()

    print()
    print(torch_res)
    print(f"{torch_res.shape = }")

    print(f"{torch_bn.weight.shape = }")
    print(f"{torch_bn.bias.shape = }")

    print(f"{torch_bn.running_mean.shape = }")
    print(f"{torch_bn.running_var.shape = }")


@pytest.mark.parametrize("shape", ((10, 20, 30),))
@pytest.mark.kernel
def test_batch_norm_forward(shape):
    inp = create_input(shape)

    notiredt_res = use_batch_norm_forward_kernel(inp)

    print(f"{notiredt_res = }")


@pytest.mark.parametrize("shape", ((10, 20, 30),))
@pytest.mark.kernel
def test_batch_norm_backward(shape):
    inp = create_input(shape)

    notiredt_res = use_batch_norm_backward_kernel(inp)
    print()

    print(f"{inp = }")

    print(f"{notiredt_res = }")


@pytest.mark.parametrize("shape", default_shapes(min_dim=2, max_dim=4))
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("momentum", [0.1, 0.2])
@pytest.mark.parametrize("track_running_stats", [True, False])
@pytest.mark.parametrize("add_pre_act", [True, False])
@pytest.mark.parametrize("act_func", default_act_func())
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("amp", [True, False])
def test_batch_norm_layer(
    amp,
    input_dtype,
    act_func,
    add_pre_act,
    track_running_stats,
    momentum,
    eps,
    affine,
    shape,
) -> None:
    # why uses batchnorm if the batch size is 0
    if shape[0] == 1 or input_dtype is torch.float16 and not amp:
        return

    bn_name = "BatchNorm2d" if len(shape) == 4 else "BatchNorm1d"
    notiredt_inp = create_input(shape, dtype=input_dtype)
    torch_inp = create_input(shape, dtype=input_dtype)

    if add_pre_act:
        notiredt_residual = create_input(shape, dtype=input_dtype, seed=1)
        torch_residual = create_input(shape, dtype=input_dtype, seed=1)
    else:
        notiredt_residual = torch_residual = None

    notiredt_bn = getattr(notiredt, bn_name)(
        num_features=shape[1],
        eps=eps,
        affine=affine,
        track_running_stats=track_running_stats,
        momentum=momentum,
    )
    torch_bn = getattr(nn, bn_name)(
        num_features=shape[1],
        eps=eps,
        affine=affine,
        track_running_stats=track_running_stats,
        momentum=momentum,
        device="cuda",
    )

    pytorch_act = nn.Identity() if act_func is None else getattr(nn, act_func)()

    if affine:
        set_seed(0)
        init.normal_(notiredt_bn.weight)
        init.normal_(notiredt_bn.bias)

        set_seed(0)
        init.normal_(torch_bn.weight)
        init.normal_(torch_bn.bias)

    with autocast(enabled=amp):
        if add_pre_act:
            notiredt_out = notiredt_bn(notiredt_inp, notiredt_residual)
            torch_out = torch_bn(torch_inp + torch_residual)

        else:
            notiredt_out = notiredt_bn(notiredt_inp)
            torch_out = torch_bn(torch_inp)

    assert_close((notiredt_out, torch_out))
