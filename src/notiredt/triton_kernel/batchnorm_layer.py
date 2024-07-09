import torch
import triton
from torch import Tensor, nn

from notiredt.triton_kernel.act_kernels import act_func_backward_kernel
from notiredt.triton_kernel.batchnorm_kernel import (
    batch_norm_backward_kernel,
    batch_norm_forward_kernel,
)


def make_3d_for_bn(input_tensor: Tensor) -> Tensor:
    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(-1)
    elif input_tensor.ndim == 4:
        input_tensor = input_tensor.flatten(start_dim=2, end_dim=-1)
    return input_tensor


class BatchNormAutoGrad(torch.autograd.Function):
    # I think they didn't use custom_fwd (and custom_bwd) later
    # is because batch norm needs to be always in fp32
    @staticmethod
    def forward(
        ctx,
        input_tensor: Tensor,
        training: bool,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        running_mean: Tensor | None = None,
        running_var: Tensor | None = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
        track_running_stats: bool = True,
        pre_act_add: Tensor | None = None,
        act_func: str | None = None,
    ):
        add_pre_act = pre_act_add is not None
        pre_act_add = (
            pre_act_add if add_pre_act else torch.empty((1, 1, 1), device="cuda")
        )
        input_3d = make_3d_for_bn(input_tensor)
        pre_act_add = make_3d_for_bn(pre_act_add)
        transpose = False

        if input_3d.shape[-1] > 1:
            input_3d = input_3d.transpose(0, -1)
            pre_act_add = pre_act_add.transpose(0, -1)
            transpose = True

        affine = weight is not None and bias is not None
        requires_grad = (
            input_tensor.requires_grad
            or (affine and weight.requires_grad)
            or (affine and bias.requires_grad)
        )

        save_pre_act = requires_grad and (act_func is not None)

        batch_dim, feat_dim, spatial_dim = input_3d.shape
        output = torch.empty_like(input_3d)
        pre_act = torch.empty_like(input_3d) if save_pre_act else output

        if requires_grad:
            mean = torch.empty(
                feat_dim, dtype=torch.float32, device=input_tensor.device
            )
            inv_std = torch.empty(
                feat_dim, dtype=torch.float32, device=input_tensor.device
            )
        else:
            mean = inv_std = None

        running_mean = input_tensor if running_mean is None else running_mean
        running_var = input_tensor if running_var is None else running_var

        # launch 1D grid where each program operates over one feature
        grid = lambda _: (feat_dim,)
        # fmt: off
        batch_norm_forward_kernel[grid](
            input_3d, weight, bias,
            mean, inv_std,
            pre_act_add,
            pre_act,
            output,
            running_mean, running_var,
            batch_dim, spatial_dim,
            *input_3d.stride(),
            *pre_act_add.stride(),
            *pre_act.stride(),
            *output.stride(),
            momentum,
            eps,
            affine=affine,
            save_stats=requires_grad,
            track_running_stats=track_running_stats,
            is_train=training,
            add_pre_act=add_pre_act,
            act_func=act_func,
            save_pre_act=save_pre_act,
        )
        if transpose:
            output = output.transpose(0, -1)
            if save_pre_act:
                pre_act = pre_act.transpose(0, -1)

        ctx.affine = affine
        ctx.act_func = act_func
        ctx.add_pre_act = add_pre_act
        if requires_grad:
            ctx.save_for_backward(
                input_tensor, mean, inv_std, weight, pre_act if save_pre_act else None
            )
        return output.view_as(input_tensor)

    @staticmethod
    def backward(ctx, output_grad):
        # unpack save_for_backward
        inp, mean, inv_std, weight, pre_act = ctx.saved_tensors

        # what makes this interesting is that
        # for dropout, we always operates in 1D
        # whereas for this activation function, we always operates in
        # 3D
        inp_3d = make_3d_for_bn(inp)

        if ctx.act_func is None:
            pre_act_grad = make_3d_for_bn(output_grad)

        else:
            size = output_grad.numel()
            pre_act_grad = torch.empty(
                size, device=output_grad.device, dtype=output_grad.dtype
            )

            grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
            act_func_backward_kernel[grid](
                output_grad.flatten(),
                inp,
                pre_act_grad,
                size,
                0.0,
                0.0,
                ctx.act_func,
                False,
            )

            pre_act_grad = pre_act_grad.view_as(pre_act)

        transpose = False
        if inp_3d.shape[-1] > 1:
            inp_3d = inp_3d.transpose(0, -1)
            pre_act_grad = pre_act_grad.transpose(0, -1)
            transpose = True

        b_dim, f_dim, s_dim = inp_3d.shape
        inp_grad = torch.empty_like(inp_3d)

        if ctx.affine:
            weight_grad = torch.empty((f_dim,), device=inp.device)
            bias_grad = torch.empty_like(weight_grad)
        else:
            weight_grad = bias_grad = None

        # Do things for each f_dim
        grid = lambda _: (f_dim,)

        batch_norm_backward_kernel[grid](
            pre_act_grad,
            inp_3d,
            mean,
            inv_std,
            weight,
            inp_grad,
            weight_grad,
            bias_grad,
            b_dim,
            s_dim,
            *pre_act_grad.stride(),
            *inp_3d.stride(),
            *inp_grad.stride(),
            affine=ctx.affine,
        )

        if transpose:
            inp_3d = inp_3d.transpose(0, -1)
            pre_act_grad = pre_act_grad.transpose(0, -1)

        # pads output with None, we need to make input and output stays the same
        # with input gradients
        return (
            inp_grad.view_as(inp),
            None,
            weight_grad,
            bias_grad,
            None,
            None,
            None,
            None,
            None,
            pre_act_grad.view_as(pre_act) if ctx.add_pre_act else None,
            None,
        )


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_func: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self.act_func = act_func

    def forward(self, inp: Tensor, pre_act_add: Tensor | None = None) -> Tensor:
        # Check if input is 3D
        self._check_input_dim(inp)

        return BatchNormAutoGrad.apply(
            inp,
            self.training,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.momentum,
            self.eps,
            self.track_running_stats,
            pre_act_add,
            self.act_func,
        )


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_func: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )
        self.act_func = act_func

    def forward(self, inp: Tensor, pre_act_add: Tensor | None = None) -> Tensor:
        # check if input is 4D
        self._check_input_dim(inp)

        return BatchNormAutoGrad.apply(
            inp,
            self.training,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.momentum,
            self.eps,
            self.track_running_stats,
            pre_act_add,
            self.act_func,
        )
