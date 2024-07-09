import warnings
from random import randint

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

import triton
from notiredt.triton_kernel.dropout_kernel import (
    dropout_backward_kernel,
    dropout_forward_kernel,
)


class DropoutAutoGrad(torch.autograd.Function):
    @custom_fwd
    @staticmethod
    def forward(ctx, inp: Tensor, drop_p: float, training: bool) -> Tensor:
        # best scenario

        ctx.do_dropout = True
        if not training or drop_p == 0.0:
            ctx.do_dropout = False
            return inp

        # worst scenario
        ctx.drop_all = False
        if drop_p == 1.0:
            ctx.drop_all = True
            return torch.zeros_like(inp)

        flattened_input = inp.flatten()
        size = flattened_input.numel()
        out = torch.empty_like(flattened_input)

        seed = randint(0, 2**16 - 1) if training else 0
        ctx.seed = seed
        ctx.drop_p = drop_p

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        dropout_forward_kernel[grid](flattened_input, out, size, drop_p, seed)

        return out.view_as(inp)

    @custom_bwd
    @staticmethod
    def backward(ctx, output_grad: Tensor) -> Tensor:
        if not ctx.do_dropout:
            return output_grad, None, None

        if ctx.drop_all:
            return torch.zeros_like(output_grad), None, None

        orig_shape = output_grad.shape
        output_grad = output_grad.flatten()
        size = output_grad.numel()
        input_grad = torch.empty_like(output_grad)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

        dropout_backward_kernel[grid](
            output_grad, input_grad, size, ctx.drop_p, ctx.seed
        )

        # Pad is necessary for all input arguments
        return input_grad.view(*orig_shape), None, None


class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = p

        if inplace:
            warnings.warn("Inplace is not supported! ", "Failing back to out-of-place")

    def forward(self, inp: Tensor) -> Tensor:
        return DropoutAutoGrad.apply(inp, self.p, self.training)
