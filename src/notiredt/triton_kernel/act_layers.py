from random import randint

import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd

import triton
from notiredt.aliases import Context, SequenceOrTensor
from notiredt.triton_kernel.act_kernels import act_func_forward_kernel


class ActFuncAutoGrad(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx: Context,
        input_tensor: SequenceOrTensor,
        act_func: str,
        drop_p: float,
        training: bool,
    ) -> SequenceOrTensor:
        ctx.act_func = act_func
        ctx.drop_p = drop_p
        ctx.dropout = drop_p > 0 and training
        seed = randint(0, 2**16 - 1) if training else 0
        ctx.seed = seed

        if input_tensor.requires_grad:
            ctx.save_for_backward(
                input_tensor,
            )

        flattened_input = input_tensor.flatten()
        size = input_tensor.numel()
        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        output = torch.empty_like(flattened_input)
        act_func_forward_kernel[grid](
            flattened_input, output, size, drop_p, seed, act_func, ctx.dropout
        )

        return output.view_as(input_tensor)

    @staticmethod
    @custom_bwd
    def backward(ctx: Context, output_grad: SequenceOrTensor) -> SequenceOrTensor:
        pass


class Sigmoid(nn.Sigmoid):
    """Applies sigmoid to the input"""

    def __init__(self, drop_p: float = 0.0) -> None:
        super().__init__()
        self.drop_p = drop_p

    def forward(self, input_tensor: SequenceOrTensor) -> SequenceOrTensor:
        return ActFuncAutoGrad.apply(
            input_tensor, "sigmoid", self.drop_p, self.training
        )
