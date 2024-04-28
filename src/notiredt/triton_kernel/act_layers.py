import torch
import triton
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd

from notiredt.aliases import Context, SequenceOrTensor
from notiredt.triton_kernel.act_kernels import act_func_forward_kernel


class ActFuncAutoGrad(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx: Context, input_tensor: SequenceOrTensor, act_func: str
    ) -> SequenceOrTensor:
        """Applies and activation function to the input

        Args:
            ctx (Context): _description_
            input_tensor (SequenceOrTensor): _description_
            act_func (str): _description_

        Returns:
            SequenceOrTensor: _description_
        """
        ctx.act_func = act_func
        if input_tensor.requires_grad:
            ctx.save_for_backward(input_tensor)
        flattened_input_tensor = input_tensor.flatten()
        size = len(flattened_input_tensor)
        output = torch.empty_like(flattened_input_tensor)

        # Launches 1D grid where each program operates over
        # BLOCK_SIZE elements
        grid = lambda META: (triton.cdiv(size, META["BLOCK_SIZE"]),)
        act_func_forward_kernel[grid](
            flattened_input_tensor, output, size, True, 3748, act_func, True
        )

        return output.view_as(input_tensor)


class Sigmoid(nn.Sigmoid):
    """Applies sigmoid to the input"""

    def forward(self, input_tensor: SequenceOrTensor) -> SequenceOrTensor:
        return ActFuncAutoGrad.apply(input_tensor, "sigmoid")
