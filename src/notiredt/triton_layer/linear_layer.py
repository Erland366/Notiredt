import os

import torch
from jaxtyping import Float

from notiredt.triton_kernel.linear_kernel import linear_forward_kernel
from notiredt.utils.tensor_utils import create_input


def linear_forward():
    b = 1024
    input_shape = 128
    output_shape = 256
    example_input: Float[torch.Tensor, "b input_shape"] = create_input(
        (b, input_shape), device="cuda"
    )
    example_weight: Float[torch.Tensor, "input_shape output_shape"] = create_input(
        (input_shape, output_shape), device="cuda"
    )
    example_bias: Float[torch.Tensor, "output_shape"] = create_input(
        (output_shape,), device="cuda"
    )
    pre_act: Float[torch.Tensor, "b output_shape"] = torch.zeros(
        (b, output_shape), device="cuda"
    )
    example_output: Float[torch.Tensor, " b output_shape"] = torch.zeros(
        (b, output_shape), device="cuda"
    )

    linear_forward_kernel[(1,)](
        example_input,
        example_weight,
        example_bias,
        pre_act,
        example_output,
        b,
        input_shape,
        output_shape,
        example_input.stride(0),
        example_input.stride(1),
        example_weight.stride(0),
        example_weight.stride(1),
        pre_act.stride(0),
        pre_act.stride(1),
        example_output.stride(0),
        example_output.stride(1),
        add_bias=True,
        act_func="relu",
        save_pre_act=False,
        fp16=True,
        tf32=False,
    )


def main():
    linear_forward()


if __name__ == "__main__":
    os.environ["TRITON_INTERPRET"] = "1"
    main()
