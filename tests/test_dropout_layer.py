import pytest
from notiredt.triton_kernel.dropout_kernel import dropout_forward_kernel
from .utils import create_input_like, default_shapes, create_input, create_zeros_like
import lovely_tensors as lt

lt.monkey_patch()


@pytest.mark.parametrize("input_shape", default_shapes(min_dim=1, max_dim=2))
def test_dropout_forward_kernel(input_shape: tuple[int, ...]):
    import os

    os.environ["TRITON_INTERPRET"] = "1"
    example_input = create_input(input_shape)
    example_output = create_zeros_like(example_input)

    dropout_forward_kernel[(1,)](
        example_input, example_output, example_input.numel(), 1, 48
    )

    assert example_output == create_zeros_like(example_input)

    dropout_forward_kernel[(1,)](
        example_input, example_output, example_input.numel(), 0, 48
    )

    assert example_output == example_input

    dropout_forward_kernel[(1,)](
        example_input, example_output, example_input.numel(), 0.5, 48
    )

    assert example_output != example_input
    assert example_output != create_zeros_like(example_input)
