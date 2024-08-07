"""
Utilities for Testing
"""

import torch
from notiredt.aliases import Device, SequenceOrTensor


def default_shapes(min_dim: int = 0, max_dim: int = 4) -> list[tuple[int, ...]]:
    """Return typical data shapes for testing

    Parameters
    ----------
    min_dim : int, optional
        Minimum dimensionality of the shapes returned, by default 0
    max_dim : int, optional
        Maximum dimensionality of the shapes returned, by default 4

    Returns
    -------
    list[tuple[int, ...]]
        _description_
    """
    shapes = [
        (96,),
        (128,),
        (196,),
        (384,),
        (768,),
        (1024,),
        (3200,),
        (4800,),
        (8000,),
        (12288,),
        (1, 8000),
        (4, 2000),
        (8, 1024),
        (32, 1024),
        (128, 1024),
        (2048, 768),
        (6144, 256),
        (8096, 32),
        (12288, 1),
        (1, 1024, 3072),
        (8, 960, 196),
        (64, 768, 128),
        (128, 960, 196),
        (2048, 64, 16),
        (1, 3, 224, 224),
        (8, 3, 224, 224),
        (64, 64, 56, 56),
        (256, 128, 28, 28),
        (256, 2048, 7, 7),
    ]
    return list(filter(lambda shape: min_dim <= len(shape) <= max_dim, shapes))


def create_input(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Device = "cuda",
    requires_grad: bool = True,
    seed: int | None = 3407,
):
    if seed is not None:
        torch.manual_seed(seed)
    example_input = torch.randn(
        shape, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return example_input


def create_input_like(
    input_tensor: SequenceOrTensor,
    requiers_grad: bool = False,
    seed: int | None = 3407,
):
    if seed is not None:
        torch.manual_seed(seed)
    assert isinstance(
        input_tensor, torch.Tensor
    ), "Tensor must be a type of torch.Tensor"

    return torch.randn_like(input_tensor, requires_grad=requiers_grad)


def create_zeros_like(
    input_tensor: SequenceOrTensor,
    requiers_grad: bool = False,
    seed: int | None = 3407,
):
    if seed is not None:
        torch.manual_seed(seed)
    assert isinstance(
        input_tensor, torch.Tensor
    ), "Tensor must be a type of torch.Tensor"

    return torch.zeros_like(input_tensor, requires_grad=requiers_grad)


def assert_close(
    *tensor_pairs: tuple[SequenceOrTensor, SequenceOrTensor],
    rtol: float | None = None,
    atol: float | None = None,
):
    for pair in tensor_pairs:
        torch.testing.assert_close(*pair, rtol=rtol, atol=atol)


def default_act_func(exclude: list[str] | None = None) -> list[str | None]:
    """
    Return the default activation function for layers.

    Returns:
        list[str | None]: A list containing the default activation function.
            If the first element is None, the layer does not apply activation function.
            Otherwise, the first element is the name of the activation function.
    """
    # The default activation function is None, meaning the layer does not apply activation function.
    # If the first element is not None, the first element is the name of the activation function.
    # Currently, the only supported activation function is "sigmoid".
    all_act_func = [None, "sigmoid"]
    if exclude is None:
        exclude = []
    return [act_func for act_func in all_act_func if act_func not in exclude]


__all__ = [
    "assert_close",
    "create_input",
    "create_input_like",
    "default_shapes",
]
