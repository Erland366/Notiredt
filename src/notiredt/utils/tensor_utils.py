import torch
from notiredt.aliases import SequenceOrTensor


def create_input(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
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

    return torch.randn_like(input_tensor, requires_grad=requiers_grad)
