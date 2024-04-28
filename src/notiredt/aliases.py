from typing import Any, Sequence, Union

import torch

__all__ = ["SequenceOrTensor"]

SequenceOrTensor = Union[Sequence[torch.Tensor], torch.Tensor]
Device = torch.device | str | None
Context = Any
