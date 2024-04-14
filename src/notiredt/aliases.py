from typing import Sequence, Union

import torch

__all__ = ["SequenceOrTensor"]

SequenceOrTensor = Union[Sequence[torch.Tensor], torch.Tensor]
