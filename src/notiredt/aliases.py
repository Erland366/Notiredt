from typing import Any, Sequence

import torch

__all__ = ["SequenceOrTensor", "Device", "Context"]

SequenceOrTensor = Sequence | torch.Tensor
Device = torch.device | str | None
Context = Any
