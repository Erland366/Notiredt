import math
from typing import TypeVar

import torch

from notiredt.aliases import SequenceOrTensor

T = TypeVar("T", bound=torch.Tensor)


def naive_attn(q: T, k: T, v: T, scale: float) -> torch.Tensor:
    s = q @ k.mT * scale
    # Equivalent with
    a = torch.softmax(s, dim=-1)
    # s = q @ k.transpose(-2, -1) * scale
    return a @ v


def scaled_dot_product_attention(
    query: T,
    key: T,
    value: T,
    attn_mask: T | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = False,
) -> T:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None, "attn_mask should be None for causal attention"
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        breakpoint()


L = 3
S = 5
query = torch.rand(2, 3, L, 4).to("cuda")
temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
print(temp_mask)
