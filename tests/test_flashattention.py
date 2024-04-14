import pytest
import torch

MAX_HEADDIM_SM8x = 192

is_sm75 = torch.cuda.is_available() and torch.cuda.get_device_capability("cuda") == (
    7,
    5,
)
is_sm8x = torch.cuda.is_available() and torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability("cuda") == (
    8,
    0,
)
is_sm90 = torch.cuda.is_available() and torch.cuda.get_device_capability("cuda") == (
    9,
    0,
)

print(f"{is_sm8x = }")


@pytest.mark.parametrize(
    argnames="dtype",
    argvalues=[torch.float16] if is_sm75 else [torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(argnames="local", argvalues=[False, True])
@pytest.mark.parametrize(
    argnames="d", argvalues=[32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256]
)
@pytest.mark.parametrize(argnames="swap_sq_sk", argvalues=[False, True])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
def test_flash_attn(seqlen_q, seqlen_k, swap_sq_sk, d, local, dtype):
    pass
