from src.notiredt.triton_kernel.utils import allow_tf32


def test_allow_tf32():
    result = allow_tf32()
    assert result is True
