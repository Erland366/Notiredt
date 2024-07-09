import lovely_tensors as lt
import pytest
import torch

lt.monkey_patch()


class TestSWAModule(torch.nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, embed_dim)

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(0, 1)

        return self.linear(hidden_states)


@pytest.mark.experiment
def test_swa():
    hidden_size = 512
    seq_len, batch_size, embed_dim = 1024, 2, 512

    hidden_states = torch.ones((batch_size, seq_len, embed_dim))
    model = TestSWAModule(hidden_size=hidden_size, embed_dim=embed_dim)

    res = model(hidden_states)
    print()

    print(f"{res.p = }")
    print(f"{res = }")


@pytest.mark.experiment
def test_einsum():
    x = torch.randn((1, 8, 768))
    y = torch.randn((1, 8, 768))

    def _chunk(hidden_states, window_overlap: int) -> torch.Tensor:
        chunk_size = (
            hidden_states.size(0),
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        )
        overlapping_chunks = torch.empty(
            chunk_size, device=hidden_states.device, dtype=hidden_states.dtype
        )
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :,
                chunk * window_overlap : chunk * window_overlap + 2 * window_overlap,
                :,
            ]
        return overlapping_chunks

    x_chunks = _chunk(x, 3)
    y_chunks = _chunk(y, 3)

    print(f"{x_chunks.p = }")
    print(f"{y_chunks.p = }")
