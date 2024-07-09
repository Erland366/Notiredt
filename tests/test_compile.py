import torch
import torch.nn as nn
from torch._inductor.utils import print_performance


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.batch_norm(x)
        return x


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


def main():
    mod = MyModule().to("cuda")
    opt_mod = torch.compile(
        mod,
    )
    print(opt_mod(torch.randn(10, 100, device="cuda")))


def compile_batchnorm(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4, "input must be 4d"
    model = TestModule().to("cuda")
    compiled_model = torch.compile(model)
    result = compiled_model(x)
    result.backward(torch.rand_like(x, device="cuda"))
    return result


if __name__ == "__main__":
    shape = (1, 3, 64, 64)
    compile_batchnorm(torch.randn(shape, device="cuda"))
    fn = lambda: compile_batchnorm(torch.randn(shape, device="cuda"))
    print_performance(fn, times=10, repeat=5)
    # main()
