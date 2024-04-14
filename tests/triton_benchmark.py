import torch

from torch.utils.benchmark import Timer
from typing import Callable
from notiredt.triton_kernel.add import add
from notiredt.triton_kernel.add import add_torch


class TritonBenchmark:
    def __init__(self, triton_func: Callable, torch_func: Callable, *args, **kwargs):
        print(
            f"Make sure both the triton_func '{triton_func.__name__}' and the torch_func '{torch_func.__name__}' take the same arguments"
        )
        self.triton_func = triton_func
        self.torch_func = torch_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        print("Test if value is the same", flush=True, end=": ")
        torch.allclose(
            self.triton_func(*self.args, **self.kwargs),
            self.torch_func(*self.args, **self.kwargs),
        )
        print("Yes, the result is the same")

        print("Benchmarking Triton Function")
        triton_timer = Timer(
            stmt="self.triton_func(*self.args, **self.kwargs)",
            globals={"self": self},
            label="Triton Function",
            description="Triton Benchmark",
        )

        print(triton_timer.timeit(10))

        print("Benchmarking PyTorch Function")

        torch_timer = Timer(
            stmt="self.torch_func(*self.args, **self.kwargs)",
            globals={"self": self},
            label="Torch Function",
            description="Torch Benchmark",
        )

        print(torch_timer.timeit(10))


X = torch.randn(1000, device="cuda")
Y = torch.randn(1000, device="cuda")

benchmark = TritonBenchmark(add, add_torch, X, Y)
benchmark.run()
