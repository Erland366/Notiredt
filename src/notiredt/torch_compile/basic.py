import warnings
from typing import Callable

import numpy as np
import torch
import torch._dynamo
from torch import Tensor
from torchvision.models import densenet121

torch.backends.cuda.matmul.allow_tf32 = False

N_ITERS = 16

gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 8), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn("GPU is not NVIDIA V100, A100, or H100")


@torch.compile
def foo(x: Tensor, y: Tensor) -> Tensor:
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


def timed(fn: Callable):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def generate_data(b: int):
    return (
        torch.randn((b, 3, 128, 128), device="cuda", dtype=torch.float32),
        torch.randint(1000, (b,), device="cuda"),
    )


def init_model():
    return densenet121().to(torch.float32).cuda()


def all_settings_printing():
    print(f"{torch._inductor.list_mode_options()}")
    print(f"{torch._dynamo.list_backends(None)}")


def old_main() -> None:
    model = init_model()
    torch._dynamo.reset()

    model_opt = torch.compile(model, mode="reduce-overhead")

    inp = generate_data(16)[0]
    with torch.no_grad():
        print("eager: ", timed(lambda: model(inp))[1])
        print("compiled: ", timed(lambda: model_opt(inp))[1])

    print(all_settings_printing)


def old_main_2() -> None:
    model = init_model()
    torch._dynamo.reset()

    model_opt = torch.compile(model, mode="reduce-overhead")

    eager_times: list[int] = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, eager_time = timed(lambda: model(inp))

        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")

    print("~" * 10)

    compiled_times: list[int] = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        with torch.no_grad():
            _, compiled_time = timed(lambda: model_opt(inp))

        compiled_times.append(compiled_time)
        print(f"compiled eval time {i}: {compiled_time}")

    print("~" * 10)

    eager_med = np.median(eager_times)
    compiled_med = np.median(compiled_times)
    print(f"{eager_med = }")
    print(f"{compiled_med = }")
    speedup = eager_med / compiled_med
    print(
        f"(eval) eager median: {eager_med}, compiled median: {compiled_med}, speedup: {speedup}"
    )
    print("~" * 10)


def main() -> None:
    model = init_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch._dynamo.reset()
    torch._dynamo.config.repro_after = "aot"

    def train(model: Callable, data: Tensor) -> None:
        opt.zero_grad(True)
        res = model(data[0])
        loss = torch.nn.CrossEntropyLoss()(res, data[1])
        loss.backward()
        opt.step()

    eager_times: list[int] = []
    for i in range(N_ITERS):
        inp = generate_data(16)
        _, eager_time = timed(lambda: train(model, inp))

        eager_times.append(eager_time)
        print(f"eager train time {i}: {eager_time}")

    print("~" * 10)

    model = init_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_opt = torch.compile(train, mode="max-autotune")

    compiled_times: list[int] = []
    for i in range(N_ITERS):
        inp = generate_data(16)
        _, compiled_time = timed(lambda: train_opt(model, inp))

        compiled_times.append(compiled_time)
        print(f"compiled train time {i}: {compiled_time}")

    print("~" * 10)

    eager_med = np.median(eager_times)
    compiled_med = np.median(compiled_times)
    print(f"{eager_med = }")
    print(f"{compiled_med = }")
    speedup = eager_med / compiled_med
    print(
        f"(eval) eager median: {eager_med}, compiled median: {compiled_med}, speedup: {speedup}"
    )
    print("~" * 10)


if __name__ == "__main__":
    main()
