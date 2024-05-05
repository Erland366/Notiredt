import gc
import time

import torch
from torch import nn
from tqdm import tqdm

start_time = None


def start_timer() -> None:
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()


def end_timer_and_print(local_msg: str) -> None:
    torch.cuda.synchronize()
    assert start_time
    end_time = time.perf_counter()
    print("\n" + local_msg)
    print("Total execution time : {:.3f} sec".format(end_time - start_time))
    print(
        "Max memory used by tensors = {} bytes".format(
            torch.cuda.max_memory_allocated()
        )
    )


def make_model(in_size: int, out_size: int, num_layers: int) -> nn.Module:
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()


def main() -> None:
    batch_size = 512
    in_size = 4096
    out_size = 4096
    num_layers = 3
    num_batches = 50
    epochs = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.set_default_device(device)

    # create data in default preciso
    # The same data is used for both default and mixed precision trials below
    # You don't need to manually change inputs `dtype` when enabling mixed precision
    data = [torch.randn(batch_size, in_size) for _ in range(num_batches)]
    targets = [torch.randn(batch_size, out_size) for _ in range(num_batches)]

    loss_fn = torch.nn.MSELoss().cuda()

    # Without torch.cuda.amp
    net = make_model(in_size, out_size, num_layers)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)

    print("Start timer for non precision")
    start_timer()
    for epoch in tqdm(range(epochs)):
        for inp, target in zip(data, targets):
            outputtt = net(inp)
            loss = loss_fn(outputtt, target)
            loss.backward()
            opt.step()
            opt.zero_grad()  # set_to_none=True here can modestly improve performance
    end_timer_and_print("Default precision: ")

    print("Start timer for precision")
    start_timer()
    for epoch in tqdm(range(epochs)):
        for inp, target in zip(data, targets):
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputtt = net(inp)

                assert outputtt.dtype is torch.float16
                loss = loss_fn(outputtt, target)

                assert loss.dtype is torch.float32
            # Exits autocast before backward()
            # Backward passes under autocast are not recommended
            # Backward ops run in the same dtype autocast chose for corresponding forward ops
            loss.backward()
            opt.step()
            opt.zero_grad()  # set_to_none=True here can modestly improve performance
    end_timer_and_print("Mixed precision: ")

    print("Start timer for precision with scaler")
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    start_timer()
    for epoch in tqdm(range(epochs)):
        for inp, target in zip(data, targets):
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputtt = net(inp)

                assert outputtt.dtype is torch.float16
                loss = loss_fn(outputtt, target)

                assert loss.dtype is torch.float32
            # Exits autocast before backward()
            # Backward passes under autocast are not recommended
            # Backward ops run in the same dtype autocast chose for corresponding forward ops
            scaled_loss = scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()  # set_to_none=True here can modestly improve performance
    end_timer_and_print("Mixed precision: ")

    # All together
    use_amp = True

    net = make_model(in_size, out_size, num_layers)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=use_amp)

    start_timer()
    for epoch in tqdm(range(epochs)):
        for inp, target in zip(data, targets):
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=use_amp
            ):
                output = net(inp)
                loss = loss_fn(output, target)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
    end_timer_and_print("Mixed precision :")


def test_things():
    start_timer()
    models = make_model(10, 20, 3)
    print(models)
    end_timer_and_print("Aku mau makan")


if __name__ == "__main__":
    main()
