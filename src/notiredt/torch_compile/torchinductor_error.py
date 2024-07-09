import torch
import torch._dynamo as dynamo

torch._dynamo.config.repro_after = "aot"


def main():
    model = torch.nn.Sequential(*[torch.nn.Linear(200, 200) for _ in range(5)])

    def test_backend_error():
        y = torch.ones(200, 200)
        x = torch.ones(200, 200)
        z = x + y
        a = torch.ops.aten._foobar(z)  # dummy function which errors
        return model(a)

    compiled_test_backend_error = torch.compile(test_backend_error, backend="aot_eager")
    result = compiled_test_backend_error()
    print(result)


if __name__ == "__main__":
    main()
