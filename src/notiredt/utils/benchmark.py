import time

import lovely_tensors as lt
import torch
from rich import box
from rich.console import Console
from rich.table import Table

lt.monkey_patch()


class RichPrint:
    def __init__(self):
        self.console = Console()

    def print(self, text, style=""):
        self.console.print(text, style=style)

    def print_table(self, column_names, rows, title=None):
        table = Table(title=title, box=box.SIMPLE)
        for name in column_names:
            table.add_column(name, justify="center")
        for row in rows:
            table.add_row(*[str(item) for item in row])
        self.console.print(table)


def equal(inp: torch.Tensor, other: torch.Tensor, eps: float = 1e-2):
    return torch.allclose(input=inp, other=other, rtol=0, atol=eps)


class TimeIt:
    def __init__(self, name: str | None = None, style="bold yellow"):
        self.name = name
        self.rich = RichPrint()
        self.style = style

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, *args):
        elapsed_time = time.perf_counter() - self.start_time
        self.rich.print(
            f"Amount of time to run {self.name} is {elapsed_time:.8} s",
            style=self.style,
        )
