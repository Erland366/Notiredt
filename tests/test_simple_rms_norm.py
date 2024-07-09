import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import pytest
import torch
from notiredt.triton_kernel.batchnorm_kernel import FastSimpleRMSNorm, SimpleRMSNorm


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "dim"),
    ((1, 100, 1000), (8, 100, 1000)),
)
class TestSimpleRMSNorm:
    @pytest.fixture(autouse=True)
    def setup(self, dim):
        self.cpu_norm = SimpleRMSNorm(dim=dim)
        self.gpu_norm = FastSimpleRMSNorm(dim=dim)

    @staticmethod
    def example_input_cpu_cuda(batch_size, seq_len, dim):
        x_cuda = torch.randn(
            batch_size, seq_len, dim, device="cuda", requires_grad=True
        )
        x_cpu = x_cuda.cpu()
        return x_cuda, x_cpu

    def test_simple_rms_norm(self, batch_size, seq_len, dim):
        x_cuda, x_cpu = self.example_input_cpu_cuda(batch_size, seq_len, dim)
        print(x_cuda)
        print(self.cpu_norm)
