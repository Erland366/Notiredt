import torch
import triton.language as tl
import lovely_tensors as lt

lt.monkey_patch()

DEBUG = True


@triton.jit(interpret=DEBUG)
def _gptq_kernel():
    pass
