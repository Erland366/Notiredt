import lovely_tensors as lt
import torch
import triton.language as tl

lt.monkey_patch()

DEBUG = True


@triton.jit(interpret=DEBUG)
def _gptq_kernel():
    pass
