# From : https://github.com/cuda-mode/lectures/tree/main/lecture%2014
import os

import torch


def check_tensors_gpu_ready(*tensors: torch.Tensor):
    for t in tensors:
        assert t.is_contiguous(), "A tensor is not contiguous"
        if not os.environ.get("TRITON_INTERPRET") == "1":
            assert t.is_cuda, "A tensor is not on GPU"


def test_pid_conds(conds_str: str, pid_0=[0], pid_1=[0], pid_2=[0]) -> bool:
    pids = pid_0[0], pid_1[0], pid_2[0]
    conds = conds_str.replace(" ", "").split(",")
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond == "":
            continue
        try:
            op, threshold = cond[0], int(cond[1:])
        except ValueError as e:
            if len(cond[1:]) == 2:
                op, threshold = cond[0:2], int(cond[2:])
            else:
                raise ValueError(e)
        if op not in ["<", ">", ">=", "<=", "=", "!="]:
            raise ValueError(
                f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{cond}'."
            )
        op = "==" if op == "=" else op
        if not eval(f"{pid} {op} {threshold}"):
            return False
    return True
