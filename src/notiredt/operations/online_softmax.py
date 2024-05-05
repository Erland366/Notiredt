import lovely_tensors as lt
import torch

lt.monkey_patch()


def torch_softmax(x: torch.FloatTensor) -> torch.FloatTensor:
    return torch.softmax(x, dim=-1)


def online_softmax(x: torch.FloatTensor) -> torch.FloatTensor:
    # Simulates parallelization where each thread working on their current index only
    d_old = 0
    m_old = torch.tensor(float("-inf"))
    for i in range(x.numel()):
        m_new = torch.max(m_old, x[i])
        exp_new = m_old - m_new
        d_new = d_old * torch.exp(exp_new) + torch.exp(x[i] - m_new)
        d_old = d_new
        m_old = m_new

    a = torch.empty_like(x)
    for i in range(x.numel()):
        a[i] = torch.exp(x[i] - m_new) / d_new

    return a


a = torch.randn(100, device="cuda")
print(torch_softmax(a))
print(online_softmax(a))
print(torch.allclose(torch_softmax(a), online_softmax(a)))
