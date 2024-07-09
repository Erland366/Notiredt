import torch
import torch._dynamo as dynamo


def test_assertion_error():
    y = torch.ones((200, 200))
    z = {y: 5}
    return z


compiled_y = torch.compile(test_assertion_error)
compiled_y()
