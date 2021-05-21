"""
Wrap commonly-used torch builtins in nn.Module subclass
for easier automatic construction of script
"""
import torch.nn

class argmax(torch.nn.Module):
        def __init__(self):
                super(argmax, self).__init__()
        def forward(self, A):
                return torch.argmax(A)

class matmul(torch.nn.Module):
        def __init__(self):
                super(matmul, self).__init__()
        def forward(self, A, B):
                return torch.matmul(A, B.T)

class add(torch.nn.Module):
        def __init__(self):
                super(add, self).__init__()
        def forward(self, A, B):
                return torch.add(A, B)

# TODO: Many more to be implemented


__all__ = ["argmax", "matmul", "add"]
