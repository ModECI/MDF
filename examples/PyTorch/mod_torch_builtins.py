"""
Wrap commonly-used torch builtins in nn.Module subclass
for easier automatic construction of script
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class argmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.argmax(A)


class argmin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.argmin(A)


class matmul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.matmul(A, B.T)


class add(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.add(A, B)


class sin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.sin(A)


class cos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.cos(A)


class abs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.abs(A)


class flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.reshape(A, (1, -1))


class clip(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, min_val, max_val):
        return torch.clamp(A, min_val, max_val)


class shape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.tensor(A.size()).to(torch.int64)


class det(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return torch.det(A)


class And(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.logical_and(A > 0, B > 0)


class Or(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.logical_or(A > 0, B > 0)


class Xor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.logical_xor(A > 0, B > 0)


class concat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, axis=0):

        return torch.cat(A, axis)


class ceil(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):

        return torch.ceil(A)


class floor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):

        return torch.floor(A)


class bitshift(torch.nn.Module):
    def __init__(self, DIR):
        super().__init__()
        self.dir = DIR

    def forward(self, A, B):
        if self.dir == "RIGHT":
            return A.to(torch.int64) >> B.to(torch.int64)
        else:
            return A.to(torch.int64) << B.to(torch.int64)


class conv(torch.nn.Module):
    def __init__(
        self,
        auto_pad="NOTSET",
        kernel_shape=None,
        group=1,
        strides=[1, 1],
        dilations=[1, 1],
        pads=[0, 0, 0, 0],
    ):
        super().__init__()
        self.group = group
        self.auto_pad = auto_pad
        self.strides = tuple(strides)
        self.dilations = tuple(dilations)
        self.kernel_shape = kernel_shape

    def forward(self, A, W, B=None):
        if self.auto_pad == "NOTSET":
            self.pads = tuple(pads)

        elif self.auto_pad == "VALID":
            self.pads = (0, 0, 0, 0)
        elif self.auto_pad == "SAME_UPPER":
            pad_dim1 = (
                torch.ceil(torch.tensor(A.shape[2]).to(torch.float32) / strides[0])
                .to(torch.int64)
                .item()
            )
            pad_dim2 = (
                torch.ceil(torch.tensor(A.shape[3]).to(torch.float32) / strides[1])
                .to(torch.int64)
                .item()
            )
            if pad_dim1 % 2 == 0 and pad_dim2 % 2 == 0:
                self.pads = (pad_dim1 // 2, pad_dim1 // 2, pad_dim2 // 2, pad_dim2 // 2)
            elif pad_dim1 % 2 == 0 and pad_dim2 % 2 != 0:
                self.pads = (
                    pad_dim1 // 2,
                    pad_dim1 // 2,
                    pad_dim2 // 2,
                    pad_dim2 // 2 + 1,
                )
            elif pad_dim1 % 2 != 0 and pad_dim2 % 2 == 0:
                self.pads = (
                    pad_dim1 // 2,
                    pad_dim1 // 2 + 1,
                    pad_dim2 // 2,
                    pad_dim2 // 2,
                )
            elif pad_dim1 % 2 != 0 and pad_dim2 % 2 != 0:
                self.pads = (
                    pad_dim1 // 2,
                    pad_dim1 // 2 + 1,
                    pad_dim2 // 2,
                    pad_dim2 // 2 + 1,
                )

        elif self.auto_pad == "SAME_LOWER":

            pad_dim1 = (
                torch.ceil(torch.tensor(A.shape[2]).to(torch.float32) / strides[0])
                .to(torch.int64)
                .item()
            )
            pad_dim2 = (
                torch.ceil(torch.tensor(A.shape[3]).to(torch.float32) / strides[1])
                .to(torch.int64)
                .item()
            )
            if pad_dim1 % 2 == 0 and pad_dim2 % 2 == 0:
                self.pads = (pad_dim1 // 2, pad_dim1 // 2, pad_dim2 // 2, pad_dim2 // 2)
            elif pad_dim1 % 2 == 0 and pad_dim2 % 2 != 0:
                self.pads = (
                    pad_dim1 // 2,
                    pad_dim1 // 2,
                    pad_dim2 // 2 + 1,
                    pad_dim2 // 2,
                )
            elif pad_dim1 % 2 != 0 and pad_dim2 % 2 == 0:
                self.pads = (
                    pad_dim1 // 2 + 1,
                    pad_dim1 // 2,
                    pad_dim2 // 2,
                    pad_dim2 / 2,
                )
            elif pad_dim1 % 2 != 0 and pad_dim2 % 2 != 0:
                self.pads = (
                    pad_dim1 // 2 + 1,
                    pad_dim1 // 2,
                    pad_dim2 // 2 + 1,
                    pad_dim2 // 2,
                )

        A = F.pad(A, self.pads)
        return F.conv2d(
            A,
            W,
            bias=B,
            stride=self.strides,
            padding=self.pads,
            dilation=self.dilations,
            groups=self.group,
        )


class elu(torch.nn.Module):
    def __init__(self, alpha=1.0):

        super().__init__()
        self.alpha = alpha

    def forward(self, A):

        return nn.ELU(alpha=self.alpha)(A.to(torch.float32))


class hardsigmoid(torch.nn.Module):
    def __init__(self, alpha=0.2, beta=0.5):

        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, A):

        return torch.clamp(self.alpha * (A.to(torch.float32)) + self.beta, 0, 1)


class hardswish(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.alpha = 1.0 / 6
        self.beta = 0.5

    def forward(self, A):

        return A * torch.clamp(self.alpha * (A.to(torch.float32)) + self.beta, 0, 1)


class hardmax(torch.nn.Module):
    def __init__(self, axis=-1):
        super().__init__()

        self.axis = axis

    def forward(self, A):
        A = A.to(torch.float32)
        rank = A.shape
        if self.axis < 0:
            self.axis += len(rank)
        tensor = torch.arange(rank[self.axis])
        repeats = []
        repeats.append(1)
        for i, idx in enumerate(reversed(rank[: self.axis])):

            repeats.append(1)
            tensor = torch.stack([tensor] * idx)

        for i, idx in enumerate(rank[self.axis + 1 :]):
            repeats.append(idx)

            tensor = tensor.unsqueeze(-1).repeat(repeats)
            repeats[-1] = 1
        # b = torch.stack([torch.stack([torch.arange(4)] * 3)] *2)
        # print(tensor.shape)
        max_values, _ = torch.max(A, dim=self.axis)
        # print(max_values, max_values.shape)
        # tensor = torch.reshape(tensor, tuple(rank))
        tensor[A != torch.unsqueeze(max_values, dim=self.axis)] = rank[self.axis]
        # print(b)
        first_max, _ = torch.min(tensor, dim=self.axis)

        one_hot = torch.nn.functional.one_hot(first_max, rank[self.axis])
        return one_hot


class compress(torch.nn.Module):
    def __init__(self, axis=None):
        self.axis = axis
        super().__init__()

    def forward(self, A, B):

        idx = (B.to(torch.bool) != 0).nonzero().reshape(-1)
        if self.axis != None:
            return torch.index_select(A, self.axis, idx)

        else:

            return torch.index_select(A.reshape(-1), 0, idx)


# TODO: Many more to be implemented


__all__ = [
    "argmax",
    "argmin",
    "matmul",
    "add",
    "sin",
    "cos",
    "abs",
    "flatten",
    "clip",
    "shape",
    "det",
    "And",
    "Or",
    "Xor",
    "concat",
    "ceil",
    "floor",
    "bitshift",
    "conv",
    "elu",
    "hardsigmoid",
    "hardswish",
    "compress",
]
