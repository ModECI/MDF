"""

A simple set of nodes connected A->B->C->D, built in pytorch & exported to onnx

Work in progress!!!

"""

import sys
import torch
from torch import nn
import numpy as np

import abcd_python as abcd

in_size = 1
out_size = 1

#### A
A = nn.Linear(in_size, out_size)

with torch.no_grad():
    A.weight[0][0] = abcd.A_slope
    A.bias[0] = abcd.A_intercept


#### B
class MyLogistic(nn.Module):

    def __init__(self, gain, bias, offset):
        super().__init__()
        self.gain = gain
        self.bias = bias
        self.offset = offset

    def forward(self, input: torch.Tensor):
        return 1 / (1 + torch.exp(-1 * self.gain * (input + self.bias) + self.offset))

B = MyLogistic(abcd.B_gain, abcd.B_bias, abcd.B_offset)

#### C
class MyExp(nn.Module):

    def __init__(self, scale, rate, bias, offset):
        super().__init__()
        self.scale = scale
        self.rate = rate
        self.bias = bias
        self.offset = offset

    def forward(self, input: torch.Tensor):
        return self.scale * torch.exp((self.rate * input) + self.bias) + self.offset

C = MyExp(abcd.C_scale, abcd.C_rate, abcd.C_bias, abcd.C_offset)

#### D
class MySin(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor):
        return self.scale * torch.sin(input)

D = MySin(abcd.D_scale)

m_a = nn.Sequential(A)
m_ab = nn.Sequential(A,B)
m_abc = nn.Sequential(A,B,C)
m_abcd = nn.Sequential(A,B,C,D)
print("Model: %s" % m_abcd)
# print(dir(m))


for i in abcd.test_values:
    input = torch.ones(in_size) * i
    output_a = m_a(input)
    output_ab = m_ab(input)
    output_abc = m_abc(input)
    output_abcd = m_abcd(input)

    print(f"Output calculated by pytorch (input {input}) - A={'%f'%output_a}\tB={'%f'%output_ab}\tC={'%f'%output_abc}\tD={'%f'%output_abcd}\t")

# Export the model
fn = "ABCD_from_torch.onnx"
torch_out = torch.onnx._export(
    m_abcd,  # model being run
    input,  # model input (or a tuple for multiple inputs)
    fn,  # where to save the model (can be a file or file-like object)
    export_params=True,
)  # store the trained parameter weights inside the model file

print("Done! Exported to: %s" % fn)

import onnx

onnx_model = onnx.load(fn)
# print('Model: %s'%onnx_model)

def info(a):
    print(f"Info: {a.name} ({a.type}), {a.shape}")

import onnxruntime as rt

sess = rt.InferenceSession(fn)
info(sess.get_inputs()[0])
info(sess.get_outputs()[0])

for i in abcd.test_values:

    x = np.array([i], np.float32)

    res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})
    print(f"Output calculated by onnxruntime (input: {x}):  {res}")

print("Done! ONNX inference")
