"Converted examples from MDF models version(mdf.s) to PyTorch/ONNX"
import torch
import torch.nn as nn
import onnx
import onnxruntime as rt
from math import *


class input_node(nn.Module):
    def __init__(
        self,
        input_level=torch.tensor([[1, 2.0], [3, 4]]),  # orig type: <class 'list'>
    ):
        super().__init__()
        self.input_level = input_level
        self.execution_count = torch.tensor(0)

    def forward(
        self,
    ):
        self.execution_count = self.execution_count + torch.tensor(1)
        return self.input_level


class middle_node(nn.Module):
    def __init__(
        self,
        slope=torch.tensor(0.5),  # orig type: <class 'float'>
        intercept=torch.tensor([[0.0, 1.0], [2.0, 2.0]]),  # orig type: <class 'list'>
    ):
        super().__init__()
        self.slope = slope
        self.intercept = intercept
        self.execution_count = torch.tensor(0)

    def forward(
        self,
        input_port1,
    ):
        self.execution_count = self.execution_count + torch.tensor(1)
        linear_1 = input_port1 * self.slope + self.intercept
        return linear_1


class Model(nn.Module):
    def __init__(
        self,
        input_node,
        middle_node,
    ):
        super().__init__()
        self.input_node = input_node
        self.middle_node = middle_node

    def forward(self, input):
        val_input_node = torch.zeros_like(input)
        val_middle_node = torch.zeros_like(input)

        val_input_node = val_input_node + self.input_node()
        val_middle_node = val_middle_node + self.middle_node(
            val_input_node * torch.tensor([[1, 0], [0, 1]])
        )

        return (
            val_input_node,
            val_middle_node,
        )


model = Model(
    input_node=input_node(),
    middle_node=middle_node(),
)
model = torch.jit.script(model)
dummy_input = torch.tensor(
    [[1, 2.0], [3, 4]],
)
output = model(dummy_input)
torch.onnx.export(
    model,
    dummy_input,
    "Arrays.onnx",
    verbose=True,
    input_names=[],
    opset_version=9,
)
onnx_model = onnx.load("Arrays.onnx")
onnx.checker.check_model(onnx_model)
sess = rt.InferenceSession("Arrays.onnx")
# res = sess.run(None, {sess.get_inputs()[0].name: dummy_input.numpy()})
if __name__ == "__main__":
    print("Exported to PyTorch and ONNX")
