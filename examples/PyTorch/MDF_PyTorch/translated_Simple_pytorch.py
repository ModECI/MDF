"Converted examples from MDF models version(mdf.0) to PyTorch/ONNX"
import torch
import torch.nn as nn
import onnx
import onnxruntime as rt
from math import *


class input_node(nn.Module):
    def __init__(
        self,
        input_level=torch.tensor(0.5),
    ):
        super().__init__()
        self.input_level = input_level
        self.execution_count = torch.tensor(0)

    def forward(
        self,
    ):
        self.execution_count = self.execution_count + torch.tensor(1)
        return self.input_level


class processing_node(nn.Module):
    def __init__(
        self,
        lin_slope=torch.tensor(0.5),
        lin_intercept=torch.tensor(0),
        log_gain=torch.tensor(3),
    ):
        super().__init__()
        self.lin_slope = lin_slope
        self.lin_intercept = lin_intercept
        self.log_gain = log_gain
        self.execution_count = torch.tensor(0)

    def forward(self, input_port1):
        self.execution_count = self.execution_count + torch.tensor(1)
        linear_1 = input_port1 * self.lin_slope + self.lin_intercept
        logistic_1 = 1 / (1 + exp(-1 * self.log_gain * (linear_1 + 0) + 0))
        return logistic_1


class Model(nn.Module):
    def __init__(
        self,
        input_node,
        processing_node,
    ):
        super().__init__()
        self.input_node = input_node
        self.processing_node = processing_node

    def forward(self, input):
        val_input_node = torch.zeros_like(input)
        val_processing_node = torch.zeros_like(input)

        val_input_node = val_input_node + self.input_node()
        val_processing_node = val_processing_node + self.processing_node(
            val_input_node * torch.tensor(0.55)
        )

        return (
            val_input_node,
            val_processing_node,
        )


model = Model(
    input_node=input_node(),
    processing_node=processing_node(),
)
model = torch.jit.script(model)
dummy_input = torch.tensor(
    0.5,
)
output = model(dummy_input)
torch.onnx.export(
    model,
    dummy_input,
    "Simple.onnx",
    verbose=True,
    input_names=[],
    opset_version=9,
)
onnx_model = onnx.load("Simple.onnx")
onnx.checker.check_model(onnx_model)
sess = rt.InferenceSession("Simple.onnx")
res = sess.run(None, {sess.get_inputs()[0].name: dummy_input.numpy()})
if __name__ == "__main__":
    print("Exported to PyTorch and ONNX")
