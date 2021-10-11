"""
This file does three things:
    - It implements a simple PyTorch model.
    - Exports in to ONNX using a combination of tracing and scripting
    - Converts it to MDF
"""
import torch
import onnx

from onnx import helper

from modeci_mdf.interfaces.onnx import onnx_to_mdf, convert_file


class A(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


class B(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


class C(torch.nn.Module):
    def forward(self, x):
        return torch.cos(x)


class ABC(torch.nn.Module):
    def __init__(self):
        super(ABC, self).__init__()
        self.A = A()
        self.B = B()
        self.C = C()

    def forward(self, x):

        # Run A
        y = self.A(x)

        # Run B (loop_count times)
        y = self.B(y)

        # Run C
        y = self.C(y)

        return y

def main():

    model = ABC()
    dummy_input = torch.zeros(2, 3)
    # loop_count = torch.tensor(5, dtype=torch.long)
    torch.onnx.export(model,
                      (dummy_input),
                      'abc_basic.onnx',
                      verbose=True,
                      input_names=['input'])
	

    # Load it back in using ONNX package
    onnx_model = onnx.load("abc_basic.onnx")
    print(onnx_model)
    onnx.checker.check_model(onnx_model)

    # Extract the loop or if body as a sub-model, this is just because I want
    # to view it in netron and sub-graphs can't be rendered
    for node in [node for node in onnx_model.graph.node if node.op_type in ["Loop", 'If']]:

        # Get the GraphProto of the body
        body_graph = node.attribute[0].g

        # Turn it into a model
        model_def = helper.make_model(body_graph, producer_name='abc_basic.py')

        onnx.save(model_def, f'examples/{node.name}_body.onnx')


    convert_file("abc_basic.onnx")


if __name__ == "__main__":
    main()
