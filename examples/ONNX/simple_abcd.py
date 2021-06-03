"""
This file does three things:
    - It implements a simple PyTorch model.
    - Exports in to ONNX using a combination of tracing and scripting
    - Converts it to MDF
"""
import torch
import onnx

from onnx import helper

from modeci_mdf.interfaces.onnx import onnx_to_mdf


class SimpleIntegrator(torch.nn.Module):
    def __init__(self, shape, rate):
        super().__init__()
        self.previous_value = torch.zeros(shape)
        self.rate = rate

    def forward(self, x):
        value = self.previous_value + (x * self.rate)
        self.previous_value = value
        return value


class Linear(torch.nn.Module):
    def __init__(self, slope=1.0, intercept=0.0):
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def forward(self, x):
        return self.slope * x + self.intercept


class ABCD(torch.nn.Module):
    def __init__(self, A, B, C, D):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def forward(self, x):

        # Since we are implementing conditions that reference the number of calls
        # to A and B, we need to keep track of this.
        num_A_calls = 0
        num_B_calls = 0

        # We need to initialize outputs, torchscript jit complains if c and d
        # are not defined in the FALSE branches of our conditionals.
        a = torch.zeros_like(x)
        b = torch.zeros_like(x)
        c = torch.zeros_like(x)
        d = torch.zeros_like(x)

        for i in range(10):

            # A: pnl.AtNCalls(A, 0),
            if num_A_calls == 0:
                a = self.A(x)
                num_A_calls = num_A_calls + 1

            # B: pnl.Always()
            b = self.B(a)
            num_B_calls = num_B_calls + 1

            # C: pnl.EveryNCalls(B, 5),
            if num_B_calls % 5 == 0:
                c = self.C(b)

            # D: pnl.EveryNCalls(B, 10)
            if num_B_calls % 10 == 0:
                d = self.D(b)

        return c, d


def main():

    slope = torch.ones((1, 1)) * 2.0
    intercept = torch.ones((1, 1)) * 2.0
    model = ABCD(
        A=Linear(slope=slope, intercept=intercept),
        B=Linear(slope=slope, intercept=intercept),
        C=Linear(slope=slope, intercept=intercept),
        D=Linear(slope=slope, intercept=intercept),
    )

    model = torch.jit.script(model)

    output = model(torch.ones((1, 1)))

    print(output)

    dummy_input = torch.ones((1, 1))
    torch.onnx.export(
        model,
        (dummy_input),
        "abcd.onnx",
        verbose=True,
        input_names=["input"],
        example_outputs=output,
        opset_version=9,
    )

    # Load it back in using ONNX package
    onnx_model = onnx.load("abcd.onnx")
    onnx.checker.check_model(onnx_model)

    mdf_model = onnx_to_mdf(onnx_model)

    mdf_model.to_json_file("abcd.json")
    mdf_model.to_yaml_file("abcd.yaml")



if __name__ == "__main__":
    main()
