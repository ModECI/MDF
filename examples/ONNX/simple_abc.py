"""
This file does three things:
    - It implements a simple PyTorch model.
    - Exports in to ONNX using a combination of tracing and scripting
    - Converts it to MDF
"""
import torch
import onnx


from modeci_mdf.interfaces.onnx import onnx_to_mdf


class A(torch.nn.Module):
    def forward(self, x):
        return x + 1


@torch.jit.script
def loop_b(x, y):
    for i in range(int(y)):
        x = x / 10
    return x


class B(torch.nn.Module):
    def forward(self, x, y):
        return loop_b(x, y)


class C(torch.nn.Module):
    def forward(self, x):
        return x * 100


class ABC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = A()
        self.B = B()
        self.C = C()

    def forward(self, x, B_loop_count):

        # Run A
        y = self.A(x)

        # Run B (loop_count times)
        y = self.B(y, B_loop_count)

        # Run C
        y = self.C(y)

        return y


def main():

    model = ABC()
    dummy_input = torch.zeros(2, 3)
    loop_count = torch.tensor(5, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_input, loop_count),
        "abc.onnx",
        verbose=True,
        input_names=["input", "B_loop_count"],
        opset_version=9,
    )

    # Load it back in using ONNX package
    onnx_model = onnx.load("abc.onnx")
    onnx.checker.check_model(onnx_model)

    mdf_model = onnx_to_mdf(onnx_model)

    mdf_model.to_json_file("abc.json")
    mdf_model.to_yaml_file("abc.yaml")
    mdf_model.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=3,
        filename_root="abc",
        only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
    )


if __name__ == "__main__":
    main()
