"""
This file does three things:
    - It implements a simple PyTorch model.
    - Exports in to ONNX using a combination of tracing and scripting
    - Converts it to MDF
"""
import torch
import onnx
import sys


from modeci_mdf.interfaces.onnx import onnx_to_mdf


class A(torch.nn.Module):
    def forward(self, x):
        return x + 1


class B(torch.nn.Module):
    def forward(self, x):
        return x * 5


class AB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = A()
        self.B = B()

    def forward(self, x):

        # Run A
        y = self.A(x)

        # Run B
        y = self.B(y)

        return y


def main():

    model = AB()
    dummy_input = torch.zeros(2, 3)

    output = model(dummy_input)

    print("PyTorch model: %s created. Evaluates %s as %s"%(model,dummy_input,output))

    torch.onnx.export(
        model,
        (dummy_input),
        "ab.onnx",
        verbose=True,
        input_names=["input"],
        opset_version=9,
    )

    # Load it back in using ONNX package
    onnx_model = onnx.load("ab.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime as rt

    sess = rt.InferenceSession("ab.onnx")

    res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: dummy_input.numpy()})
    print(f"Output calculated by onnxruntime (input: {dummy_input}):  {res}")

    mdf_model = onnx_to_mdf(onnx_model)
    mdf_model.to_json_file("ab.json")
    mdf_model.to_yaml_file("ab.yaml")

    mdf_model.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=3,
        filename_root="ab",
        only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
    )
    if "-run" in sys.argv:
        verbose = True
        verbose = False

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mdf_model.graphs[0],
                            verbose=verbose)

        print('Evaluating graph...')
        test_values = [0,1,[1,2],dummy_input.numpy()]
        test_values = [0,1,[1,2]]

        for t in test_values:
            print("===================\nEvaluating MDF model with input: %s"%t)
            eg.evaluate(initializer={"input":t})
            print("Output: %s"%eg.enodes['Mul_3'].evaluable_outputs['_4'].curr_value)


if __name__ == "__main__":
    main()
