import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as rt
from torchviz import make_dot
import netron
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf
import os


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(150528, 240)  # Input is calculated as 224*224*3=150528
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 50)
        self.fc4 = nn.Linear(50, 2)  # Output node with 2 classes

    def forward(self, x):
        x = x.view(-1, 150528)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def main():
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    input_images = torch.zeros((1, 3, 224, 224), requires_grad=False)

    # Seed the random number generator to get deterministic behavior for weight initialization
    torch.manual_seed(0)

    model = SimpleNet()

    model.eval()
    # Run the model once to get some ground truth outpot (from PyTorch)
    output = model(input_images)

    from modelspec.utils import _val_info

    print("Evaluated the graph in PyTorch, output: %s" % (_val_info(output)))

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=model,
        args=(input_images),
        trace=True,
    )

    # Get the graph
    mdf_graph = mdf_model.graphs[0]

    # Add inputs to the parameters dict so we can feed this to the EvaluableGraph for initialization of graph input.
    params_dict["input1"] = input_images.numpy()

    # Evaluate the model via the MDF scheduler
    eg = EvaluableGraph(graph=mdf_graph, verbose=False)
    eg.evaluate(initializer=params_dict)
    output_mdf = eg.output_enodes[0].get_output()

    print("Evaluated the graph in PyTorch, output: %s" % (_val_info(output_mdf)))

    # Make sure the results are the same between PyTorch and MDF
    assert np.allclose(
        output.detach().numpy(),
        output_mdf,
    )
    print("Passed all comparison tests!")

    # Output the model to JSON
    mdf_model.to_json_file("simple_pytorch_to_mdf.json")

    import sys

    # Exporting as onnx model
    torch.onnx.export(
        model,
        input_images,
        "simple_pytorch_to_mdf.onnx",
        verbose=True,
        input_names=[],
        opset_version=9,
    )
    onnx_model = onnx.load("simple_pytorch_to_mdf.onnx")
    onnx.checker.check_model(onnx_model)
    sess = rt.InferenceSession("simple_pytorch_to_mdf.onnx")
    res = sess.run(None, {sess.get_inputs()[0].name: input_images.numpy()})
    print("Exported to MDF and ONNX")

    # export to mdf graph
    if "-graph" in sys.argv:
        mdf_model.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=1,
            filename_root="simple_pytorch_to_mdf.1",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
            is_horizontal=True,
            solid_color=True,
        )
        mdf_model.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="simple_pytorch_to_mdf",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
            solid_color=False,
        )
    # export to PyTorch graph
    if "-graph-torch" in sys.argv:
        make_dot(output, params=dict(list(model.named_parameters()))).render(
            "simple_pytorch_to_mdf_torchviz", format="png"
        )
    # export to onnx graph
    if "-graph-onnx" in sys.argv:
        netron.start("simple_pytorch_to_mdf.onnx")


if __name__ == "__main__":
    main()
