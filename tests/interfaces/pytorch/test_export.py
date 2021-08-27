import torch
import torch.nn as nn
import numpy as np

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

from modeci_mdf.utils import load_mdf_json
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf
from modeci_mdf.execution_engine import EvaluableGraph

from modeci_mdf.utils import load_mdf_json
import json

def _check_model(mdf_model):
    """A helper function to JIT compile a function or torch.nn.Module into Torchscript and convert to MDF and check it"""

    # Generate JSON
    mdf_model.to_json_file("test.json")

    # Load the JSON
    load_mdf_json("test.json")


def test_simple_module():
    """Test a simple torch.nn.Module"""

    class Simple(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    mdf_model, param_dict = pytorch_to_mdf(
        model=Simple(),
        args=(torch.tensor(0.0), torch.tensor(0.0)),
        example_outputs=(torch.tensor(0.0)),
        use_onnx_ops=True,
    )

    _check_model(mdf_model)


def test_simple_function():
    """Test a simple function"""

    def simple(x, y):
        return x + y

    mdf_model, param_dict = pytorch_to_mdf(
        model=simple,
        args=(torch.tensor(0.0), torch.tensor(0.0)),
        example_outputs=(torch.tensor(0.0)),
        use_onnx_ops=True,
    )

    _check_model(mdf_model)


def test_inception(inception_model_pytorch):
    """Test the InceptionBlocks model that WebGME folks provided us."""

    galaxy_images_output = torch.zeros((1, 5, 64, 64))
    ebv_output = torch.zeros((1,))
    # Run the model once to get some ground truth outpot (from PyTorch)
    output = inception_model_pytorch(galaxy_images_output, ebv_output).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=inception_model_pytorch,
        args=(galaxy_images_output, ebv_output),
        example_outputs=output,
        trace=True,
    )

    # Get the graph
    mdf_graph = mdf_model.graphs[0]


    # Add inputs to the parameters dict so we can feed this to the EvaluableGraph for initialization of all
    # graph inputs.
    params_dict["input1"] = galaxy_images_output.numpy()
    params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)


    eg.evaluate(initializer=params_dict)

    assert np.allclose(
        output,
        eg.enodes["Add_381"].evaluable_outputs["_381"].curr_value,
    )



if __name__ == '__main__':
    test_simple_module()
