import torch
import torch.nn as nn
import numpy as np
import pytest

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


def test_simple_convolution(simple_convolution_pytorch):
    """Test a simple convolution neural network model"""
    x = torch.zeros((1, 1, 28, 28))
    ebv_output = torch.zeros((10,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = simple_convolution_pytorch(x).detach().numpy()

    mdf_model, params_dict = pytorch_to_mdf(
        model=simple_convolution_pytorch,
        args=(x),
        example_outputs=output,
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    assert np.allclose(output, eg.enodes["Gemm_18"].evaluable_outputs["_18"].curr_value)


def test_convolution(convolution_pytorch):
    """Test a convolution neural network with more layers"""
    x = torch.zeros((1, 1, 28, 28))
    ebv_output = torch.zeros((10,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = convolution_pytorch(x).detach().numpy()

    mdf_model, params_dict = pytorch_to_mdf(
        model=convolution_pytorch,
        args=(x),
        example_outputs=output,
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    assert np.allclose(output, eg.enodes["Gemm_23"].evaluable_outputs["_23"].curr_value)


def test_vgg16(vgg16_pytorch):
    """Test a dummy vgg16 model"""
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    ebv_output = torch.zeros((1,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = vgg16_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=vgg16_pytorch,
        args=(x),
        example_outputs=output,
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    assert np.allclose(output, eg.enodes["Gemm_78"].evaluable_outputs["_78"].curr_value)


def test_resnet18(resnet18_pytorch):
    """Test a standard resnet model imported from PyTorch"""
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    ebv_output = torch.zeros((1,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = resnet18_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=resnet18_pytorch,
        args=(x),
        example_outputs=output,
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    assert np.allclose(
        output, eg.enodes["Gemm_191"].evaluable_outputs["_191"].curr_value
    )


@pytest.mark.xfail
def test_mobilenetv2(mobilenetv2_pytorch):
    """Test a standard mobilenetv2 model"""
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    ebv_output = torch.zeros((1,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = mobilenetv2_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=mobilenetv2_pytorch,
        args=(x),
        example_outputs=output,
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    assert np.allclose(
        output, eg.enodes["Gemm_536"].evaluable_outputs["_536"].curr_value
    )


if __name__ == "__main__":
    test_simple_module()
    test_simple_function()
    test_inception()
    test_simple_convolution()
    test_convolution()
    test_vgg16()
    test_resnet18()
    test_mobilenetv2()
