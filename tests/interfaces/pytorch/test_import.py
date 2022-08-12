"""
Tests for importing of PyTorch models into MDF. These tests use a lot of fixtures for models setup in ./conftest.py
"""
import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

except ModuleNotFoundError:
    pytest.mark.skip(
        "Skipping PyTorch interface tests because pytorch is not installed."
    )


from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import load_mdf_json


def test_simple_convolution(simple_convolution_pytorch):
    """Test a simple convolution neural network model"""
    x = torch.zeros((1, 1, 28, 28))
    ebv_output = torch.zeros((1,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = simple_convolution_pytorch(x).detach().numpy()

    mdf_model, params_dict = pytorch_to_mdf(
        model=simple_convolution_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    # params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(output, output_mdf)


def test_convolution(convolution_pytorch):
    """Test a convolution neural network with more layers"""
    x = torch.zeros((1, 1, 28, 28))
    ebv_output = torch.zeros((10,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = convolution_pytorch(x).detach().numpy()

    mdf_model, params_dict = pytorch_to_mdf(
        model=convolution_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    # params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
    )


def test_vgg16(vgg16_pytorch):
    """Test a dummy vgg16 model"""
    # changed import call

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    ebv_output = torch.zeros((1,))

    # Run the model once to get some ground truth output (from PyTorch)
    output = vgg16_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=vgg16_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    # params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
    )


def test_vgg19(vgg19_pytorch):
    """Test a dummy vgg19 model"""
    # changed import call

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    # Get rid of randomization due to Dropout
    vgg19_pytorch.eval()
    # Run the model once to get some ground truth output (from PyTorch)
    output = vgg19_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=vgg19_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
    )


def test_resnet18(resnet18_pytorch):
    """Test a standard resnet model imported from PyTorch"""
    # changed import call

    x = torch.rand((5, 3, 224, 224))

    # Run the model once to get some ground truth output (from PyTorch)
    with torch.no_grad():
        output = resnet18_pytorch(x)

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=resnet18_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()
    # params_dict["input2"] = ebv_output.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
        # We need to compare results with an epsilon, I think this is related to some ONNX warnings:
        #    Warning: ONNX Preprocess - Removing mutation from node aten::add_ on block input:
        #    'bn1.num_batches_tracked'. This changes graph semantics.
        atol=1e-5,
    )


def test_mobilenetv2(mobilenetv2_pytorch):
    """Test a standard mobilenetv2 model"""

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))

    # Run the model once to get some ground truth output (from PyTorch)
    output = mobilenetv2_pytorch(x)
    # with torch.no_grad():
    #     output = mobilenetv2_pytorch(x)

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=mobilenetv2_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
        # We need to compare results with an epsilon, I think this is related to some ONNX warnings:
        #    Warning: ONNX Preprocess - Removing mutation from node aten::add_ on block input:
        #    'bn1.num_batches_tracked'. This changes graph semantics.
        atol=1e-5,
    )


def test_shufflenetv2(shufflenetv2_pytorch):
    """Test a standard shufflenet_v2 model"""

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))

    # Run the model once to get some ground truth output (from PyTorch)
    output = shufflenetv2_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=shufflenetv2_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
    )


def test_resNext(resNext_pytorch):
    """Test a standard ResNext model"""

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))

    # Run the model once to get some ground truth output (from PyTorch)
    output = resNext_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=resNext_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
    )


def test_squeezeNet(squeezeNet_pytorch):
    """Test a standard SqueezeNet model"""

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    # Get rid of randomization due to Dropout
    squeezeNet_pytorch.eval()
    # Run the model once to get some ground truth output (from PyTorch)
    output = squeezeNet_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=squeezeNet_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
    )


def test_mnasNet(mnasNet_pytorch):
    """Test a standard MNASNet model"""

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    # Get rid of randomization due to Dropout
    mnasNet_pytorch.eval()
    # Run the model once to get some ground truth output (from PyTorch)
    output = mnasNet_pytorch(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=mnasNet_pytorch,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = x.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
    )


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
    output_mdf = eg.output_enodes[0].get_output()

    assert np.allclose(
        output,
        output_mdf,
    )


if __name__ == "__main__":
    test_simple_module()
