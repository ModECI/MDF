"""
Tests for importing of PyTorch models into MDF. These tests use a lot of fixtures for models setup in ./conftest.py
"""
import pytest
import inspect
import numpy as np

from modeci_mdf.mdf import Model
from modeci_mdf.execution_engine import EvaluableGraph

try:
    from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

    import torch
    import torchvision.models as models

except ModuleNotFoundError:
    models = None
    pytest.mark.skip(
        "Skipping PyTorch interface tests because pytorch is not installed."
    )


def _get_torchvision_models():
    """
    Get all the backbone models in torch vision, suprised there is no function to do this in torchvision.
    """

    if models is None:
        return []

    models_to_test = []
    model_classes = set()
    for model_name, model in models.__dict__.items():
        try:
            params = inspect.signature(model).parameters

            # Get the model class that this construction function returns. To cut down on tests,
            # lets only test one version of each model.
            return_type = inspect.signature(model).return_annotation

            if (
                "weights" in params
                or "pretrained" in params
                and return_type not in model_classes
            ):
                models_to_test.append(model)
                if return_type:
                    model_classes.add(return_type)

        except TypeError:
            continue

    # New API for specifying pretrained=False is weughts=None. pretrained keyword
    # will be removed soon. This handles that for all models depending on PyTorch
    # version.
    is_new_weights_api = "weights" in inspect.signature(models.resnet18).parameters
    model_weights_spec = (
        {"weights": None} if is_new_weights_api else {"pretrained": False}
    )

    return [(model, model(**model_weights_spec)) for model in models_to_test]


def _run_and_check_model(model, input=None):
    """
    Helper function that runs a complete set of tests on a model
        - Runs the model in PyTorch to get expected output.
        - Converts model to MDF and runs in Python execution engine.
        - Compares the results.
    """

    # Create some test inputs for the model
    if input is None:
        input = torch.rand((1, 3, 224, 224))

    # Get rid of randomization due to Dropout
    model.eval()

    with torch.no_grad():
        # Run the model once to get some ground truth output (from PyTorch)
        output = model(input).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=model,
        args=input,
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    params_dict["input1"] = input.numpy()

    eg = EvaluableGraph(graph=mdf_graph, verbose=False)

    eg.evaluate(initializer=params_dict)

    output_mdf = eg.output_enodes[0].get_output()
    assert np.allclose(
        output,
        output_mdf,
        rtol=1e-03,
    )

    # Convert to JSON and back
    mdf_model2 = Model.from_json(mdf_model.to_json())


@pytest.mark.parametrize("model_init, model", _get_torchvision_models())
def test_torchvision_models(model_init, model):
    """Test importing the PyTorch model into MDF, executing in execution engine"""
    _run_and_check_model(model)


def test_simple_convolution(simple_convolution_pytorch):
    """Test a simple convolution neural network model"""
    _run_and_check_model(simple_convolution_pytorch, torch.zeros((1, 1, 28, 28)))


def test_convolution(convolution_pytorch):
    """Test a convolution neural network with more layers"""
    _run_and_check_model(convolution_pytorch, torch.zeros((1, 1, 28, 28)))


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

    mdf_model2 = Model.from_json(mdf_model.to_json())


def test_simple_function():
    """Test a simple function"""

    def simple(x, y):
        return x + y

    mdf_model, param_dict = pytorch_to_mdf(
        model=simple,
        args=(torch.tensor(0.0), torch.tensor(0.0)),
        use_onnx_ops=True,
    )

    mdf_model2 = Model.from_json(mdf_model.to_json())


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

    mdf_model2 = Model.from_json(mdf_model.to_json())
