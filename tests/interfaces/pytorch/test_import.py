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

    try:

        import torchvision.models
        from torchvision.models import get_model_builder, list_models

    except ModuleNotFoundError:
        pytest.mark.skip(
            "Skipping PyTorch interface tests because pytorch is not installed."
        )
        return []

    def list_model_fns(module):
        return [(name, get_model_builder(name)) for name in list_models(module)]

    # Copied from https://github.com/pytorch/vision/blob/main/test/test_models.py
    skipped_big_models = {
        "vit_h_14": {("Windows", "cpu"), ("Windows", "cuda")},
        "regnet_y_128gf": {("Windows", "cpu"), ("Windows", "cuda")},
        "mvit_v1_b": {("Windows", "cuda"), ("Linux", "cuda")},
        "mvit_v2_s": {("Windows", "cuda"), ("Linux", "cuda")},
        "swin_t": {},
        "swin_s": {},
        "swin_b": {},
        "swin_v2_t": {},
        "swin_v2_s": {},
        "swin_v2_b": {},
    }

    # Copied from https://github.com/pytorch/vision/blob/main/test/test_models.py
    # speeding up slow models:
    slow_models = [
        "convnext_base",
        "convnext_large",
        "resnext101_32x8d",
        "resnext101_64x4d",
        "wide_resnet101_2",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
        "regnet_y_16gf",
        "regnet_y_32gf",
        "regnet_y_128gf",
        "regnet_x_16gf",
        "regnet_x_32gf",
        "swin_t",
        "swin_s",
        "swin_b",
        "swin_v2_t",
        "swin_v2_s",
        "swin_v2_b",
    ]

    if models is None:
        return []

    models_to_test = []
    model_classes = set()
    for model_name, model in list_model_fns(torchvision.models):
        try:

            if model_name in skipped_big_models:
                continue

            params = inspect.signature(model).parameters

            # Get the model class that this construction function returns. To cut down on tests,
            # lets only test one version of each model.
            return_type = inspect.signature(model).return_annotation

            if (
                "weights" in params or "pretrained" in params
            ) and return_type not in model_classes:
                models_to_test.append(model)
                if return_type:
                    model_classes.add(return_type)

        except TypeError:
            continue

    # New API for specifying pretrained=False is weights=None. pretrained keyword
    # will be removed soon. This handles that for all models depending on PyTorch
    # version.
    is_new_weights_api = "weights" in inspect.signature(models.resnet18).parameters
    model_weights_spec = (
        {"weights": None} if is_new_weights_api else {"pretrained": False}
    )

    xfails = {
        "inception_v3": "Inception-V3 is failing to match currently.",
        "maxvit_t": "MaxViT is failing because we are trying to call ast.parse on a string that is not valid python."
        " Need to handle string arguments requried by einops.",
        "resnet101": "Resnet101 is failing to match currently.",
        "vit_": "ViT models are failing because PyTorch cant convert to ONNX the unflatten op.",
    }

    pytest_params = []
    for model in models_to_test:

        if model.__name__ not in slow_models:
            t = (model, model_weights_spec, torch.rand((1, 3, 224, 224)))
        else:
            t = (model, model_weights_spec, torch.rand((1, 3, 64, 64)))

        xf_models = [n for n in xfails.keys() if n in model.__name__]
        if len(xf_models) > 0:
            xf_reason = xfails[xf_models[0]]
            pytest_params.append(
                pytest.param(
                    *t,
                    marks=pytest.mark.xfail(
                        reason=xf_reason,
                    ),
                )
            )
        else:
            pytest_params.append(t)

    return pytest_params


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
        atol=1e-05,
    ), f"Output from PyTorch and MDF do not match. MaxAbsError={np.max(np.abs(output - output_mdf))}"

    # Convert to JSON and back
    mdf_model2 = Model.from_json(mdf_model.to_json())


@pytest.mark.parametrize("model_init, kwargs, input", _get_torchvision_models())
def test_torchvision_models(model_init, kwargs, input):
    """Test importing the PyTorch model into MDF, executing in execution engine"""
    _run_and_check_model(model_init(**kwargs), input=input)


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
