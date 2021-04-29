import torch
import torch.nn as nn

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

from modeci_mdf.export.torchscript.converter import torchscript_to_mdf


def _check_model(mdf_model):
    """A helper function to JIT compile a function or torch.nn.Module into Torchscript and convert to MDF and check it"""

    # Generate JSON
    json_str = mdf_model.to_json()

    # Load the JSON
    # load_mdf_json()


def test_simple_module():
    """Test a simple torch.nn.Module"""

    class Simple(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    mdf_model = torchscript_to_mdf(
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

    mdf_model = torchscript_to_mdf(
        model=simple,
        args=(torch.tensor(0.0), torch.tensor(0.0)),
        example_outputs=(torch.tensor(0.0)),
        use_onnx_ops=True,
    )

    _check_model(mdf_model)
