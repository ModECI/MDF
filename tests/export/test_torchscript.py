import torch
from modeci_mdf.export.torchscript.converter import torchscript_to_mdf


def _check_model(model):
    """A helper function to JIT compile a function or torch.nn.Module into Torchscript and convert to MDF and check it"""

    # JIT compile the model into TorchScript
    model = torch.jit.script(model)

    mdf_model = torchscript_to_mdf(model)

    # Generate JSON
    json_str = mdf_model.to_json()

    # Load the JSON
    # load_mdf_json()


def test_simple_module():
    """Test a simple torch.nn.Module"""

    class Simple(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    _check_model(Simple())


def test_simple_function():
    """Test a simple function"""

    def simple(x, y):
        return x + y

    _check_model(simple)
