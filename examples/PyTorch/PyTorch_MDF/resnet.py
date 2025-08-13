import torchvision.models as models

import torch
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf


def get_pytorch_model():
    model = models.resnet18(pretrained=False)
    return model


def get_example_input():
    x = torch.rand((5, 3, 224, 224))
    return x


def main():
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = get_example_input()

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    model = get_pytorch_model()
    model.eval()

    # Run the model once to get some ground truth output (from PyTorch)
    # with torch.no_grad():
    output = model(x).detach().numpy()
    # print(output)

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=model,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    # Output the model to JSON
    mdf_model.to_json_file("resnet.json")


if __name__ == "__main__":
    main()
