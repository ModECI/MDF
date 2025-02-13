import torchvision.models as models
import torch
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

vgg19 = models.vgg19(pretrained=False)


def get_pytorch_model():
    model = models.vgg19(pretrained=False)
    return model


def get_example_input():
    x = torch.zeros((1, 3, 224, 224))
    return x


def main():
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = get_example_input()
    ebv_output = torch.zeros((1,))

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    model = get_pytorch_model()
    model.eval()

    # Run the model once to get some ground truth outpot (from PyTorch)
    output = vgg19(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=model,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    # Output the model to JSON
    mdf_model.to_json_file("vgg19.json")


if __name__ == "__main__":
    main()
