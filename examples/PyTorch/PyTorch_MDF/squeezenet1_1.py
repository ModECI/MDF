import torchvision.models as models
import torch
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

squeezenet1_1 = models.squeezenet1_1(pretrained=False)


def main():
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    ebv_output = torch.zeros((1,))

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    squeezenet1_1.eval()

    # Run the model once to get some ground truth outpot (from PyTorch)
    output = squeezenet1_1(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=squeezenet1_1,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    # Output the model to JSON
    mdf_model.to_json_file("squeezenet1_1.json")


if __name__ == "__main__":
    main()
