import numpy as np
import torch
import torch.nn.functional as F
from torch import nn  # All neural network modules
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.fc1 = nn.Linear(8 * 25 * 25, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Hyperparameters
in_channels = 1
num_classes = 10


def get_pytorch_model():
    model = CNN(in_channels=in_channels, num_classes=num_classes)
    return model


def get_example_input():
    x = torch.zeros((1, 1, 28, 28))
    return x


def main():
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = get_example_input()
    ebv_output = torch.zeros((10,))

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    model = get_pytorch_model()
    model.eval()

    # Run the model once to get some ground truth outpot (from PyTorch)
    output = model(x)

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=model,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    # Output the model to JSON
    mdf_model.to_json_file("simple_convolution.json")


if __name__ == "__main__":
    main()
