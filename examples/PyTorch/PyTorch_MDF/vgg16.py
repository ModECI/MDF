import numpy as np
import torch

import torch.nn.functional as F
from torch import nn
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )

        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )

        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )

        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv4_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )

        self.conv5_1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv5_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )
        self.conv5_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)  # dropout was included to combat overfitting
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x


# Hyperparameters
in_channels = 3
num_classes = 1


def get_pytorch_model():
    model = VGG16()
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
    output = model(x).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=model,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    # Output the model to JSON
    mdf_model.to_json_file("vgg16.json")


if __name__ == "__main__":
    main()
