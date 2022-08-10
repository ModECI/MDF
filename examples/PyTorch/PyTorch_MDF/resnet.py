import torchvision.models as models

import torch
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

resnet18 = models.resnet18(pretrained=False)


def main():
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    from torchvision import datasets, transforms
    from torchvision.io import read_image

    transform = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    images = datasets.ImageFolder("pytorch_example_images/", transform=transform)
    dataloader = torch.utils.data.DataLoader(images, batch_size=5)
    x = next(iter(dataloader))[0]  # discard the lable

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    resnet18.eval()

    # Run the model once to get some ground truth outpot (from PyTorch)
    # with torch.no_grad():
    output = resnet18(x).detach().numpy()
    # print(output)

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=resnet18,
        args=(x),
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    # Output the model to JSON
    mdf_model.to_json_file("resnet.json")


if __name__ == "__main__":
    main()
