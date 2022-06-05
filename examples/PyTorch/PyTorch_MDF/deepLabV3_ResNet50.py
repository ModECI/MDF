import torchvision.models as models
import torch
from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=False)


def main():
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    x = torch.zeros((1, 3, 224, 224))
    ebv_output = torch.zeros((1, 3, 224, 224))

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    deeplabv3.eval()

    # Run the model once to get some ground truth outpot (from PyTorch)
    output = deeplabv3(x)

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=deeplabv3,
        args=(x),
        example_outputs=output,
        trace=True,
    )
    # Get the graph
    mdf_graph = mdf_model.graphs[0]
    # Output the model to JSON
    mdf_model.to_json_file("deepLabV3_ResNet50.json")


if __name__ == "__main__":
    main()
