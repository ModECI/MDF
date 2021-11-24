import numpy as np
import torch
import torch.nn as nn

from modeci_mdf.interfaces.pytorch import pytorch_to_mdf


class MlpBlocks(nn.Module):
    """An Mlp-like model provided as a test case by the WebGME folks"""

    def __init__(self):
        super().__init__()
        self.inp_hid = nn.Linear(196, 128, bias=True)
        self.relu1 = nn.ReLU()
        self.hid_hid = nn.Linear(128, 128, bias=True)
        self.relu2 = nn.ReLU()
        self.hid_out = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.inp_hid(x)

        x = self.relu1(x)
        x = self.hid_hid(x)
        x = self.relu2(x)
        x = self.hid_out(x)
        x = torch.argmax(x, dim=1)
        return x


def main():
    # changed import call
    from modeci_mdf.execution_engine import EvaluableGraph

    # Create some test inputs for the model
    images_input = torch.zeros((1, 14 * 14))

    # Seed the random number generator to get deterministic behavior for weight initialization
    torch.manual_seed(0)

    model = MlpBlocks()

 

    # Turn on eval mode for model to get rid of any randomization due to things like BatchNorm or Dropout
    model.eval()

    # Run the model once to get some ground truth outpot (from PyTorch)
    output = model(images_input).detach().numpy()

    # Convert to MDF
    mdf_model, params_dict = pytorch_to_mdf(
        model=model,
        args=images_input,
        example_outputs=output,
        trace=True,
    )

    # print(params_dict)
    # Get the graph
    mdf_graph = mdf_model.graphs[0]

    # Add inputs to the parameters dict so we can feed this to the EvaluableGraph for initialization of all
    # graph inputs.
    params_dict["input1"] = images_input.numpy()

    # Evaluate the model via the MDF scheduler
    eg = EvaluableGraph(graph=mdf_graph, verbose=False)
    print(eg.enodes)
    eg.evaluate(initializer=params_dict)

    assert np.allclose(
        output,
        eg.enodes["ArgMax_12"].evaluable_outputs["_12"].curr_value,
    )
    print("Passed all comparison tests!")

    # Output the model to JSON
    mdf_model.to_json_file("mlp_classifier_new.json")

    import sys

    if "-graph" in sys.argv:
        mdf_model.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=1,
            filename_root="mlp_classifier",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
