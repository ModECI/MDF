from modeci_mdf.interfaces.pytorch import pytorch_to_mdf

import time
import re
import torch
from torch import nn

from typing import Tuple

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


def ddm(
    starting_value: torch.Tensor,
    drift_rate: torch.Tensor,
    non_decision_time: torch.Tensor,
    threshold: torch.Tensor,
    noise: torch.Tensor,
    time_step_size: torch.Tensor,
):
    """
    A model that simulates a simple noisy drift diffusion model using Euler-Maruyama integration. This is implemented
    without performance in mind.

    Args:
        starting_value: The starting value of the particle.
        drift_rate: The constant drift rate the particles is under.
        non_decision_time: A constant amount of time added to the reaction time that signifies automatic processing.
        threshold: The threshold that a particle must reach to stop integration.
        noise: The standard deviation of the Gaussian noise added to the particles position at each time step.
        time_step_size: The time step size (in seconds) for the integration process.

    Returns:
       A two element tuple containing the reaction times and the decisions
    """

    x = starting_value
    rt = torch.tensor(0.0)

    # Integrate until the negative or positive threshold is reached
    while torch.abs(x) < threshold:
        x = x + torch.normal(mean=drift_rate * time_step_size, std=noise) * torch.sqrt(
            time_step_size
        )
        rt = rt + 1.0

    # Compute the actual reaction time and the decision (as a bool)
    rt = non_decision_time + rt * time_step_size
    decision = x >= threshold

    return rt, decision


def main():

    ddm_params = dict(
        starting_value=0.0,
        drift_rate=0.3,
        non_decision_time=0.15,
        threshold=0.6,
        noise=1.0,
        time_step_size=0.001,
    )

    # Move params to device
    for key, val in ddm_params.items():
        ddm_params[key] = torch.tensor(val).to(dev)

    # Run a single ddm
    rt, decision = ddm(**ddm_params)

    mdf_model, param_dict = pytorch_to_mdf(
        model=ddm,
        args=tuple(ddm_params.values()),
        example_outputs=(rt, decision),
        use_onnx_ops=True,
    )

    # Output the model to JSON
    mdf_model.to_json_file("ddm.json")

    import sys
    if "-graph" in sys.argv:
        mdf_model.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=2,
            filename_root="ddm",
            only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
