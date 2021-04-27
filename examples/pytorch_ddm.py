#%%
from modeci_mdf.export.torchscript.converter import torchscript_to_mdf

import time
import re
import torch
from torch import nn

from typing import Tuple

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


@torch.jit.script
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

#%%

from torch.onnx.utils import _model_to_graph
from torch.onnx import TrainingMode

graph, params_dict, torch_out = _model_to_graph(
    model=torch.jit.script(ddm),
    args=tuple(ddm_params.values()),
    example_outputs=(rt, decision),
    do_constant_folding=False,
    training=TrainingMode.EVAL,
    _retain_param_name=True,
    operator_export_type=torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    dynamic_axes={},
)


# mdf_model = torchscript_to_mdf(ddm)
#
# print(mdf_model.to_yaml())

#%%
# with open('ddm.onnx', 'wb') as file:
#     torch.onnx.export(model=torch.jit.script(ddm),
#                       args=tuple(ddm_params.values()),
#                       example_outputs=(rt, decision),
#                       f=file,
#                       verbose=True,
#                       opset_version=12,
#                       operator_export_type=torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

#%%
# import seaborn as sns
# sns.kdeplot(rts)

#%%
