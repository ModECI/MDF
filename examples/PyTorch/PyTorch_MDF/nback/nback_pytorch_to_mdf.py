#%%
import torch
import numpy as np

from modeci_mdf.interfaces.pytorch import pytorch_to_mdf
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.mdf import Model

# exp params (imported)
# context params
CDIM = 25

# layers
SDIM = 20


class FFWM(torch.nn.Module):
    """
    Model used for Sternberg and N-back

    Source: https://github.com/andrebeu/nback-paper/blob/master/utilsWM.py

    Slight modification in which concatenation of input tensors was removed
    as the first op of the model and handled as pre-processing. This was
    done to allow execution in the MDF execution engine which currently does
    not seem to support Concatenating a list of tensors.
    """

    def __init__(self, indim, hiddim, outdim=2, bias=False):
        super().__init__()
        self.indim = indim
        self.hiddim = hiddim
        self.hid1_layer = torch.nn.Linear(indim, indim, bias=bias)
        self.hid2_layer = torch.nn.Linear(indim, hiddim, bias=bias)
        self.out_layer = torch.nn.Linear(hiddim, outdim, bias=bias)
        self.drop2 = torch.nn.Dropout(p=0.05, inplace=False)
        bias_dim = indim
        max_num_bias_modes = 10
        self.embed_bias = torch.nn.Embedding(max_num_bias_modes, bias_dim)
        return None

    def forward(self, inputL, control_bias_int=0):
        """inputL is list of tensors"""
        hid1_act = self.hid1_layer(inputL).relu()
        control_bias = self.embed_bias(torch.tensor(control_bias_int))
        hid2_in = hid1_act + control_bias
        hid2_in = self.drop2(hid2_in)
        hid2_act = self.hid2_layer(hid2_in).relu()
        yhat_t = self.out_layer(hid2_act)
        return yhat_t


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

# init net
indim = 2 * (CDIM + SDIM)
hiddim = SDIM * 4
net = FFWM(indim, hiddim)

dummy_input = torch.cat(
    [torch.rand(20), torch.rand(20), torch.rand(25), torch.rand(25)], -1
)

# net.eval()
output = net(dummy_input).detach().numpy()

mdf_model, params_dict = pytorch_to_mdf(
    model=net,
    args=(dummy_input),
    trace=True,
)

# Get the graph
mdf_graph = mdf_model.graphs[0]
params_dict["input1"] = dummy_input.detach().numpy()

eg = EvaluableGraph(graph=mdf_graph, verbose=False)

eg.evaluate(initializer=params_dict)

mdf_model.to_json_file("nback_mdf.json")

output_mdf = eg.output_enodes[0].get_output()
assert np.allclose(
    output,
    output_mdf,
), f"Output from PyTorch and MDF do not match. MaxAbsError={np.max(np.abs(output - output_mdf))}"

# Convert to JSON and back
mdf_model2 = Model.from_json(mdf_model.to_json())
