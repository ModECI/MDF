#%%
import psyneulink as pnl

LEAK = 0.2
COMPETITION = 0.2
THRESHOLD = 0.4
GAIN = 2.0

input_mech = pnl.TransferMechanism(size=2, name="input_layer")

lca = pnl.LCAMechanism(
    size=2,
    function=pnl.Logistic(gain=GAIN),
    leak=LEAK,
    competition=COMPETITION,
    self_excitation=0,
    noise=0.1,
    threshold=THRESHOLD,
    termination_measure=pnl.TimeScale.TRIAL,
    time_step_size=0.1,
    name="accumulator_layer",
)

comp = pnl.Composition(pathways=[input_mech, lca])

comp.show_graph(show_projection_labels=True)
