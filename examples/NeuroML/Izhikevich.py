from neuromllite import Network, Cell, InputSource, Population, Synapse
from neuromllite import Projection, RandomConnectivity, Input, Simulation
import sys

################################################################################
###   Build new network

net = Network(id="IzhikevichTest")
net.notes = "Example Izhikevich"
net.parameters = {"N": 1}

if not "-iaf" in sys.argv:  # for testing...

    cell = Cell(id="izhCell", neuroml2_cell="izhikevich2007Cell")
    cell.parameters = {}

    params = {
        "v0": "-60mV",
        "C": "100 pF",
        "k": "0.7 nS_per_mV",
        "vr": "-60 mV",
        "vt": "-40 mV",
        "vpeak": "35 mV",
        "a": "0.03 per_ms",
        "b": "-2 nS",
        "c": "-50 mV",
        "d": "100 pA",
    }
else:

    cell = Cell(id="iaf", neuroml2_cell="iafCell")
    cell.parameters = {}

    params = {
        "leakReversal": "-50mV",
        "thresh": "-55mV",
        "reset": "-70mV",
        "C": "0.2nF",
        "leakConductance": "0.01uS",
    }

for p in params:
    cell.parameters[p] = p
    net.parameters[p] = params[p]

net.cells.append(cell)


pop = Population(
    id="izhPop", size="1", component=cell.id, properties={"color": ".7 0 0"}
)
net.populations.append(pop)

net.parameters["delay"] = "100ms"
net.parameters["stim_amp"] = "100pA"
net.parameters["duration"] = "500ms"
input_source = InputSource(
    id="iclamp_0",
    neuroml2_input="PulseGenerator",
    parameters={"amplitude": "stim_amp", "delay": "delay", "duration": "duration"},
)
net.input_sources.append(input_source)

net.inputs.append(
    Input(id="stim", input_source=input_source.id, population=pop.id, percentage=100)
)

print(net)
print(net.to_json())
new_file = net.to_yaml_file("%s.nmllite.yaml" % net.id)


################################################################################
###   Build Simulation object & save as JSON

recordVariables = {"v": {"all": "*"}}
if not "-iaf" in sys.argv:
    recordVariables["u"] = {"all": "*"}

sim = Simulation(
    id="Sim%s" % net.id,
    network=new_file,
    duration="700",
    dt="0.025",
    recordVariables=recordVariables,
)

sim.to_yaml_file()


################################################################################
###   Run in some simulators

from neuromllite.NetworkGenerator import check_to_generate_or_run
import sys

check_to_generate_or_run(sys.argv, sim)
