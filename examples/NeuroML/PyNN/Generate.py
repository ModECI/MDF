from neuromllite import (
    Network,
    Cell,
    InputSource,
    Population,
    Synapse,
    RectangularRegion,
    RandomLayout,
)
from neuromllite import Projection, RandomConnectivity, Input, Simulation
import sys


def generate(
    ref,
    num_pop0=2,
    num_pop1=2,
    conn_prob=1,
    conn_weight=0.3,
    input_percentage=50,
    input_weight=".8",
    pynn_cell="IF_curr_alpha",
):

    ################################################################################
    ###   Build new network

    net = Network(id=ref)
    net.notes = "Example: %s" % ref
    net.parameters = {"input_amp": 0.99}

    cell = Cell(id="testcell", pynn_cell=pynn_cell)

    cell.parameters = {"i_offset": 0.0}
    if "IF_" in pynn_cell:
        cell.parameters["tau_refrac"] = 5
    else:
        cell.parameters = {"i_offset": 0.05}
        net.parameters = {"input_amp": 0}
    net.cells.append(cell)

    input_source = InputSource(
        id="i_clamp",
        pynn_input="DCSource",
        parameters={"amplitude": "input_amp", "start": 200.0, "stop": 800.0},
    )
    net.input_sources.append(input_source)

    r1 = RectangularRegion(
        id="region1", x=0, y=0, z=0, width=1000, height=100, depth=1000
    )
    net.regions.append(r1)

    p0 = Population(
        id="pop0",
        size=num_pop0,
        component=cell.id,
        properties={"color": "1 0 0", "radius": 20},
        random_layout=RandomLayout(region=r1.id),
    )
    net.populations.append(p0)

    if num_pop1 > 0:
        p1 = Population(
            id="pop1",
            size=num_pop1,
            component=cell.id,
            properties={"color": "0 1 0", "radius": 20},
            random_layout=RandomLayout(region=r1.id),
        )
        net.populations.append(p1)

    """p2 = Population(
        id="pop2",
        size=1,
        component=cell2.id,
        properties={"color": "0 0 1", "radius": 20},
        random_layout=RandomLayout(region=r1.id),
    )

    net.populations.append(p2)"""

    net.synapses.append(
        Synapse(
            id="ampaSyn",
            pynn_receptor_type="excitatory",
            pynn_synapse_type="curr_alpha",
            parameters={"tau_syn": 20},
        )
    )
    """net.synapses.append(
        Synapse(
            id="gabaSyn",
            pynn_receptor_type="inhibitory",
            pynn_synapse_type="cond_alpha",
            parameters={"e_rev": -80, "tau_syn": 10},
        )
    )"""

    if num_pop1 > 0:
        net.projections.append(
            Projection(
                id="proj0",
                presynaptic=p0.id,
                postsynaptic=p1.id,
                synapse="ampaSyn",
                delay=2,
                weight=conn_weight,
            )
        )
        net.projections[0].random_connectivity = RandomConnectivity(
            probability=conn_prob
        )

    """net.projections.append(
        Projection(
            id="proj1",
            presynaptic=p0.id,
            postsynaptic=p2.id,
            synapse="gabaSyn",
            delay=2,
            weight=0.01,
        )
    )
    net.projections[1].random_connectivity = RandomConnectivity(probability=1)"""

    net.inputs.append(
        Input(
            id="stim",
            input_source=input_source.id,
            population=p0.id,
            percentage=input_percentage,
            weight=input_weight,
        )
    )

    net_json_file = net.to_json_file("%s.json" % net.id)
    net_yaml_file = net.to_yaml_file("%s.yaml" % net.id)

    ################################################################################
    ###   Build Simulation object & save as JSON

    sim = Simulation(
        id="Sim%s" % net.id,
        network=net_json_file,
        duration="1000",
        dt="0.01",
        record_traces={"all": "*"},
        record_spikes={"all": "*"} if "IF_" in pynn_cell else {},
    )

    sim.to_json_file()
    sim.network = net_yaml_file
    sim.to_yaml_file("%s.yaml" % sim.id)

    sim.network = net_json_file  # reverting, for call below...

    return sim, net


if __name__ == "__main__":

    if "-one" in sys.argv:
        sim, net = generate("OneCell", num_pop0=1, num_pop1=0, input_percentage=100)
    elif "-hh" in sys.argv:
        sim, net = generate(
            "HH", num_pop0=1, num_pop1=0, input_percentage=0, pynn_cell="HH_cond_exp"
        )
    elif "-input_weights" in sys.argv:
        sim, net = generate(
            "InputWeights",
            num_pop0=4,
            num_pop1=0,
            input_percentage=62,
            input_weight=".8*random()",
        )
    elif "-simple_net" in sys.argv:
        sim, net = generate(
            "SimpleNet",
            num_pop0=1,
            num_pop1=1,
            input_percentage=100,
            input_weight="1",
        )
    elif "-net1" in sys.argv:
        sim, net = generate(
            "Net1",
            num_pop0=2,
            num_pop1=3,
            conn_weight="random()",
            input_percentage=100,
            input_weight="2*random()",
        )
    else:
        sim, net = generate("All")

    ################################################################################
    ###   Run in some simulators

    from neuromllite.NetworkGenerator import check_to_generate_or_run

    check_to_generate_or_run(sys.argv, sim)
