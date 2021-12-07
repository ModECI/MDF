from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import Node, Edge, OutputPort, Parameter

from modeci_mdf.execution_engine import EvaluableGraph

from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW
import sys
import numpy as np

verbose = True
verbose = False


def execute(multi=False):

    mdf_model = load_mdf("IzhikevichTest.mdf.yaml")
    mod_graph = mdf_model.graphs[0]

    dt = 0.001
    duration = 0.03

    if not multi:

        izh_node = mod_graph.nodes[0]

        if not "-iaf" in sys.argv:  # for testing...
            izh_node.get_parameter("v0").value = [-0.08]
            izh_node.get_parameter("u").default_initial_value = [0.0]
        else:
            izh_node.get_parameter("v").default_initial_value = [-0.08]
            izh_node.get_parameter("reset").value = [-0.07]
        input = np.array([0])

        mdf_model.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="Izh",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

    else:

        izh_node = mod_graph.nodes[0]

        size = 1
        max_amp = 5e-10
        input = np.array([max_amp * (-1 + 2.0 * i / size) for i in range(size)])
        input = [1.0e-10]
        input_node = Node(id="input_node")
        input_node.parameters.append(Parameter(id="input_level", value=input))

        op1 = OutputPort(id="out_port", value="input_level")
        input_node.output_ports.append(op1)
        mod_graph.nodes.append(input_node)

        izh_node.get_parameter("v0").value = np.array([-0.07] * len(input))
        izh_node.get_parameter("c").value = np.array([-0.05] * len(input))
        izh_node.get_parameter("u").default_initial_value = np.array([0.0] * len(input))

        print(izh_node)

        e1 = Edge(
            id="input_edge",
            sender=input_node.id,
            sender_port=op1.id,
            receiver="izhPop_0",
            receiver_port="synapses_i",
        )

        mod_graph.edges.append(e1)

        mdf_model.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="Izh_multi",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

        duration = 0.5

    eg = EvaluableGraph(mod_graph, verbose)
    # duration= 2
    t = 0

    times = []
    vv = {}
    uu = {}

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

    while t < duration + dt:
        times.append(t)
        print("======   Evaluating at t = %s  ======" % (t))
        if t == 0:
            eg.evaluate(array_format=format)  # replace with initialize?
        else:
            eg.evaluate(array_format=format, time_increment=dt)

        for i in range(len(eg.enodes["izhPop_0"].evaluable_parameters["v"].curr_value)):
            if not i in vv:
                vv[i] = []
                uu[i] = []
            v = eg.enodes["izhPop_0"].evaluable_parameters["v"].curr_value[i]
            u = eg.enodes["izhPop_0"].evaluable_parameters["u"].curr_value[i]
            u = 0
            vv[i].append(v)
            uu[i].append(u)
            if i == 0:
                print(f"    Value at {t}: v={v}, u={u}")
        t += dt

    import matplotlib.pyplot as plt

    for vi in vv:
        plt.plot(times, vv[vi], label="v %.3f" % input[vi])
        plt.plot(times, uu[vi], label="u %.3f" % input[vi])
    plt.legend()

    if not multi:
        plt.savefig("Izh_run.png", bbox_inches="tight")

    if not "-nogui" in sys.argv:
        plt.show()


if __name__ == "__main__":

    execute("-multi" in sys.argv)
