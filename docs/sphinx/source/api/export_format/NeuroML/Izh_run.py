from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import Node, Edge, OutputPort, Parameter

from modeci_mdf.execution_engine import EvaluableGraph

from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW
import sys
import os
import numpy as np


def execute():

    mdf_model = load_mdf("IzhikevichTest.mdf.yaml")
    mod_graph = mdf_model.graphs[0]

    dt = 0.0005
    duration = 0.7

    input_node = mod_graph.get_node("InputList_stim")
    izh_node = mod_graph.get_node("izhPop")

    num_cells = 1

    if not "-iaf" in sys.argv:  # for testing...
        izh_node.get_parameter("v0").value = [-0.08] * num_cells
        izh_node.get_parameter("u").default_initial_value = [0.0] * num_cells
        izh_node.get_parameter("c").value = [
            izh_node.get_parameter("c").value[0]
        ] * num_cells

        input_node.get_parameter("i").conditions[0].value = [0] * num_cells
        input_node.get_parameter("amplitude").value = [
            (i + 1) * 1e-10 for i in range(num_cells)
        ]
        input_node.get_parameter("i").conditions[2].value = [0] * num_cells
    else:
        izh_node.get_parameter("v").default_initial_value = [-0.08]
        izh_node.get_parameter("reset").value = [-0.07]

    print(izh_node.to_dict())

    mdf_model.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=3,
        filename_root="Izh",
        only_warn_on_fail=(
            os.name == "nt"
        ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
    )

    verbose = "-v" in sys.argv

    eg = EvaluableGraph(mod_graph, verbose)
    # duration= 2
    t = 0

    times = []
    vv = {}
    uu = {}
    ii = {}

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

    while t < duration + dt:
        times.append(t)
        print("======   Evaluating at t = %s  ======" % (t))
        if t == 0:
            eg.evaluate(array_format=format)  # replace with initialize?
        else:
            eg.evaluate(array_format=format, time_increment=dt)

        for i in range(len(eg.enodes["izhPop"].evaluable_parameters["v"].curr_value)):
            if not i in vv:
                vv[i] = []
                uu[i] = []
                ii[i] = []
            v = eg.enodes["izhPop"].evaluable_parameters["v"].curr_value[i]
            u = eg.enodes["izhPop"].evaluable_parameters["u"].curr_value[i]
            vv[i].append(v)
            uu[i].append(u)

            ic = eg.enodes[input_node.id].evaluable_parameters["i"].curr_value[i]
            ii[i].append(ic)

        print(
            f"    Value at {t}: v={eg.enodes['izhPop'].evaluable_parameters['v'].curr_value }, \
            u={eg.enodes['izhPop'].evaluable_parameters['u'].curr_value},\
            i={eg.enodes[input_node.id].evaluable_parameters['i'].curr_value}"
        )
        t += dt

    import matplotlib.pyplot as plt

    scale = "1e8"

    for vi in vv:
        plt.plot(times, vv[vi], label="v cell %i" % vi)
        plt.plot(
            times,
            [x * float(scale) for x in uu[vi]],
            label="u cell %i (*%s)" % (vi, scale),
        )
        plt.plot(
            times,
            [x * float(scale) for x in ii[vi]],
            label="i cell %i (*%s)" % (vi, scale),
        )
    plt.legend()

    plt.savefig("Izh_run.png", bbox_inches="tight")

    if not "-nogui" in sys.argv:
        plt.show()


if __name__ == "__main__":

    execute()
