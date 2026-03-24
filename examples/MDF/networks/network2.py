"""
    Example of ModECI MDF - Testing networks
"""

from modeci_mdf.mdf import *
import sys
import os
from modeci_mdf.utils import simple_connect


def main(ref="network2"):

    mod = Model(id="Net_%s" % ref)

    mod_graph = Graph(id="Net_%s" % ref)
    mod.graphs.append(mod_graph)

    for id in ["A", "B"]:

        node = Node(id=id)

        if id == "A":
            ip1 = InputPort(id="input", default_value=5)
            node.input_ports.append(ip1)

        if id == "B":
            ip1 = InputPort(id="input")
            node.input_ports.append(ip1)

        p1 = Parameter(id="total_inputs", value="total_inputs + input")
        node.parameters.append(p1)

        p2 = Parameter(id="execute_count", value="execute_count + 1")
        node.parameters.append(p2)

        op1 = OutputPort(id="out_port", value=p1.id)
        node.output_ports.append(op1)

        mod_graph.nodes.append(node)

    # simple_connect(mod_graph.get_node("A"), mod_graph.get_node("B"), mod_graph)
    # simple_connect(mod_graph.get_node("B"), mod_graph.get_node("A"), mod_graph)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    eg = None

    if "-run" in sys.argv:
        verbose = True
        # verbose = False
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 1

        duration = 2
        t = 0
        recorded = {}
        times = []
        ai = []
        bi = []
        at = []
        bt = []
        ao = []
        bo = []
        while t <= duration:
            times.append(t)
            print("======   Evaluating at t = %s  ======" % (t))
            if t == 0:
                eg.evaluate()  # replace with initialize?
            else:
                eg.evaluate(time_increment=dt)

            ai.append(eg.enodes["A"].evaluable_parameters["execute_count"].curr_value)
            bi.append(eg.enodes["B"].evaluable_parameters["execute_count"].curr_value)
            at.append(eg.enodes["A"].evaluable_parameters["total_inputs"].curr_value)
            bt.append(eg.enodes["B"].evaluable_parameters["total_inputs"].curr_value)
            ao.append(eg.enodes["A"].evaluable_outputs["out_port"].curr_value)
            bo.append(eg.enodes["B"].evaluable_outputs["out_port"].curr_value)
            t += dt

        if "-nogui" not in sys.argv:
            import matplotlib.pyplot as plt

            plt.plot(times, ai, marker="x", label="A exe count")
            plt.plot(times, bi, label="B exe count")
            plt.plot(times, at, marker="D", label="A total_inputs")
            plt.plot(times, bt, label="B total_inputs")

            plt.legend()
            plt.show()

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root=mod_graph.id,
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

    return mod_graph, eg


if __name__ == "__main__":
    g, eg = main()

    print("--------------------")
    print(g.to_yaml())
    print("--------------------")
    if eg is not None:
        for n in eg.enodes:
            enode = eg.enodes[n]
            print(enode)
            print([enode.evaluable_outputs for enode in eg.enodes.values()])
    else:
        print("No execution engine created. Run with: -run")
