"""
    Example of ModECI MDF - Testing state variables
"""

from modeci_mdf.mdf import *
from modeci_mdf.utils import simple_connect

import sys
import os


def create_increment_node(node_id, mod_graph, input_port=False):

    ## A node
    a_node = Node(id=node_id)

    if input_port:
        ip1 = InputPort(id="input_port")
        a_node.input_ports.append(ip1)

    p1 = Parameter(
        id="input_total", value="input_total+input_port" if input_port else 0
    )
    a_node.parameters.append(p1)

    p2 = Parameter(id="count", value="count + 1")
    a_node.parameters.append(p2)

    a_node.output_ports.append(OutputPort(id="op_count", value=p2.id))
    a_node.output_ports.append(OutputPort(id="op_in_tot", value=p1.id))

    mod_graph.nodes.append(a_node)

    return a_node


def main(ref="acyclical"):

    mod = Model(id="Net_%s" % ref)

    mod_graph = Graph(id="Net_%s" % ref)
    mod.graphs.append(mod_graph)

    a = create_increment_node("A", mod_graph)
    b = create_increment_node("B", mod_graph, input_port=True)

    simple_connect(a, b, mod_graph)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

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
        s = []
        while t <= duration:
            times.append(t)
            print("======   Evaluating at t = %s  ======" % (t))
            if t == 0:
                eg.evaluate()  # replace with initialize?
            else:
                eg.evaluate(time_increment=dt)

            s.append(eg.enodes["A"].evaluable_outputs["op_count"].curr_value)
            t += dt

        for n in eg.enodes:
            print("Final state of node: %s" % str(eg.enodes[n].get_output()))

        """
        if "-nogui" not in sys.argv:
            import matplotlib.pyplot as plt

            plt.plot(times, s)
            plt.show()"""

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

    return mod_graph


if __name__ == "__main__":
    main()
