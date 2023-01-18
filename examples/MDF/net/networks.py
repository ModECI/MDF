"""
    Example of ModECI MDF - Testing state variables
"""

from modeci_mdf.mdf import *
import sys


def main(ref='acyclical'):

    mod = Model(id="Net_%s"%ref)

    mod_graph = Graph(id="Net_%s"%ref)
    mod.graphs.append(mod_graph)

    ## A node
    a_node = Node(id="A")

    p1 = Parameter(id="increment", value=1)
    a_node.parameters.append(p1)

    p2 = Parameter(id="count", value="count + increment")
    a_node.parameters.append(p2)

    op1 = OutputPort(id="out_port", value=p2.id)
    a_node.output_ports.append(op1)

    mod_graph.nodes.append(a_node)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        # verbose = False
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.01

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

            s.append(eg.enodes["A"].evaluable_outputs["out_port"].curr_value)
            t += dt

        if "-nogui" not in sys.argv:
            import matplotlib.pyplot as plt

            plt.plot(times, s)
            plt.show()

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root=mod_graph.id,
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

    return mod_graph


if __name__ == "__main__":
    main()
