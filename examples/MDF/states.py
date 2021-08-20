"""
    Example of ModECI MDF - Testing state variables
"""

from modeci_mdf.mdf import *
import sys


def main():
    mod = Model(id="States")
    mod_graph = Graph(id="state_example")
    mod.graphs.append(mod_graph)

    ## Counter node
    counter_node = Node(id="counter_node")

    p1 = Parameter(id="increment", value=1)
    counter_node.parameters.append(p1)

    p2 = Parameter(id="count", value="count + increment")
    counter_node.parameters.append(p2)

    op1 = OutputPort(id="out_port", value=p2.id)
    counter_node.output_ports.append(op1)

    mod_graph.nodes.append(counter_node)

    ## Sine node...
    sine_node = Node(id="sine_node")

    sine_node.parameters.append(Parameter(id="amp", value=3))
    sine_node.parameters.append(Parameter(id="period", value=0.4))

    s1 = Parameter(id="level", default_initial_value=0, time_derivative='6.283185 * rate / period')
    sine_node.parameters.append(s1)

    s2 = Parameter(id="rate", default_initial_value=1, time_derivative='-1 * 6.283185 * level / period')
    sine_node.parameters.append(s2)

    op1 = OutputPort(id="out_port", value="amp * level")
    sine_node.output_ports.append(op1)

    mod_graph.nodes.append(sine_node)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        #verbose = False
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.01

        duration= 2
        t = 0
        recorded = {}
        times = []
        s = []
        while t<=duration:
            times.append(t)
            print("======   Evaluating at t = %s  ======"%(t))
            if t == 0:
                eg.evaluate() # replace with initialize?
            else:
                eg.evaluate(time_increment=dt)

            s.append(eg.enodes['sine_node'].evaluable_outputs['out_port'].curr_value)
            t+=dt


        if "-nogui" not in sys.argv:
            import matplotlib.pyplot as plt
            plt.plot(times,s)
            plt.show()


    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="states",
            only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


    return mod_graph


if __name__ == "__main__":
    main()
