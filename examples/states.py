"""
    Example of ModECI MDF - Testing state variables
"""

from modeci_mdf.mdf import *
import sys

def main():
    mod = Model(id="States")
    mod_graph = Graph(id="state_example")
    mod.graphs.append(mod_graph)

    sine_node = Node(id="sine_node", parameters={"amp": 3})

    s1 = State(id="level", default_initial_value=2)
    sine_node.states.append(s1)

    op1 = OutputPort(id="out_port", value="level")
    sine_node.output_ports.append(op1)

    mod_graph.nodes.append(sine_node)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)


    if '-run' in sys.argv:
        verbose = True
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.simple_scheduler import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        eg.evaluate()

    return mod_graph


if __name__ == "__main__":
    main()
