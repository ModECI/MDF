"""
    Example of ModECI MDF - Testing arrays as inputs
"""

from modeci_mdf.mdf import *
import sys
import numpy as np


def main():
    mod = Model(id="Arrays")
    mod_graph = Graph(id="array_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input_node")

    input_node.parameters.append(Parameter(id="input_level", value=[[1, 2.], [3, 4]]))

    op1 = OutputPort(id="out_port", value="input_level")
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    middle_node = Node(id="middle_node")
    middle_node.parameters.append(Parameter(id="slope", value=0.5))
    middle_node.parameters.append(Parameter(id="intercept", value=np.array([[0, 1.], [2, 2]])))

    ip1 = InputPort(id="input_port1")
    middle_node.input_ports.append(ip1)
    mod_graph.nodes.append(middle_node)

    f1 = Parameter(
        id="linear_1",
        function="linear",
        args={"variable0": ip1.id, "slope": "slope", "intercept": "intercept"},
    )
    middle_node.parameters.append(f1)

    middle_node.output_ports.append(OutputPort(id="output_1", value="linear_1"))

    e1 = Edge(
        id="input_edge",
        parameters={"weight": [[1, 0], [0, 1]]},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=middle_node.id,
        receiver_port=ip1.id,
    )

    mod_graph.edges.append(e1)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        #verbose = False
        from modeci_mdf.execution_engine import EvaluableGraph

        from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        eg = EvaluableGraph(mod_graph, verbose=True)
        eg.evaluate(array_format=format)

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="arrays",
            only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

if __name__ == "__main__":
    main()
