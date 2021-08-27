"""
    Example of ModECI MDF - A simple 2 node graph.
"""

from modeci_mdf.mdf import (
    Model,
    Graph,
    Node,
    Parameter,
    Function,
    InputPort,
    OutputPort,
    Edge,
)

import sys

def main():
    mod = Model(id="Simple")
    mod_graph = Graph(id="simple_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input_node")
    input_node.parameters.append(Parameter(id="input_level", value=0.5))
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    processing_node = Node(id="processing_node")
    mod_graph.nodes.append(processing_node)

    processing_node.parameters.append(Parameter(id="lin_slope", value=0.5))
    processing_node.parameters.append(Parameter(id="lin_intercept", value=0))
    processing_node.parameters.append(Parameter(id="log_gain", value=3))
    ip1 = InputPort(id="input_port1")
    processing_node.input_ports.append(ip1)

    f1 = Parameter(
        id="linear_1",
        function="linear",
        args={"variable0": ip1.id, "slope": "lin_slope", "intercept": "lin_intercept"},
    )
    f2 = Parameter(
        id="logistic_1",
        function="logistic",
        args={"variable0": f1.id, "gain": "log_gain", "bias": 0, "offset": 0},
    )
    processing_node.parameters.append(f1)
    processing_node.parameters.append(f2)

    processing_node.output_ports.append(OutputPort(id="output_1", value="logistic_1"))

    e1 = Edge(
        id="input_edge",
        parameters={"weight": 0.55},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=processing_node.id,
        receiver_port=ip1.id,
    )

    mod_graph.edges.append(e1)

    print(mod)

    print("------------------")
    print(mod.to_json())

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        #verbose = False
        from modeci_mdf.execution_engine import EvaluableGraph

        from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        eg = EvaluableGraph(mod_graph, verbose=verbose)
        eg.evaluate(array_format=format)

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=1,
            filename_root="simple",
            only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="simple_3",
            only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
