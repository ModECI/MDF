"""
    Example of ModECI MDF for NLP
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
    mod = Model(id="TestNLP")
    mod_graph = Graph(id="simple_nlp")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input_node")
    ip1 = InputPort(id="input_str")
    input_node.input_ports.append(ip1)
    input_node.parameters.append(Parameter(id="inner_str", value="input_str"))
    input_node.parameters.append(Parameter(id="mapping", value={1: "a", 2: "b"}))

    """
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)"""
    mod_graph.nodes.append(input_node)

    print(mod)

    print("------------------")
    print(mod.to_yaml())

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        # verbose = False
        from modeci_mdf.execution_engine import EvaluableGraph

        from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        eg = EvaluableGraph(mod_graph, verbose=verbose)
        eg.evaluate(array_format=format, initializer={"input_port1": 2})

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=1,
            filename_root=mod.id,
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="%s_3" % mod.id,
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
