"""
    Example of ModECI MDF - A simple 3 node graph with scheduling conditions
"""

import os

import abcd_python as abcd

from modeci_mdf.mdf import (
    Condition,
    ConditionSet,
    Parameter,
    Graph,
    InputPort,
    Model,
    Node,
    OutputPort,
)
from modeci_mdf.utils import print_summary, simple_connect


def main():
    mod = Model(id="abc_conditions")
    mod_graph = Graph(id="abc_conditions_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0")
    input_node.parameters.append(Parameter(id="input_level", value=0.0))
    input_node.parameters.append(Parameter(id="count_0", value="count_0 + 1"))
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    def create_simple_node(graph, id_, sender=None):
        n = Node(id=id_)
        graph.nodes.append(n)

        ip1 = InputPort(id="input_port1", shape="(1,)")
        n.input_ports.append(ip1)

        n.output_ports.append(OutputPort(id="output_1", value=ip1.id))

        if sender is not None:
            simple_connect(sender, n, graph)

        return n

    a = create_simple_node(mod_graph, "A", input_node)
    a.parameters.append(Parameter(id="count_A", value="count_A + 1"))

    b = create_simple_node(mod_graph, "B", a)
    b.parameters.append(Parameter(id="count_B", value="count_B + 1"))

    c = create_simple_node(mod_graph, "C", a)

    c.parameters.append(Parameter(id="count_C", value="count_C+ 1"))

    cond_i = Condition(type="BeforeNCalls", dependencies=input_node.id, n=1)
    cond_a = Condition(type="Always")
    cond_b = Condition(type="EveryNCalls", dependencies=a.id, n=2)
    cond_c = Condition(type="EveryNCalls", dependencies=a.id, n=3)
    cond_term = Condition(
        type="And",
        dependencies=[
            Condition(type="AfterNCalls", dependencies=c.id, n=2),
            Condition(type="JustRan", dependencies=a.id),
        ],
    )

    mod_graph.conditions = ConditionSet(
        node_specific={input_node.id: cond_i, a.id: cond_a, b.id: cond_b, c.id: cond_c},
        termination={"environment_state_update": cond_term},
    )

    mod.to_json_file(os.path.join(os.path.dirname(__file__), "%s.json" % mod.id))
    mod.to_yaml_file(os.path.join(os.path.dirname(__file__), "%s.yaml" % mod.id))

    print_summary(mod_graph)

    import sys

    if "-run" in sys.argv:
        verbose = True
        # verbose = False
        from modeci_mdf.execution_engine import EvaluableGraph
        from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        eg = EvaluableGraph(mod_graph, verbose=verbose)
        eg.evaluate(array_format=format)

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="abc_conditions",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
