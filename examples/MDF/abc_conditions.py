"""
    Example of ModECI MDF - A simple 3 node graph with scheduling conditions
"""

import os

import abcd_python as abcd
import graph_scheduler
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

    def create_simple_node(graph, id_, sender=None):
        n = Node(id=id_)
        graph.nodes.append(n)

        ip1 = InputPort(id="input_port1", shape="(1,)")
        n.input_ports.append(ip1)

        n.output_ports.append(OutputPort(id="output_1", value=ip1.id))

        if sender is not None:
            simple_connect(sender, n, graph)

        return n

    a = create_simple_node(mod_graph, "A", sender=None)
    a.parameters.append(Parameter(id="count_A", value="count_A + 1"))

    b = create_simple_node(mod_graph, "B", a)
    b.parameters.append(Parameter(id="count_B", value="count_B + 1"))

    c = create_simple_node(mod_graph, "C", a)

    c.parameters.append(Parameter(id="count_C", value="count_C+ 1"))

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
        node_specific={a.id: cond_a, b.id: cond_b, c.id: cond_c},
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
        # evaluating the current state of the graph's parameters
        print(
            "\n Output of A: %s"
            % eg.enodes["A"].evaluable_outputs["output_1"].curr_value
        )
        print(
            "\n Output of B: %s"
            % eg.enodes["B"].evaluable_outputs["output_1"].curr_value
        )
        print(
            "\n Output of C: %s"
            % eg.enodes["C"].evaluable_outputs["output_1"].curr_value
        )
    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="abc_conditions",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
