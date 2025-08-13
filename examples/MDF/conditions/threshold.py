"""
Example of ModECI MDF- A simple 1 Node graph satisfying the Threshold Condition

"""

import graph_scheduler
import os
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
    mod = Model(id="threshold_condition")
    mod_graph = Graph(id="threshold_example")
    mod.graphs.append(mod_graph)

    def create_simple_node(graph, id_, sender=None):
        n = Node(id=id_)
        graph.nodes.append(n)

        ip1 = InputPort(id="input_port1", shape="(1,)")
        n.input_ports.append(ip1)

        n.output_ports.append(OutputPort(id="output_1", value="param_A"))

        if sender is not None:
            simple_connect(sender, n, graph)

        return n

    a = create_simple_node(mod_graph, "A", sender=None)
    a.parameters.append(
        Parameter(id="param_A", value="param_A + 1", default_initial_value=0)
    )
    cond_term = Condition(
        type="Threshold",
        dependency=a,
        parameter="param_A",
        threshold=5,
        comparator=">=",
    )
    # The threshold condition is satisfied when the comparison btwn the value of the parameter
    # and threshold using the comparator evaluates to true
    mod_graph.conditions = ConditionSet(
        termination={
            "environment_state_update": cond_term
        },  # the graph terminates when the parameter >= 5 (Executing 5 times in this example)
    )
    mod.to_json_file(os.path.join(os.path.dirname(__file__), "%s.json" % mod.id))
    mod.to_yaml_file(os.path.join(os.path.dirname(__file__), "%s.yaml" % mod.id))
    # print_summary(mod_graph)
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
    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="threshold",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
