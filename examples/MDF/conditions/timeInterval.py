"""
    Example of ModECI MDF - A simple Graph satisfying a time based condition
"""
import sys
import os
import graph_scheduler
from modeci_mdf.mdf import *
from modeci_mdf.utils import print_summary, simple_connect


def main():
    mod = Model(id="timeinterval_condition")
    mod_graph = Graph(id="timeinterval_example")
    mod.graphs.append(mod_graph)

    def create_simple_node(graph, id_, sender=None):
        n = Node(id=id_)
        graph.nodes.append(n)

        ip1 = InputPort(id="input_port1", shape="(1,)")
        n.input_ports.append(ip1)

        n.output_ports.append(OutputPort(id="output_1", value="0"))

        if sender is not None:
            simple_connect(sender, n, graph)

        return n

    # node A
    a = create_simple_node(mod_graph, "A", sender=None)
    a.parameters.append(Parameter(id="param_A", value="param_A + 1"))
    a.get_output_port("output_1").value = "param_A"

    # node B
    b = create_simple_node(mod_graph, "B", sender=a)
    b.parameters.append(Parameter(id="param_B", value="param_B + 1"))
    b.get_output_port("output_1").value = "param_B"

    # node C
    c = create_simple_node(mod_graph, "C", sender=b)
    c.parameters.append(Parameter(id="param_C", value="param_C + 1"))
    c.get_output_port("output_1").value = "param_C"

    # See documentation: https://kmantel.github.io/graph-scheduler/Condition.html#graph_scheduler.condition.TimeInterval for more arguments you can add to the Time Interval condition

    cond_a = Condition(type="Always")
    cond_b = Condition(type="AfterPass", n=1)
    cond_c = Condition(
        type="AfterPass", n=4
    )  #  AfterEnvironmentStateUpdate, AfterPass, AfterNEnvironmentSequences

    mod_graph.conditions = ConditionSet(
        node_specific={a.id: cond_a, b.id: cond_b, c.id: cond_c},
    )
    mod.to_json_file(os.path.join(os.path.dirname(__file__), "%s.json" % mod.id))
    mod.to_yaml_file(os.path.join(os.path.dirname(__file__), "%s.yaml" % mod.id))
    print_summary(mod_graph)

    if "-run" in sys.argv:
        verbose = True
        # verbose = False
        from modeci_mdf.execution_engine import EvaluableGraph
        from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        eg = EvaluableGraph(mod_graph, verbose=verbose)

        # Evaluate once, the conditions set how many times each node executes
        eg.evaluate(array_format=format)

        for n in ["A", "B", "C"]:
            print(
                "\n Final value of node %s: %s"
                % (n, eg.enodes[n].evaluable_outputs["output_1"].curr_value)
            )

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="timeinterval",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )
    return mod_graph


if __name__ == "__main__":
    main()
