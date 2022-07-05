"""
Example of ModECI MDF- A simple 3 Node graph satisfying the EveryNCalls Condition

"""

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
    mod = Model(id="everyncalls_condition")
    mod_graph = Graph(id="everyncalls_example")
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
    a.parameters.append(Parameter(id="param_A", value="param_A + 1"))

    b = create_simple_node(mod_graph, "B", a)
    b.parameters.append(Parameter(id="param_B", value="param_B + 1"))

    c = create_simple_node(mod_graph, "C", b)

    c.parameters.append(Parameter(id="param_C", value="param_C+ 1"))

    cond_a = Condition(type="Always")  # A is always executed
    cond_b = Condition(
        type="EveryNCalls", dependencies=a.id, n=2
    )  # B executes every 2 calls of A
    cond_c = Condition(
        type="EveryNCalls", dependencies=b.id, n=3
    )  # C executes every 3 calls of B

    mod_graph.conditions = ConditionSet(
        node_specific={a.id: cond_a, b.id: cond_b, c.id: cond_c},
        # implicit AllHaveRun Termination Condition
        # You may explicitly have your own termination condition as in abc_conditions.py
    )
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
