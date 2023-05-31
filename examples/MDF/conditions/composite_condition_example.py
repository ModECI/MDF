"""
Example of ModECI MDF - A simple three Node graph satisfying the composite - 'All' condition
"""


from modeci_mdf.mdf import *
import graph_scheduler
import os
from modeci_mdf.utils import print_summary, simple_connect


def main():
    mod = Model(id="Composite_mdf_condition")
    mod_graph = Graph(id="Composite_mdf_condition_example")
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

    # create node A
    a = create_simple_node(mod_graph, "A", sender=None)
    a.parameters.append(Parameter(id="param_A", value="param_A + 1"))

    # create node B
    b = create_simple_node(mod_graph, "B", a)
    b.parameters.append(Parameter(id="param_B", value="param_B + 1"))

    # create node C
    c = create_simple_node(mod_graph, "C", b)
    c.parameters.append(Parameter(id="param_C", value="param_C + 1"))

    # set conditions
    new_node_cond = Condition(
        type="All",
        dependencies=[
            Condition(type="AfterCall", dependencies=b.id, n=2),
            Condition(type="AfterCall", dependencies=c.id, n=3),
        ],
    )
    mod_graph.conditions = ConditionSet(
        termination={"environment_state_update": new_node_cond}
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
            filename_root="composite_example",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
