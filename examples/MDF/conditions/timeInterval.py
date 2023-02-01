"""
    Example of ModECI MDF - A simple 2 Node Graph satisfying the Time Interval condition
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

        n.output_ports.append(OutputPort(id="output_1", value=ip1.id))

        if sender is not None:
            simple_connect(sender, n, graph)

        return n

    # node A
    a = create_simple_node(mod_graph, "A", sender=None)
    a.parameters.append(Parameter(id="param_A", value="param_A + 1"))
    # node B
    b = create_simple_node(mod_graph, "B", sender=a)
    b.parameters.append(Parameter(id="param_B", value="param_B + 1"))
    # node C
    c = create_simple_node(mod_graph, "C", sender=b)
    c.parameters.append(Parameter(id="param_C", value="param_C + 1"))

    # See documentation: https://kmantel.github.io/graph-scheduler/Condition.html#graph_scheduler.condition.TimeInterval for more arguments you can add to the Time Interval condition

    # A will always run, B starts executing after 5ms while B after 10ms
    cond_a = Condition(type="Always")
    cond_b = Condition(type="TimeInterval", start=5)
    cond_c = Condition(type="TimeInterval", start=10)
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

        # Using a time series to execute the graph 5 times
        dt = 1
        duration = 5
        t = 0
        times = []
        while t <= duration:
            times.append(t)
            print("===== Evaluating at t = %s  ======" % (t))
            if t == 0:
                eg.evaluate(array_format=format)
            else:
                eg.evaluate(time_increment=dt)
            t += dt
    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="timeinterval",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )
    return mod_graph


if __name__ == "__main__":
    main()
