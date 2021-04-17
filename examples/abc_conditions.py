"""
    Example of ModECI MDF - A simple 3 node graph with scheduling conditions
"""

import os

import abcd_python as abcd

from modeci_mdf.mdf import (
    Condition,
    ConditionSet,
    Function,
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

    input_node = Node(id="input0", parameters={"input_level": 0.0})
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    def create_simple_node(graph, id_, function, parameters, sender=None):
        n = Node(id=id_)
        graph.nodes.append(n)

        n.parameters = parameters
        ip1 = InputPort(id="input_port1", shape="(1,)")
        n.input_ports.append(ip1)

        function.args["variable0"] = ip1.id
        n.functions.append(function)

        n.output_ports.append(OutputPort(id="output_1", value=function.id))

        if sender is not None:
            simple_connect(sender, n, graph)

        return n

    p_a = {"slope": abcd.A_slope, "intercept": abcd.A_intercept}
    f_a = Function(id="linear_func", function="linear", args={k: k for k in p_a.keys()})
    a = create_simple_node(mod_graph, "A", f_a, p_a, input_node)

    p_b = {"gain": abcd.B_gain, "bias": abcd.B_bias, "offset": abcd.B_offset}
    f_b = Function(
        id="logistic_func", function="logistic", args={k: k for k in p_b.keys()}
    )
    b = create_simple_node(mod_graph, "B", f_b, p_b, a)

    p_c = {
        "scale": abcd.C_scale,
        "rate": abcd.C_rate,
        "bias": abcd.C_bias,
        "offset": abcd.C_offset,
    }
    f_c = Function(
        id="exponential_func", function="exponential", args={k: k for k in p_c.keys()}
    )
    c = create_simple_node(mod_graph, "C", f_c, p_c, a)

    cond_a = Condition(type="Always")
    cond_b = Condition(type="EveryNCalls", dependency=a.id, n=1)
    cond_c = Condition(type="EveryNCalls", dependency=a.id, n=3)
    cond_term = Condition(
        type="And",
        dependencies=[
            Condition(type="AfterNCalls", dependency=c.id, n=2),
            Condition(type="JustRan", dependency=a.id),
        ],
    )

    mod_graph.conditions = ConditionSet(
        node_specific={a.id: cond_a, b.id: cond_b, c.id: cond_c},
        termination={"trial": cond_term},
    )

    mod.to_json_file(os.path.join(os.path.dirname(__file__), "%s.json" % mod.id))
    mod.to_yaml_file(os.path.join(os.path.dirname(__file__), "%s.yaml" % mod.id))

    print_summary(mod_graph)


if __name__ == "__main__":
    main()
