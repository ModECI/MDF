"""
    Example of ModECI MDF - A simple 4 node graph
"""

from modeci_mdf.mdf import *

from modeci_mdf.utils import simple_connect, print_summary

import abcd_python as abcd


def main():
    mod = Model(id="ABCD")
    mod_graph = Graph(id="abcd_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0", parameters={"input_level": 0.0})
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    print(input_node)
    print(input_node.output_ports)

    # a = create_example_node('A', mod_graph)
    a = Node(id="A")
    mod_graph.nodes.append(a)

    a.parameters = {"slope": abcd.A_slope, "intercept": abcd.A_intercept}
    ip1 = InputPort(id="input_port1", shape="(1,)")
    a.input_ports.append(ip1)

    f1 = Function(
        id="linear_func",
        function="linear",
        args={"variable0": ip1.id, "slope": "slope", "intercept": "intercept"},
    )
    a.functions.append(f1)
    a.output_ports.append(OutputPort(id="output_1", value="linear_func"))

    e1 = simple_connect(input_node, a, mod_graph)

    b = Node(id="B")
    mod_graph.nodes.append(b)

    b.parameters = {"gain": abcd.B_gain, "bias": abcd.B_bias, "offset": abcd.B_offset}
    ip1 = InputPort(id="input_port1", shape="(1,)")
    b.input_ports.append(ip1)

    f1 = Function(
        id="logistic_func",
        function="logistic",
        args={"variable0": ip1.id, "gain": "gain", "bias": "bias", "offset": "offset"},
    )
    b.functions.append(f1)
    b.output_ports.append(OutputPort(id="output_1", value="logistic_func"))

    simple_connect(a, b, mod_graph)

    c = Node(id="C")
    mod_graph.nodes.append(c)

    c.parameters = {
        "scale": abcd.C_scale,
        "rate": abcd.C_rate,
        "bias": abcd.C_bias,
        "offset": abcd.C_offset,
    }
    ip1 = InputPort(id="input_port1", shape="(1,)")
    c.input_ports.append(ip1)

    f1 = Function(
        id="exponential_func",
        function="exponential",
        args={
            "variable0": ip1.id,
            "scale": "scale",
            "rate": "rate",
            "bias": "bias",
            "offset": "offset",
        },
    )
    c.functions.append(f1)
    c.output_ports.append(OutputPort(id="output_1", value="exponential_func"))

    simple_connect(b, c, mod_graph)

    d = Node(id="D")
    mod_graph.nodes.append(d)

    d.parameters = {"scale": abcd.D_scale}
    ip1 = InputPort(id="input_port1", shape="(1,)")
    d.input_ports.append(ip1)

    f1 = Function(
        id="sin_func", function="sin", args={"variable0": ip1.id, "scale": "scale"}
    )
    d.functions.append(f1)
    d.output_ports.append(OutputPort(id="output_1", value="sin_func"))

    simple_connect(c, d, mod_graph)

    print(mod)

    print("------------------")
    # print(mod.to_json())
    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    print_summary(mod_graph)


if __name__ == "__main__":
    main()
