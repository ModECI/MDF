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

    input_node = Node(id="input0", metadata={"color": ".8 .8 .8"})
    input_node.parameters.append(Parameter(id="input_level", value=0.0))
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    print(input_node)
    print(input_node.output_ports)

    # a = create_example_node('A', mod_graph)
    a = Node(id="A", metadata={"color": ".8 0 0"})
    mod_graph.nodes.append(a)
    ip1 = InputPort(id="input_port1")
    a.input_ports.append(ip1)

    a.parameters.append(Parameter(id="slope", value=abcd.A_slope))
    a.parameters.append(Parameter(id="intercept", value=abcd.A_intercept))

    f1 = Parameter(
        id="linear_func",
        function="linear",
        args={"variable0": ip1.id, "slope": "slope", "intercept": "intercept"},
    )
    a.parameters.append(f1)
    a.output_ports.append(OutputPort(id="output_1", value="linear_func"))

    e1 = simple_connect(input_node, a, mod_graph)

    b = Node(id="B", metadata={"color": "0 .8 0"})
    mod_graph.nodes.append(b)
    ip1 = InputPort(id="input_port1")
    b.input_ports.append(ip1)

    b.parameters.append(Parameter(id="gain", value=abcd.B_gain))
    b.parameters.append(Parameter(id="bias", value=abcd.B_bias))
    b.parameters.append(Parameter(id="offset", value=abcd.B_offset))

    f1 = Parameter(
        id="logistic_func",
        function="logistic",
        args={"variable0": ip1.id, "gain": "gain", "bias": "bias", "offset": "offset"},
    )
    b.parameters.append(f1)
    b.output_ports.append(OutputPort(id="output_1", value="logistic_func"))

    simple_connect(a, b, mod_graph)

    c = Node(id="C", metadata={"color": "0 0 .8"})
    mod_graph.nodes.append(c)
    ip1 = InputPort(id="input_port1", shape="(1,)")
    c.input_ports.append(ip1)

    c.parameters.append(Parameter(id="scale", value=abcd.C_scale))
    c.parameters.append(Parameter(id="rate", value=abcd.C_rate))
    c.parameters.append(Parameter(id="bias", value=abcd.C_bias))
    c.parameters.append(Parameter(id="offset", value=abcd.C_offset))

    f1 = Parameter(
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
    c.parameters.append(f1)
    c.output_ports.append(OutputPort(id="output_1", value="exponential_func"))

    simple_connect(b, c, mod_graph)

    d = Node(id="D", metadata={"color": ".8 0 .8"})
    mod_graph.nodes.append(d)

    ip1 = InputPort(id="input_port1", shape="(1,)")
    d.input_ports.append(ip1)
    d.parameters.append(Parameter(id="scale", value=abcd.D_scale))

    f1 = Parameter(
        id="sin_func", function="sin", args={"variable0": ip1.id, "scale": "scale"}
    )
    d.parameters.append(f1)
    d.output_ports.append(OutputPort(id="output_1", value="sin_func"))

    simple_connect(c, d, mod_graph)

    print(mod)

    print("------------------")
    # print(mod.to_json())
    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    print_summary(mod_graph)

    import sys

    if "-run" in sys.argv:
        verbose = True
        #verbose = False
        from modeci_mdf.execution_engine import EvaluableGraph
        from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        eg = EvaluableGraph(mod_graph, verbose=verbose)
        eg.evaluate(array_format=format)

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=1,
            filename_root="abcd",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="abcd_3",
            only_warn_on_fail=True  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
