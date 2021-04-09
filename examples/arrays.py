"""
    Example of ModECI MDF - Testing arrays as inputs
"""

from modeci_mdf.mdf import *


def main():
    mod = Model(id="Arrays")
    mod_graph = Graph(id="array_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input_node", parameters={"input_level": [[1,2],[3,4]]})
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    middle_node = Node(id="middle_node")
    mod_graph.nodes.append(middle_node)

    middle_node.parameters = {"slope": 0.5, "intercept": [[0,1],[2,2]]}

    ip1 = InputPort(id="input_port1")
    middle_node.input_ports.append(ip1)

    f1 = Function(
        id="linear_1",
        function="linear",
        args={"variable0": ip1.id, "slope": "slope", "intercept": "intercept"},
    )
    middle_node.functions.append(f1)
    middle_node.output_ports.append(OutputPort(id="output_1", value="linear_1"))

    e1 = Edge(
        id="input_edge",
        parameters={"weight": [[1,0],[0,1]]},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=middle_node.id,
        receiver_port=ip1.id
    )

    mod_graph.edges.append(e1)

    print(mod)

    print("------------------")
    print(mod.to_json())

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)


if __name__ == "__main__":
    main()
