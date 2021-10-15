"""
    Useful utility functions for dealing with MDF objects.
"""

from modeci_mdf.mdf import Model, Graph, Node, Edge, OutputPort, Function, InputPort


def create_example_node(node_id: str, graph: Graph) -> Node:
    """
    Create a simple example node with Input inside a graph

    Args:
        node_id: The unique id for the first node in the graph.
        graph: The graph to add the example node.

    Returns:
        The node (with id=node_id) created in the graph.
    """

    a = Node(id=node_id)
    graph.nodes.append(a)

    a.parameters = {"logistic_gain": 3, "slope": 0.5, "intercept": 0}
    ip1 = InputPort(id="input_port1", shape="(1,)")
    a.input_ports.append(ip1)

    f1 = Function(
        id="logistic_1",
        function="logistic",
        args={"variable0": ip1.id, "gain": "logistic_gain", "bias": 0, "offset": 0},
    )
    a.functions.append(f1)
    f2 = Function(
        id="linear_1",
        function="linear",
        args={"variable0": f1.id, "slope": "slope", "intercept": "intercept"},
    )
    a.functions.append(f2)
    a.output_ports.append(OutputPort(id="output_1", value="linear_1"))

    return a


def simple_connect(pre_node, post_node, graph) -> Edge:
    """
    Create an edge connecting two nodes in a graph.

    Args:
        pre_node: The source node.
        post_node: The destination node.
        graph: The graph to and the edge.

    Returns:
        The edge that has been added to the graph.
    """

    e1 = Edge(
        id=f"edge_{pre_node.id}_{post_node.id}",
        sender=pre_node.id,
        sender_port=pre_node.output_ports[0].id,
        receiver=post_node.id,
        receiver_port=post_node.input_ports[0].id,
    )

    graph.edges.append(e1)
    return e1


def print_summary(graph: Graph):
    """Print a summary of a graph to standard out."""
    print(
        "Graph %s with %i nodes and %s edges\n"
        % (graph.id, len(graph.nodes), len(graph.edges))
    )
    for node in graph.nodes:
        print("%s" % node)
    for edge in graph.edges:
        print("%s" % edge)


def load_mdf(filename: str) -> Model:
    """
    Load an MDF file from JSON or YAML. File type is detected automatically based on extension.
    """

    if filename.endswith("yaml") or filename.endswith("yml"):
        return load_mdf_yaml(filename)
    else:
        return load_mdf_json(filename)


def load_mdf_json(filename: str) -> Model:
    """
    Load an MDF JSON file
    """

    from neuromllite.utils import load_json, _parse_element

    data = load_json(filename)

    print(f"Loaded a graph from {filename}, Root(s): {data.keys()}")
    if data.keys() == "graphs":
        data = {"UNSPECIFIED": data}
    model = Model()
    model = _parse_element(data, model)

    return model


def load_mdf_yaml(filename: str) -> Model:
    """
    Load an MDF YAML file
    """

    from neuromllite.utils import load_yaml, _parse_element

    data = load_yaml(filename)

    print(f"Loaded a graph from {filename}, Root(s): {data.keys()}")
    if data.keys() == "graphs":
        data = {"UNSPECIFIED": data}
    model = Model()
    model = _parse_element(data, model)

    return model


def color_rgb_to_hex(rgb):
    """Convert a rgb color to hexadecimal format."""
    color = "#"
    print("Converting %s to hex color" % rgb)
    for a in rgb.split():
        color = color + "%02x" % int(float(a) * 255)
    return color


def is_number(s):
    """Return :code:`True` if cast to :code:`float` does not throw ValueError, :code:`False` otherwise. """
    try:
        float(s)
        return True
    except ValueError:
        return False
