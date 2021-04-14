"""
    Example of ModECI MDF - Testing arrays as inputs
"""

from modeci_mdf.mdf import *
from modeci_mdf.simple_scheduler import EvaluableGraph
import numpy as np
import sys


def generate_test_model(
    id,
    input_shape=(2, 2),
    hidden_shape=(2, 2),
    hidden_layers=2,
    output_shape=(2, 2),
    save_to_file=True,
    seed=1234,
):
    np.random.seed(seed)
    mod = Model(id=id)
    mod_graph = Graph(id=id)
    mod.graphs.append(mod_graph)

    input_node = Node(
        id="input_node",
        parameters={"input_level": np.random.random_sample(input_shape).tolist()},
    )

    op1 = OutputPort(id="out_port", value="input_level")
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    middle_node = Node(
        id="hidden_node",
        parameters={"slope": 0.5, "intercept": np.random.random_sample(hidden_shape).tolist()},
    )

    ip1 = InputPort(id="input_port1")
    middle_node.input_ports.append(ip1)
    mod_graph.nodes.append(middle_node)

    f1 = Function(
        id="linear_1",
        function="linear",
        args={"variable0": ip1.id, "slope": "slope", "intercept": "intercept"},
    )
    middle_node.functions.append(f1)

    middle_node.output_ports.append(OutputPort(id="output_1", value="linear_1"))

    e1 = Edge(
        id="input_edge",
        parameters={"weight": np.random.random_sample(input_shape).tolist()},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=middle_node.id,
        receiver_port=ip1.id,
    )

    mod_graph.edges.append(e1)

    if save_to_file:
        new_file = mod.to_json_file("%s.json" % mod.id)
        new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    return mod_graph


if __name__ == "__main__":
    mod_graph = generate_test_model("small_test")

    scale = 20
    mod_graph = generate_test_model(
        "medium_test", input_shape=(scale, scale), hidden_shape=(scale, scale)
    )

    if "-run" in sys.argv:

        from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        print("------------------")
        eg = EvaluableGraph(mod_graph, verbose=False)
        eg.evaluate(array_format=format)

        print('Finished evaluating graph using array format %s'%format)
