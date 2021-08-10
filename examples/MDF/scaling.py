"""
    Example of ModECI MDF - Testing arrays as inputs - this allows graphs of
    various scales/sizes to be generated for testing purposes
"""

from modeci_mdf.mdf import *
from modeci_mdf.execution_engine import EvaluableGraph
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
        id="input_node"
    )

    input_node.parameters.append(Parameter(id="input_level", value=np.random.random_sample(input_shape).tolist()))

    input_node.output_ports.append(OutputPort(id="out_port", value="input_level"))

    mod_graph.nodes.append(input_node)

    last_node = input_node

    for i in range(hidden_layers):

        hidden_node = Node(
            id="hidden_node_%i" % i
        )

        hidden_node.input_ports.append(InputPort(id="in_port"))
        hidden_node.parameters.append(Parameter(id="slope0", value=0.5))
        hidden_node.parameters.append(Parameter(id="intercept0", value=np.random.random_sample(hidden_shape).tolist()))

        mod_graph.nodes.append(hidden_node)

        f1 = Parameter(
            id="linear_1",
            function="linear",
            args={
                "variable0": hidden_node.input_ports[0].id,
                "slope": "slope0",
                "intercept": "intercept0",
            },
        )
        hidden_node.parameters.append(f1)

        hidden_node.output_ports.append(OutputPort(id="out_port", value="linear_1"))

        e1 = Edge(
            id="edge_%i" % i,
            parameters={"weight": np.random.random_sample(input_shape).tolist()},
            sender=last_node.id,
            sender_port=last_node.output_ports[0].id,
            receiver=hidden_node.id,
            receiver_port=hidden_node.input_ports[0].id,
        )

        mod_graph.edges.append(e1)

        last_node = hidden_node

    output_node = Node(
        id="output_node",
    )

    output_node.input_ports.append(InputPort(id="in_port"))
    output_node.output_ports.append(OutputPort(id="out_port", value="in_port"))
    mod_graph.nodes.append(output_node)

    e1 = Edge(
        id="edge_%i" % (i + 1),
        parameters={"weight": np.random.random_sample(input_shape).tolist()},
        sender=last_node.id,
        sender_port=last_node.output_ports[0].id,
        receiver=output_node.id,
        receiver_port=output_node.input_ports[0].id,
    )

    mod_graph.edges.append(e1)

    if save_to_file:
        new_file = mod.to_json_file("%s.json" % mod.id)
        new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    return mod_graph


def main():
    mod_graph = generate_test_model("small_test", save_to_file=True)

    scale = 2
    mod_graph = generate_test_model(
        "medium_test",
        input_shape=(scale, scale),
        hidden_shape=(scale, scale),
        hidden_layers=5,
        save_to_file=True,
    )

    if "-run" in sys.argv:

        from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        print("------------------")
        eg = EvaluableGraph(mod_graph, verbose=False)
        eg.evaluate(array_format=format)

        print("Finished evaluating graph using array format %s" % format)


if __name__ == "__main__":
    main()
