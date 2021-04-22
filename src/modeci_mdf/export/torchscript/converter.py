"""
Functions for converting from PyTorch Torchscript to MDF models.

This code was originally inspired by the following blog post:

    Mike He, "From Models to Computation Graphs (Part I)", https://ad1024.space/articles/22
"""
import re
import logging
import os
import itertools
from typing import Union, Dict, Any, Tuple

import torch

from modeci_mdf.mdf import Model, Graph, Node, Edge, InputPort, OutputPort, Function

logger = logging.getLogger(__name__)


def make_node_id(node: torch.Node) -> str:
    """Helper function to get a unique name (used in MDF as id) from a TorchScript Node object"""
    return "_".join(
        [node.kind().split("::")[-1]] + [str(o.unique()) for o in node.outputs()]
    )


def make_func_id(node: torch.Node) -> str:
    """Helper function to get a unique name (used in MDF as id) for a TorchScript node's op/function."""
    return f"{node.kind()}_1"


def make_model_graph_name(
    model: Union[torch.ScriptModule, torch.ScriptFunction]
) -> Tuple[str, str]:
    """Helper function that generates a clean graph and model name from a TorchScript model"""
    # Get a name for this module
    try:
        model_name = model.original_name.split(".")[-1]
        graph_name = f"{model_name}Graph"
    except AttributeError:
        model_name = model.qualified_name.split(".")[-1]
        graph_name = f"{model_name}_graph"

    return model_name, graph_name


def get_graph_constants(graph: torch.Graph) -> Dict[str, Any]:
    """
    Find all constant nodes in the graph and extract their values as a proper JSON serializable value.

    Args:
        graph: The graph to extract constants from.

    Returns:
        A Dict that maps the constant nodes unique TorchScript node ID string to its value.
    """
    consts = {}
    for n in graph.findAllNodes("prim::Constant"):
        for o in n.outputs():
            value = o.toIValue()

            # FIXME: It doesn't seem like these types are being serialized to JSON properly, encode as strings for now
            if value is None:
                value = "None"
            elif type(value) is bool:
                value = str(value).lower()
            elif type(value) is torch.device:
                value = str(value)
            elif type(value) is torch.Tensor:
                value = value.numpy().tolist()

            consts[o.unique()] = value

    return consts


def get_shape(node: torch.Node) -> Dict:
    """Helper function for extracting shape for each node output """
    outputs = dict()
    for o in node.outputs():
        typeIs = o.type()
        outputs[o.unique()] = dict(
            type=re.match(r"\w+", typeIs.str()).group(), sizes=tuple(typeIs.sizes())
        )
    return outputs


def get_value(node: torch.Node) -> Dict:
    outputs = dict()
    for o in node.outputs():
        typeIs = o.type().str()
        value = o.toIValue()
        outputs[o.unique()] = dict(
            type=typeIs,
            value=value,
            sizes=len(list(node.outputs())) if typeIs.endswith("[]") else 1,
        )
    return outputs


def torchnode_to_mdfnode(
    node: torch.Node, graph: torch.Graph, consts: Dict[str, Any] = None
) -> Union[Node, None]:
    """
    Convert a TorchScript node to an MDF node.

    Args:
        node: The node to convert.
        graph: The graph that this node is a member.
        consts: A dict containing any constants in the graph.

    Returns:
        The MDF node for this TorchScript node. prim::Constant nodes are excluded from the MDF graph and are
        instead placed as parameters. In this case, return None.
    """
    op = node.kind()

    # Exclude constants (as nodes) from the MDF graph. We will instead insert them as parameters to the nodes that
    # they project to.
    if op == "prim::Constant":
        return None

    schema = (
        torch._C.parse_schema(node.schema())
        if "no schema" not in node.schema()
        else None
    )

    outputs = [o.unique() for o in node.outputs()]
    inputs = [i.unique() for i in node.inputs()]

    if schema:
        schema_args = schema.arguments

        # Get any input to this node that is TorchScript node.kind() prim::Constant, make it a parameter
        parameters = {
            schema_args[i].name: consts[inp]
            for i, inp in enumerate(inputs)
            if inp in consts
        }

    else:
        logger.warning(
            f"Schema not found for TorchScript node ({node}), using placeholders for argument names."
        )
        schema_args = [f"arg{i}" for i in range(len(inputs))]

        # Get any input to this node that is TorchScript node.kind() prim::Constant, make it a parameter
        parameters = {
            schema_args[i]: consts[inp] for i, inp in enumerate(inputs) if inp in consts
        }

    mdf_node = Node(id=make_node_id(node), parameters=parameters)

    # Add any output ports
    for o in outputs:
        mdf_node.output_ports.append(OutputPort(id=o, value=make_func_id(node)))

    # Get all constant nodes in the graph if the user didn't pass them in.
    if consts is None:
        consts = get_graph_constants(graph)

    # Get any inputs to the graph, and their debug names
    graph_inputs = {inp.unique(): inp.debugName() for inp in graph.inputs()}

    # Add any input ports to the node, exclude inputs from constant nodes, these are parameters now
    for inp_i, inp in enumerate(inputs):
        if inp not in consts:
            # If this is a graph level input, use its names for the input port id
            ip_name = graph_inputs[inp] if inp in graph_inputs else inp

            # Try to get the shape and type of the input port
            inp_type = node.inputsAt(inp_i).type()
            try:
                shape = str(inp_type.sizes()) if inp_type.sizes() else "(?)"
            except RuntimeError:
                shape = "(?)"

            mdf_node.input_ports.append(
                InputPort(id=ip_name, shape=shape, type=str(inp_type))
            )

    # Construct the arguments for the function
    if schema:
        arguments = {}
        ip_i = 0
        for arg_i, arg in enumerate(schema_args):
            if inputs[arg_i] in consts:
                value = (
                    arg.name
                )  # Just use the parameter name, there is no input port for constants
            else:
                value = mdf_node.input_ports[ip_i].id
                ip_i = ip_i + 1

            arguments[arg.name] = value
    else:
        arguments = {f"arg{i}": ip.id for i, ip in enumerate(mdf_node.input_ports)}

    # Add function
    f = Function(id=make_func_id(node), function=op, args=arguments)
    mdf_node.functions.append(f)

    return mdf_node


def torchscript_to_mdf(
    model: torch.ScriptModule, mdf_graph: Graph = None
) -> Union[Model, Graph]:
    """
    Convert a TorchScript model to an MDF model.

    Args:
        model: The model to translate into MDF.
        mdf_graph: If the graph that is constructed should be added to an existing mdf model, pass it here. By default,
            this is None which means a new MDF Model instance will be constructed and returned.

    Returns:
        The translated MDF model
    """

    # Get the graph and nodes from the TorchScript model
    try:
        graph = model.graph
    except AttributeError:

        # Looks like the model is not compiled. Lets try to compile it
        logger.warning(
            "Model argument does not appear to be a torch.ScriptModule or torch.ScriptFunction, trying to "
            "JIT compile it ... cross your fingers."
        )
        model = torch.jit.script(model)

    # If mdf_graph is None we are probably at the top level of the possibly recursive construction process of a
    # TorchScript -> MDF conversion. In this case, we will construct a MDF Model and graph to for the top level.
    if mdf_graph is None:
        model_name, graph_name = make_model_graph_name(model)

        mdf_model = Model(id=model_name)
        mdf_graph = Graph(id=graph_name)
        mdf_model.graphs.append(mdf_graph)
    else:
        mdf_model = None

    # Get all constant nodes in the graph
    consts = get_graph_constants(graph)

    # Get any inputs to the graph, and their debug names
    graph_inputs = {inp.unique(): inp.debugName() for inp in graph.inputs()}

    # For every node, cache its input edges. This will let us look this up quickly for
    # any node.
    node_to_in_edge = {
        node: [i.unique() for i in node.inputs()] for node in graph.nodes()
    }

    for node in graph.nodes():

        mdf_node = torchnode_to_mdfnode(node=node, graph=graph, consts=consts)

        # If we are excluding this node from the MDF graph, skip it.
        if mdf_node is None:
            continue

        mdf_graph.nodes.append(mdf_node)

        # Now we need to examine all outgoing edges from this node and add them to the MDF graph. We do this by looping
        # over all nodes in the graph and seeing if they have an input from the node we just constructed. This is
        # O(n^2) in terms of the number of the nodes!
        outputs = [o.unique() for o in node.outputs()]
        for to in graph.nodes():

            # Lookup this nodes input edges
            to_inputs = node_to_in_edge[to]

            edges = set(outputs) & set(to_inputs)
            for edge in edges:
                from_id = make_node_id(node)
                from_port = mdf_node.output_ports[outputs.index(edge)].id
                to_id = make_node_id(to)
                to_port = mdf_node.output_ports[outputs.index(edge)].id
                mdf_edge = Edge(
                    id=f"{from_id}_{to_id}",
                    sender=from_id,
                    sender_port=f"{from_port}",
                    receiver=to_id,
                    receiver_port=f"{to_port}",
                )
                mdf_graph.edges.append(mdf_edge)

    # If we haven't wrapped this graph in a model class
    if mdf_model is None:
        return mdf_graph
    else:
        return mdf_model


if __name__ == "__main__":
    """Test a simple function"""

    def simple(x, y):
        b = x + y
        return 2 * b

    model = torch.jit.script(simple)
    mdf_model = torchscript_to_mdf(model)

    print(mdf_model.to_yaml())
