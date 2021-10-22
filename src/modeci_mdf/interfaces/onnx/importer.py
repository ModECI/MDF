"""
Code for importing ONNX models into MDF.
"""
import typing

import onnx

from onnx import (
    ModelProto,
    TensorProto,
    GraphProto,
    AttributeProto,
    numpy_helper,
    shape_inference,
)
from onnx.defs import get_schema

from modeci_mdf.mdf import *


def id_to_port(id: str):
    """Turn unique ONNX output and input value names into valid MDF input and outport names"""

    new_name = str(id).replace(".", "_")

    # If the first character is a digit, precede with an underscore so this can never be interpreted
    # as number down the line.
    if new_name[0].isdigit():
        new_name = "_" + new_name

    return new_name


def get_shape_params(shape: onnx.TensorShapeProto) -> typing.Tuple:
    """
    Small helper function to extract a tuple from the TensorShapeProto. These objects
    can contain both integer dimensions and parameter dimensions that are variable, like
    'batch_size'.

    Args:
        shape: The ONNX shape proto to process.

    Returns:
        A tuple that can contain both integers and strings for parameter dimensions.
    """
    shape = tuple(d.dim_param if d.dim_param != "" else d.dim_value for d in shape.dim)

    # If shape is empty tuple, its a scalar, make it size 1
    if len(shape) == 0:
        shape = (1,)

    return shape


def get_onnx_attribute(a):

    # Use the helpers to get the appropriate value
    val = onnx.helper.get_attribute_value(a)

    # get_attribute_value() can return TensorProto's, lets convert them to a list for JSON
    # FIXME: This begs the question, is JSON a good format for storing large tensors (nope)
    if type(val) == TensorProto:
        return numpy_helper.to_array(val).tolist()
    else:
        return val


def onnx_node_to_mdf(
    node: typing.Union[onnx.NodeProto, onnx.ValueInfoProto],
    onnx_initializer: typing.Dict[str, typing.Dict[str, typing.Any]],
) -> Node:
    """
    Construct an MDF node (and function) from an ONNX NodeProto or ValueInfoProto

    Args:
        node: The ONNX node to use to form the MDF node. Can be a node from the model or
            a ValueInfoProto specifying an input or output.
        onnx_initializer: A specification of values in the graph that ONNX has
            marked as initializer's. This dict is keyed on the name of the parameter,
            the value is another dict with three entries; shape, type, and value.

    Returns:
        The equivalent MDF node for the ONNX node passed in as argument.
    """

    # If this is a ONNX Node,
    if type(node) == onnx.NodeProto:

        # Create and MDF node with parameters
        # FIXME: We need to preserve type info somewhere
        params_dict = {a.name: get_onnx_attribute(a) for a in node.attribute}

        # For any attributes that are sub-graphs, we need to recurse
        for aname, val in params_dict.items():
            if type(val) == GraphProto:
                params_dict[aname] = onnx_to_mdf(val, onnx_initializer=onnx_initializer)

        # If we have we have value constants that feed into this node. Make them parameters
        # instead of input ports
        non_constant_inputs = []
        func_args = {}
        for inp_i, inp in enumerate(node.input):

            # Get the name of the formal argument that corresponds to this input.
            # We need to go to the schema for this.
            # FIXME: We need to make sure we are going the correct schema here ... yuck!
            try:
                arg_name = get_schema(node.op_type).inputs[inp_i].name
            except IndexError:
                arg_name = f"arg_{inp}"

            if inp in onnx_initializer and "value" in onnx_initializer[inp]:
                params_dict[arg_name] = onnx_initializer[inp]["value"]
                func_args[arg_name] = arg_name
            else:
                non_constant_inputs.append(inp)
                func_args[arg_name] = id_to_port(inp)

        # FIXME: parameters must be set or we get JSON serialization error later
        mdf_node = Node(id=node.name)

        for p in params_dict:
            if type(params_dict[p]) == Graph:
                mdf_node.parameters.append(
                    Parameter(
                        id=p, value={"graph_%s" % params_dict[p].id: params_dict[p]}
                    )
                )
            else:
                mdf_node.parameters.append(Parameter(id=p, value=params_dict[p]))

        # Add the function
        # FIXME: There is probably more stuff we need to preserve for ONNX Ops
        func = Parameter(id=node.name, function=f"onnx::{node.op_type}", args=func_args)
        mdf_node.parameters.append(func)

        # Recreate inputs and outputs of ONNX node as InputPorts and OutputPorts
        for inp in non_constant_inputs:
            param_info = onnx_initializer.get(inp, None)
            shape = param_info["shape"] if param_info else ""
            ip = InputPort(id=id_to_port(inp), shape=shape)
            mdf_node.input_ports.append(ip)

        for out in node.output:
            op = OutputPort(id=id_to_port(out), value=func.get_id())
            mdf_node.output_ports.append(op)

    elif type(node) == onnx.ValueInfoProto:
        raise NotImplementedError()
        # # Lets start with an MDF node that uses the ONNX node name as its id. No parameters
        # mdf_node = Node(id=node.name)
        #
        # # This is an input or output node. No Op\Function or parameters. This is just
        # # a simple pass through node with an input and output port with the correct
        # # shape.
        # # FIXME: Should this be necessary? ONNX treats input and output nodes as simple named values.
        # ip1 = InputPort(id=f"in_port",
        #                 shape=str(get_shape_params(node.type.tensor_type.shape))) # FIXME: Why string?
        # mdf_node.input_ports.append(ip1)
        # op1 = OutputPort(id=node.name)
        # op1.value = f"in_port"
        # mdf_node.output_ports.append(op1)

    return mdf_node


def onnx_to_mdf(
    onnx_model: typing.Union[ModelProto, GraphProto],
    onnx_initializer: typing.Dict[str, typing.Dict[str, typing.Any]] = None,
):
    """
    Convert a loaded ONNX model into a MDF model.

    Args:
        onnx_model: The ONNX model to convert. Typically, this is the result of a call to onnx.load()
        onnx_initializer: A specification of values in the graph that ONNX has
            marked as initializer's. This dict is keyed on the name of the parameter,
            the value is another dict with three entries; shape, type, and value.

    Returns:
        An MDF description of the ONNX model.
    """

    if onnx_initializer is None:
        onnx_initializer = {}

    if type(onnx_model) == ModelProto:

        # Do shape inference on the model so we can get shapes of intermediate outputs
        # FIXME: This function has side-effects, it probably shouldn't
        try:
            onnx_model = shape_inference.infer_shapes(onnx_model)
        except RuntimeError:
            pass

        graph = onnx_model.graph

    else:
        graph = onnx_model

    # Get all the nodes in the onnx model, even the inputs and outputs
    onnx_nodes = list(graph.node)

    if hasattr(graph, "initializer"):
        # Parameters that have been initialized with values.
        # FIXME: We need a cleaner way to extract this info.
        onnx_initializer_t = {}
        for t in graph.initializer:
            t_np = numpy_helper.to_array(t)
            onnx_initializer_t[t.name] = {"shape": t_np.shape, "type": str(t_np.dtype)}

        # And the input and intermediate node shapes as well
        for vinfo in list(graph.input) + list(graph.value_info):
            vshape = get_shape_params(vinfo.type.tensor_type.shape)

            try:
                vtype = onnx.helper.printable_type(vinfo.type)
            except AssertionError:
                # Couldn't extract type
                vtype = None

            onnx_initializer_t[vinfo.name] = {"shape": vshape, "type": vtype}

        onnx_initializer = {**onnx_initializer, **onnx_initializer_t}

    # Finally, some nodes are constants, extract the values and drop the nodes.
    # They will be removed in the MDF and passed as named parameters to the Node
    constants = {}
    onnx_nodes_nc = []
    for onnx_node in onnx_nodes:
        if onnx_node.op_type == "Constant":
            v = get_onnx_attribute(onnx_node.attribute[0])
            constants[onnx_node.output[0]] = {
                "shape": v.shape if hasattr(v, "shape") else "(1,)",
                "type": str(v.dtype) if hasattr(v, "dtype") else str(type(v)),
                "value": v,
            }
        else:
            onnx_nodes_nc.append(onnx_node)
    onnx_nodes = onnx_nodes_nc

    # Add constants to the initializer dict
    onnx_initializer = {**onnx_initializer, **constants}

    mod_graph = Graph(id=graph.name)

    # Construct the equivalent nodes in MDF
    mdf_nodes = [
        onnx_node_to_mdf(node=node, onnx_initializer=onnx_initializer)
        for node in onnx_nodes
    ]

    mod_graph.nodes.extend(mdf_nodes)

    # Construct the edges, we will do this by going through all the nodes.
    node_pairs = list(zip(onnx_nodes, mod_graph.nodes))
    for onnx_node, mdf_node in node_pairs:
        if len(onnx_node.output) > 0:
            for i, out in enumerate(onnx_node.output):
                out_port_id = mdf_node.output_ports[i].id

                # Find all node input ports with this outport id
                # FIXME: This is slow for big graphs with lots of edges. Best to build a data structure for this.
                receiver = [
                    (m, ip)
                    for n, m in node_pairs
                    for ip in m.input_ports
                    if out_port_id == ip.id
                ]

                # Make an edge for each receiver of this output port
                for receiver_node, receiver_port in receiver:
                    edge = Edge(
                        id=f"{mdf_node.id}.{out_port_id}_{receiver_node.id}.{receiver_port.id}",
                        sender=mdf_node.id,
                        sender_port=out_port_id,
                        receiver=receiver_node.id,
                        receiver_port=receiver_port.id,
                    )

                    mod_graph.edges.append(edge)

    # If they passed an ONNX model, wrap the graph in a MDF model
    if type(onnx_model) == ModelProto:
        mod = Model(id="ONNX Model")
        mod.graphs.append(mod_graph)
        return mod

    else:
        return mod_graph


def find_subgraphs(
    graph: onnx.GraphProto, graph_dict: typing.Dict[str, GraphProto] = None
) -> typing.Dict[str, GraphProto]:
    """
    Recurse through an ONNX graph and find all subgraphs.

    Args:
        graph: The graph to search.
        graph_list: Insert graphs we find into this dict. Use the parent node name as a key.
            If None, intitialize to empty dict.

    Returns:
        All the subgraphs in the for the graph.
    """

    if graph_dict is None:
        graph_dict = {}

    for node in graph.node:
        for ai, attr in enumerate(node.attribute):
            if attr.type == AttributeProto.GRAPH:
                subgraph = onnx.helper.get_attribute_value(attr)
                graph_dict[f"{node.name}_attr{ai}"] = subgraph
                graph_dict = find_subgraphs(subgraph, graph_dict)
            elif attr.type == AttributeProto.GRAPHS:
                subgraphs = onnx.helper.get_attribute_value(attr)
                for gi, subgraph in enumerate(subgraphs):
                    graph_dict[f"{node.name}_attr{ai}_g{gi}"] = subgraph
                    graph_dict = find_subgraphs(subgraph, graph_dict)

    return graph_dict


def convert_file(input_file: str):
    """
    Simple converter from ONNX to MDF. Takes in ONNX files and generates MDF JSON/YAML files.

    Args:
        input_file: The input file path to the ONNX file. Output files are generated in same
            directory with -mdf.json and -mdf.yml extensions.

    Returns:
        MoneType
    """

    import os

    out_filename = f"{os.path.splitext(input_file)[0]}-mdf"
    onnx_model = onnx.load(input_file)
    onnx.checker.check_model(onnx_model)
    mdf_model = onnx_to_mdf(onnx_model)
    mdf_model.to_json_file(f"{out_filename}.json")
    mdf_model.to_yaml_file(f"{out_filename}.yaml")


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Simple converter from ONNX to MDF. "
        "Takes in ONNX files and generates MDF JSON/YAML"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="An input ONNX file. "
        "Output files are generated in same directory "
        "with -mdf.json and -mdf.yml extensions.",
    )

    args = parser.parse_args()

    convert_file(args.input_file)


if __name__ == "__main__":
    main()
