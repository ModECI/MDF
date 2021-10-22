"""
Code for exporting MDF models to ONNX.
"""

import re

from modeci_mdf.utils import load_mdf
from modeci_mdf.execution_engine import EvaluableGraph

import onnx
from onnx import helper, shape_inference
from onnx import AttributeProto, TensorProto, GraphProto
from onnx.defs import get_schema

from ast import literal_eval

import argparse
import os


def mdf_to_onnx(mdf_model):
    """
    Takes an MDF model object and returns a list of ONNX models for each graph in the model.
    """

    # An MDF model can have multiple graphs. Each graph will be an onnx model
    onnx_models = []
    for graph in mdf_model.graphs:
        print("Processing Graph ", graph.id)

        # Use edges and nodes to construct execution order
        nodenames_in_execution_order = []
        evaluable_graph = EvaluableGraph(graph, verbose=False)
        for idx, edge in enumerate(evaluable_graph.ordered_edges):
            if idx == 0:
                nodenames_in_execution_order.append(edge.sender)
            nodenames_in_execution_order.append(edge.receiver)

        # print(nodenames_in_execution_order, graph.nodes, graph.edges)

        # Generate onnx graph
        onnx_graph = generate_onnx_graph(graph, nodenames_in_execution_order)

        # Make an onnx model from graph
        onnx_model = helper.make_model(onnx_graph)

        # Infer shapes
        onnx_model = shape_inference.infer_shapes(onnx_model)

        # Check model
        onnx.checker.check_model(onnx_model)

        onnx_models.append(onnx_model)

    return onnx_models


def generate_onnx_graph(graph, nodenames_in_execution_order):
    print("Generating ONNX graph for ", graph.id)

    onnx_graph_inputs = []
    onnx_graph_outputs = []
    onnx_nodes = []
    onnx_initializer = []

    for nodename in nodenames_in_execution_order:
        node = graph.get_node(nodename)

        # Get the node, and graph inputs, outputs from that node and the node's initializer
        (
            onnx_node,
            onnx_graph_input,
            onnx_graph_output,
            node_initializer,
        ) = generate_onnx_node(node, graph)

        onnx_nodes.append(onnx_node)
        onnx_graph_inputs.extend(onnx_graph_input)
        onnx_graph_outputs.extend(onnx_graph_output)
        onnx_initializer.extend(node_initializer)

    # Make the final graph
    onnx_graph = helper.make_graph(
        onnx_nodes, graph.id, onnx_graph_inputs, onnx_graph_outputs, onnx_initializer
    )

    return onnx_graph


def generate_onnx_node(node, graph):
    """
    Convert an MDF node into an ONNX node.
    Takes an MDF node, MDF graph and returns the ONNX node, any inputs to the node coming from  outside the graph,
    any outputs from the node going outside the graph, and an initializer for constants
    """

    onnx_graph_inputs = []
    onnx_graph_outputs = []
    onnx_initializer = []

    sender_port_name = {}  # Names of ports that send values to this node

    # Go over all parameters
    # Assumption: there may be multiple parameters but there will only be one function.
    # If there are multiple functions, the code should change so that each function should become its own node.
    for param in node.parameters:
        # If this is a constant
        if param.value:
            # Create a constant onnx node
            name = node.id + "_" + param.id
            constant = helper.make_tensor(
                name, data_type=TensorProto.FLOAT, dims=[], vals=[param.value]
            )
            onnx_initializer.append(constant)
            # The following will be the sender port from the constant onnx node for this parameter
            sender_port_name[param.id] = name
        elif param.function:
            # This is a function and will be part of an onnx node corresponding to this MDF node
            function_name = param.function

            onnx_function_prefix = "onnx::"
            pattern = re.compile(onnx_function_prefix)
            if re.match(pattern, function_name):
                # This is an onnx function
                function_name = function_name[len(onnx_function_prefix) :]
                # Get the arguments that this onnx function expects
                schema = get_schema(function_name)
                # The MDF description would have specified all the expected arguments of ths function
                function_input_names = [param.args[arg.name] for arg in schema.inputs]
            else:
                # Error
                raise "Cannot generate onnx function for the unknown function: {} specfied in the MDF node {}".format(
                    function_name,
                    node.id,
                )

    # Find the inputs to the new ONNX node. These are the senders of the in edges to this node
    node_in_edges = [edge for edge in graph.edges if edge.receiver == node.id]
    for in_edge in node_in_edges:
        sender_port_name[in_edge.receiver_port] = (
            in_edge.sender + "_" + in_edge.sender_port
        )

    onnx_node_input_names = [
        sender_port_name[function_input_name]
        if function_input_name in sender_port_name
        else function_input_name
        for function_input_name in function_input_names
    ]

    # No parameters. Constants became their own nodes earlier
    onnx_node_parameters = {}

    # Find the outputs of the new ONNX node. These are the output ports of the node
    onnx_node_output_names = [node.id + "_" + port.id for port in node.output_ports]

    # print(node.id, node_in_edges,node_out_edges)
    # print(function_name, onnx_node_input_names, onnx_node_output_names)

    # Create an ONNX node
    onnx_node = helper.make_node(
        function_name,
        onnx_node_input_names,
        onnx_node_output_names,
        name=node.id,
        **onnx_node_parameters,
    )

    # Check if any of the node's inputs are the inputs to the ONNX graph itself.
    # These are the node's inputs that don't have an incoming edge.
    input_ports_with_edge = [in_edge.receiver_port for in_edge in node_in_edges]
    input_ports_without_edge = [
        input_port
        for input_port in node.input_ports
        if input_port.id not in input_ports_with_edge
    ]
    if input_ports_without_edge:
        # Create ONNX graph input ports
        for input_port in input_ports_without_edge:
            shape = literal_eval(input_port.shape)
            value_info = helper.make_tensor_value_info(
                input_port.id, TensorProto.FLOAT, shape
            )
            onnx_graph_inputs.append(value_info)

    # Check if any of the node's outputs are the outputs of the ONNX graph.
    # These are the node's outputs that don't have an outgoing edge
    node_out_edges = [edge for edge in graph.edges if edge.sender == node.id]

    output_ports_with_edge = [out_edge.sender_port for out_edge in node_out_edges]
    output_ports_without_edge = [
        output_port
        for output_port in node.output_ports
        if output_port.id not in output_ports_with_edge
    ]
    if output_ports_without_edge:
        # Create ONNX graph output ports
        for output_port in output_ports_without_edge:
            # No need to create output shapes because they are inferred by ONNX
            value_info = helper.make_empty_tensor_value_info(
                node.id + "_" + output_port.id
            )
            onnx_graph_outputs.append(value_info)
    # print("Graph ip op", input_ports_without_edge, output_ports_without_edge)

    return onnx_node, onnx_graph_inputs, onnx_graph_outputs, onnx_initializer


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Converter from MDF to ONNX. "
        "Takes in a JSON/YAML file and generates ONNX files in the same directory as the input file"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="An input JSON/YAML file. "
        "Output files are generated in same directory "
        "with a -m2o.onnx extension",
    )

    args = parser.parse_args()

    convert_mdf_file_to_onnx(args.input_file)


def convert_mdf_file_to_onnx(input_file: str):
    """
    Converter from MDF to ONNX. Takes in a JSON/ONNX file and generates ONNX files.

    Args:
        input_file: The input file path to the MDF file. Output files are generated in same
            directory with -m2o.onnx extensions.

    Returns:
        NoneType
    """

    import os

    # Load the MDF model from file - this is not used anymore
    mdf_model = load_mdf(input_file)
    # print("Loaded MDF file from ", mdf_model_file)

    onnx_models = mdf_to_onnx(mdf_model)

    for onnx_model in onnx_models:
        out_filename = (
            f"{os.path.splitext(input_file)[0]}_{onnx_model.graph.name}-m2o.onnx"
        )
        onnx.save(onnx_model, out_filename)
        print("ONNX output saved in ", out_filename)


# Standalone execution
if __name__ == "__main__":
    main()
