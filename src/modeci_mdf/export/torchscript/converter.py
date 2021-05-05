"""
Functions for converting from PyTorch Torchscript to MDF models.

This code was originally inspired by the following blog post:

    Mike He, "From Models to Computation Graphs (Part I)", https://ad1024.space/articles/22
"""
import re
import inspect
import logging
import os
import itertools
import numpy as np

from typing import Union, Dict, Any, Tuple, List, Callable
import onnx.defs


import torch

from modeci_mdf.mdf import Model, Graph, Node, Edge, InputPort, OutputPort, Function
from modeci_mdf.export.onnx import get_onnx_attribute
from modeci_mdf.onnx_functions import onnx_opset_version as modeci_onnx_opset_version


logger = logging.getLogger(__name__)


def get_mdf_graph_inputs(mdf_graph: Graph) -> List[Tuple[Node, InputPort]]:
    """Simple helper to enumerate InputPorts for and MDF graph that specify no incoming edge."""

    # Get all input ports
    all_ips = [
        (node, ip) for node in mdf_model.graphs[0].nodes for ip in node.input_ports
    ]

    # Get all sender ports
    all_sender_ports = {(e.sender, e.sender_port) for e in graph.edges}

    graph_inputs = filter(lambda x: x not in all_sender_ports, all_ips)

    return graph_inputs


def convert_to_serializable(value):
    """Helper function that converts some common unserializable types to JSON seriralizable types"""
    if type(value) is torch.device:
        value = str(value)
    elif type(value) is torch.Tensor:
        value = value.numpy().tolist()

    return value


def make_node_id(node: torch.Node) -> str:
    """Helper function to get a unique name (used in MDF as id) from a TorchScript Node object"""
    return "_".join(
        [node.kind().split("::")[-1]] + [str(o.unique()) for o in node.outputs()]
    )


def make_func_id(node: torch.Node) -> str:
    """Helper function to get a unique name (used in MDF as id) for a TorchScript node's op/function."""
    return f"{node.kind()}_1"


def make_port_name(name: str):
    """Turn unique TorchScript output and input value names into valid MDF input and outport names"""
    new_name = str(name).replace(".", "_")

    # If the first character is a digit, precede with an underscore so this can never be interpreted
    # as number down the line.
    if new_name[0].isdigit():
        new_name = "_" + new_name

    return new_name


def make_model_graph_name(
    model: Union[torch.ScriptModule, torch.ScriptFunction]
) -> Tuple[str, str]:
    """Helper function that generates a clean graph and model name from a TorchScript model"""
    # Get a name for this module
    try:
        model_name = model.original_name.split(".")[-1]
        graph_name = f"{model_name}Graph"
    except AttributeError:
        try:
            model_name = model.qualified_name.split(".")[-1]
            graph_name = f"{model_name}_graph"
        except AttributeError:
            # It hasn't been compiled yet, use the class name I guess
            model_name = type(model).__name__.split(".")[-1]
            graph_name = f"{model_name}Graph"

    return model_name, graph_name


def process_torch_schema(
    node: torch.Node, consts: Dict
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Parse a TorchScript node schema into argument names and constant attributes (parameters in MDF)

    Args:
        node: The TorchScript node to retrieve the schema from.
        consts: The constant nodes Dict for the graph we are working with.

    Returns:
        A tuple containing a list of argument names and Dict of parameter names and values.
    """

    # Get the input node names
    inputs = [i.unique() for i in node.inputs()]

    # If this is a TorchScript funciton (aten::*), it should have a schema string to parse.
    if "no schema" not in node.schema():
        schema = torch._C.parse_schema(node.schema())

        # Get the arguments and covert to a simple List[str]
        schema_args = schema.arguments
        schema_args = [schema_args[i].name for i, inp in enumerate(inputs)]

    else:
        logger.warning(
            f"Schema not found for TorchScript node ({node}), using placeholders for argument names."
        )
        schema_args = [f"arg{i}" for i in range(len(inputs))]

    # Get any input to this node that is TorchScript node.kind() prim::Constant, make those a parameter
    parameters = {
        schema_args[i]: consts[inp] for i, inp in enumerate(inputs) if inp in consts
    }

    return schema_args, parameters


def process_onnx_schema(
    node: torch.Node, graph_inputs: Dict[str, str]
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Retrieve the argument names and attributes (parameters in MDF) for this Operation.

    Args:
        op: The TorchScript node containing the ONNX operation.
        graph_inputs: A dict mapping graph level input unique ids to their debug names

    Returns:
        A two element tuple:
            - A dict representing argument names mapping to input port ids
            - A dict mapping parameters (ONNX attributes) names mapping to values
    """

    # Convert the graph inputs to their MDF port names
    graph_inputs = {
        make_port_name(k): make_port_name(v) for k, v in graph_inputs.items()
    }

    # Get the input node names
    inputs = [i.unique() for i in node.inputs()]

    # If this is an ONNX op, we need to get the schema from ONNX
    if "onnx::" in node.kind():
        try:
            schema = onnx.defs.get_schema(node.kind().split("::")[-1])

            schema_args = {}
            if len(schema.inputs) > 0:
                # If the first argument is variadic. Represent this as a list of input port names
                if (
                    schema.inputs[0].option
                    == onnx.defs.OpSchema.FormalParameterOption.Variadic
                ):
                    schema_args = {
                        schema.inputs[0].name: str(
                            [make_port_name(inp) for i, inp in enumerate(inputs)]
                        )
                    }
                else:
                    schema_args = {
                        schema.inputs[i].name: make_port_name(inp)
                        for i, inp in enumerate(inputs)
                    }

        except onnx.onnx_cpp2py_export.defs.SchemaError:
            logger.warning(
                f"Could not find ONNX OpSchema for op {node.kind()}, using placeholder names for arguments."
            )
            schema_args = {
                f"arg{i}": make_port_name(inp) for i, inp in enumerate(inputs)
            }
    else:
        raise ValueError(f"Cannot process ONNX schema for non ONNX node: {node}")

    # Replace any instance of a graph input with its debug name
    for arg_name, ip_name in schema_args.items():
        if ip_name in graph_inputs:
            schema_args[arg_name] = graph_inputs[ip_name]

    # ONNX attributes are equivalent to MDF parameters really
    parameters = {
        aname: convert_to_serializable(node[aname]) for aname in node.attributeNames()
    }

    return schema_args, parameters


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
            consts[o.unique()] = convert_to_serializable(o.toIValue())

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

    # Get all constant nodes in the graph if the user didn't pass them in.
    if consts is None:
        consts = get_graph_constants(graph)

    # Get any inputs to the graph, and their debug names
    graph_inputs = {inp.unique(): inp.debugName() for inp in graph.inputs()}

    outputs = [o.unique() for o in node.outputs()]
    inputs = [i.unique() for i in node.inputs()]

    # Get the argument names and parameter names and values for this Node's operation
    if "onnx::" in op:
        arguments, parameters = process_onnx_schema(node, graph_inputs)
    else:
        arguments, parameters = process_torch_schema(node, consts, graph_inputs)

    mdf_node = Node(id=make_node_id(node), parameters=parameters)

    # Add any output ports
    for o in outputs:
        mdf_node.output_ports.append(
            OutputPort(id=make_port_name(o), value=make_func_id(node))
        )

    # Add any input ports to the node, exclude inputs from constant nodes, these are parameters now
    for inp_i, inp in enumerate(inputs):
        if inp not in consts:
            # If this is a graph level input, use its names for the input port id
            ip_name = graph_inputs[inp] if inp in graph_inputs else inp

            # Fixup ip_name if it contains any "." characters
            # Also add an preceding "_" so its interpretted as string
            ip_name = make_port_name(ip_name)

            # Try to get the shape and type of the input port
            inp_type = node.inputsAt(inp_i).type()
            try:
                shape = str(inp_type.sizes()) if inp_type.sizes() else "(?)"
            except RuntimeError:
                shape = "(?)"

            mdf_node.input_ports.append(
                InputPort(id=ip_name, shape=shape, type=str(inp_type))
            )

    # Add function
    f = Function(id=make_func_id(node), function=op, args=arguments)
    mdf_node.functions.append(f)

    return mdf_node


def torchscript_to_mdf(
    model: Union[Callable, torch.nn.Module, torch.ScriptFunction, torch.ScriptModule],
    args: Union[None, torch.Tensor, Tuple[torch.Tensor]] = None,
    example_outputs: Union[None, torch.Tensor, Tuple[torch.Tensor]] = None,
    trace: bool = False,
    use_onnx_ops: bool = True,
    mdf_graph: Graph = None,
) -> Union[Model, Graph]:
    r"""
    Convert a TorchScript model to an MDF model. By default, this function will invoke `torch.jit.script` on the
    model to compile it down to TorchScript IR and simplify the graph before exporting the MDF. The default is
    to use ONNX operations when possible and fallback to ATEN\Torch ops when ONNX support is not available
    (`torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK` mode). To use allATEN\Torch ops, set use_onnx_ops to False.

    Args:
        model: The model to translate into MDF.
        args: The input arguments for this model. If a nn.Module is passed then the model will be traced with these
            inputs. If a ScriptModule is passed, they are still needed to deterimine input shapes.
        example_outputs: Example outputs from the model for determing output shapes.
        trace: Force the use of tracing to compile the model. The default is to use torch.jit.script
        use_onnx_ops: Use ONNX ops when possible, fallback to ATEN ops when not available. Default is True. If False,
            use only ATEN ops.
        mdf_graph: If the graph that is constructed should be added to an existing mdf model, pass it here. By default,
            this is None which means a new MDF Model instance will be constructed and returned.

    Returns:
        The translated MDF model
    """

    # Get the graph and nodes from the TorchScript model
    try:
        # If the graph attribute is available, we are dealing with a already jitted model (ScriptModule, ScriptFunciton,
        # etc.)
        graph = model.graph
        jit_model = model
    except AttributeError:

        # Lets jit things, if the user doesn't want to trace or we are dealing with a standard Python function, we need
        # to JIT script it.
        if not trace or inspect.isfunction(model):
            jit_model = torch.jit.script(model)
            graph = jit_model.graph
        else:
            # If the user wants to trace, _model_to_graph below will take care of that for us.
            graph = None

    if use_onnx_ops:
        operator_export_type = torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        operator_export_type = torch._C._onnx.OperatorExportTypes.RAW

    # Call out to a part of the ONNX exporter that simiplifies the graph before ONNX export.
    from torch.onnx.utils import _model_to_graph
    from torch.onnx import TrainingMode
    from torch.onnx.symbolic_helper import (
        _export_onnx_opset_version,
        _set_opset_version,
    )

    previous_opset_version = _export_onnx_opset_version
    _set_opset_version(modeci_onnx_opset_version)
    graph, params_dict, torch_out = _model_to_graph(
        model=jit_model if graph else model,
        args=args,
        example_outputs=example_outputs,
        do_constant_folding=False,
        training=TrainingMode.EVAL,
        _retain_param_name=True,
        operator_export_type=operator_export_type,
        dynamic_axes={},
    )
    _set_opset_version(previous_opset_version)

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

    # Replace in "." for "_" in parameter names. We have done this elsewhere when creating the input ports for these
    # parameters.
    params_dict = {make_port_name(k): v for k, v in params_dict.items()}

    # If we haven't wrapped this graph in a model class
    if mdf_model is None:
        return mdf_graph, params_dict
    else:

        # Set the ONNX opset version
        mdf_model.onnx_opset_version = _export_onnx_opset_version

        return mdf_model, params_dict


if __name__ == "__main__":
    """Test a simple function"""

    import torch
    import torch.nn as nn

    from modeci_mdf.condition_scheduler import EvaluableGraphWithConditions

    class InceptionBlocks(nn.Module):
        def __init__(self):
            super().__init__()

            self.asymmetric_pad = nn.ZeroPad2d((0, 1, 0, 1))
            self.conv2d = nn.Conv2d(
                in_channels=5, out_channels=64, kernel_size=(5, 5), padding=2, bias=True
            )
            self.prelu = nn.PReLU(init=0.0)
            self.averagepooling2d = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.conv2d2 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu2 = nn.PReLU(init=0.0)
            self.conv2d3 = nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu3 = nn.PReLU(init=0.0)
            self.conv2d4 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu4 = nn.PReLU(init=0.0)
            self.averagepooling2d2 = nn.AvgPool2d((2, 2), stride=1)
            self.conv2d5 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu5 = nn.PReLU(init=0.0)
            self.conv2d6 = nn.Conv2d(
                in_channels=64,
                out_channels=48,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu6 = nn.PReLU(init=0.0)
            self.conv2d7 = nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu7 = nn.PReLU(init=0.0)
            self.conv2d8 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.conv2d9 = nn.Conv2d(
                in_channels=240,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.conv2d10 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu8 = nn.PReLU(init=0.0)
            self.conv2d11 = nn.Conv2d(
                in_channels=64,
                out_channels=92,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            self.prelu9 = nn.PReLU(init=0.0)
            self.prelu10 = nn.PReLU(init=0.0)
            self.averagepooling2d3 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d12 = nn.Conv2d(
                in_channels=240,
                out_channels=64,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu11 = nn.PReLU(init=0.0)
            self.conv2d13 = nn.Conv2d(
                in_channels=64,
                out_channels=92,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu12 = nn.PReLU(init=0.0)
            self.prelu13 = nn.PReLU(init=0.0)
            self.averagepooling2d4 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.conv2d14 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu14 = nn.PReLU(init=0.0)
            self.conv2d15 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            self.prelu15 = nn.PReLU(init=0.0)
            self.conv2d16 = nn.Conv2d(
                in_channels=340,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu16 = nn.PReLU(init=0.0)
            self.conv2d17 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu17 = nn.PReLU(init=0.0)
            self.averagepooling2d5 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d18 = nn.Conv2d(
                in_channels=340,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu18 = nn.PReLU(init=0.0)
            self.conv2d19 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu19 = nn.PReLU(init=0.0)
            self.conv2d20 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu20 = nn.PReLU(init=0.0)
            self.conv2d21 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu21 = nn.PReLU(init=0.0)
            self.conv2d22 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu22 = nn.PReLU(init=0.0)
            self.averagepooling2d6 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d23 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu23 = nn.PReLU(init=0.0)
            self.conv2d24 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(5, 5),
                padding=2,
                bias=True,
            )
            self.prelu24 = nn.PReLU(init=0.0)
            self.conv2d25 = nn.Conv2d(
                in_channels=476,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu25 = nn.PReLU(init=0.0)
            self.averagepooling2d7 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.conv2d26 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu26 = nn.PReLU(init=0.0)
            self.averagepooling2d8 = nn.AvgPool2d((2, 2), stride=1, padding=0)
            self.conv2d27 = nn.Conv2d(
                in_channels=476,
                out_channels=92,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu27 = nn.PReLU(init=0.0)
            self.conv2d28 = nn.Conv2d(
                in_channels=92,
                out_channels=128,
                kernel_size=(3, 3),
                padding=1,
                bias=True,
            )
            self.prelu28 = nn.PReLU(init=0.0)
            self.conv2d29 = nn.Conv2d(
                in_channels=476,
                out_channels=128,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            )
            self.prelu29 = nn.PReLU(init=0.0)
            self.dense = nn.Linear(22273, 1096, bias=True)
            self.prelu30 = nn.PReLU(init=0.0)
            self.dense2 = nn.Linear(1096, 1096, bias=True)
            self.prelu31 = nn.PReLU(init=0.0)
            self.dense3 = nn.Linear(1096, 180, bias=True)

        def forward(self, galaxy_images_output, ebv_output):
            conv2d_output = self.conv2d(galaxy_images_output)
            prelu_output = self.prelu(conv2d_output)
            averagepooling2d_output = self.averagepooling2d(prelu_output)
            conv2d_output2 = self.conv2d2(averagepooling2d_output)
            prelu_output2 = self.prelu2(conv2d_output2)
            conv2d_output3 = self.conv2d3(prelu_output2)
            prelu_output3 = self.prelu3(conv2d_output3)
            conv2d_output4 = self.conv2d4(averagepooling2d_output)
            prelu_output4 = self.prelu4(conv2d_output4)
            prelu_output4 = self.asymmetric_pad(prelu_output4)
            averagepooling2d_output2 = self.averagepooling2d2(prelu_output4)
            conv2d_output5 = self.conv2d5(averagepooling2d_output)
            prelu_output5 = self.prelu5(conv2d_output5)
            conv2d_output6 = self.conv2d6(averagepooling2d_output)
            prelu_output6 = self.prelu6(conv2d_output6)
            conv2d_output7 = self.conv2d7(prelu_output6)
            prelu_output7 = self.prelu7(conv2d_output7)
            concatenate_output = torch.cat(
                (prelu_output5, prelu_output3, prelu_output7, averagepooling2d_output2),
                dim=1,
            )
            conv2d_output8 = self.conv2d8(concatenate_output)
            conv2d_output9 = self.conv2d9(concatenate_output)
            conv2d_output10 = self.conv2d10(concatenate_output)
            prelu_output8 = self.prelu8(conv2d_output10)
            conv2d_output11 = self.conv2d11(prelu_output8)
            prelu_output9 = self.prelu9(conv2d_output11)
            prelu_output10 = self.prelu10(conv2d_output8)
            prelu_output10 = self.asymmetric_pad(prelu_output10)
            averagepooling2d_output3 = self.averagepooling2d3(prelu_output10)
            conv2d_output12 = self.conv2d12(concatenate_output)
            prelu_output11 = self.prelu11(conv2d_output12)
            conv2d_output13 = self.conv2d13(prelu_output11)
            prelu_output12 = self.prelu12(conv2d_output13)
            prelu_output13 = self.prelu13(conv2d_output9)
            concatenate_output2 = torch.cat(
                (
                    prelu_output13,
                    prelu_output12,
                    prelu_output9,
                    averagepooling2d_output3,
                ),
                dim=1,
            )
            averagepooling2d_output4 = self.averagepooling2d4(concatenate_output2)
            conv2d_output14 = self.conv2d14(averagepooling2d_output4)
            prelu_output14 = self.prelu14(conv2d_output14)
            conv2d_output15 = self.conv2d15(prelu_output14)
            prelu_output15 = self.prelu15(conv2d_output15)
            conv2d_output16 = self.conv2d16(averagepooling2d_output4)
            prelu_output16 = self.prelu16(conv2d_output16)
            conv2d_output17 = self.conv2d17(averagepooling2d_output4)
            prelu_output17 = self.prelu17(conv2d_output17)
            prelu_output17 = self.asymmetric_pad(prelu_output17)
            averagepooling2d_output5 = self.averagepooling2d5(prelu_output17)
            conv2d_output18 = self.conv2d18(averagepooling2d_output4)
            prelu_output18 = self.prelu18(conv2d_output18)
            conv2d_output19 = self.conv2d19(prelu_output18)
            prelu_output19 = self.prelu19(conv2d_output19)
            concatenate_output3 = torch.cat(
                (
                    prelu_output16,
                    prelu_output19,
                    prelu_output15,
                    averagepooling2d_output5,
                ),
                dim=1,
            )
            conv2d_output20 = self.conv2d20(concatenate_output3)
            prelu_output20 = self.prelu20(conv2d_output20)
            conv2d_output21 = self.conv2d21(prelu_output20)
            prelu_output21 = self.prelu21(conv2d_output21)
            conv2d_output22 = self.conv2d22(concatenate_output3)
            prelu_output22 = self.prelu22(conv2d_output22)
            prelu_output22 = self.asymmetric_pad(prelu_output22)
            averagepooling2d_output6 = self.averagepooling2d6(prelu_output22)
            conv2d_output23 = self.conv2d23(concatenate_output3)
            prelu_output23 = self.prelu23(conv2d_output23)
            conv2d_output24 = self.conv2d24(prelu_output23)
            prelu_output24 = self.prelu24(conv2d_output24)
            conv2d_output25 = self.conv2d25(concatenate_output3)
            prelu_output25 = self.prelu25(conv2d_output25)
            concatenate_output4 = torch.cat(
                (
                    prelu_output25,
                    prelu_output21,
                    prelu_output24,
                    averagepooling2d_output6,
                ),
                dim=1,
            )
            averagepooling2d_output7 = self.averagepooling2d7(concatenate_output4)
            conv2d_output26 = self.conv2d26(averagepooling2d_output7)
            prelu_output26 = self.prelu26(conv2d_output26)
            prelu_output26 = self.asymmetric_pad(prelu_output26)
            averagepooling2d_output8 = self.averagepooling2d8(prelu_output26)
            conv2d_output27 = self.conv2d27(averagepooling2d_output7)
            prelu_output27 = self.prelu27(conv2d_output27)
            conv2d_output28 = self.conv2d28(prelu_output27)
            prelu_output28 = self.prelu28(conv2d_output28)
            conv2d_output29 = self.conv2d29(averagepooling2d_output7)
            prelu_output29 = self.prelu29(conv2d_output29)
            concatenate_output5 = torch.cat(
                (prelu_output29, prelu_output28, averagepooling2d_output8), dim=1
            )
            flatten_output = torch.flatten(concatenate_output5)
            concatenate_output6 = torch.cat((flatten_output, ebv_output), dim=0)
            dense_output = self.dense(concatenate_output6)
            prelu_output30 = self.prelu30(dense_output)
            dense_output2 = self.dense2(prelu_output30)
            prelu_output31 = self.prelu31(dense_output2)
            dense_output3 = self.dense3(prelu_output31)

            return dense_output3

    torch.manual_seed(0)

    galaxy_images_output = torch.zeros((1, 5, 64, 64))
    ebv_output = torch.zeros((1,))

    model = InceptionBlocks()

    output = model(galaxy_images_output, ebv_output)

    model.eval()

    mdf_model, params_dict = torchscript_to_mdf(
        model=model,
        args=(galaxy_images_output, ebv_output),
        example_outputs=output,
        trace=True,
    )
    print(mdf_model.to_yaml())

    mdf_graph = mdf_model.graphs[0]

    params_dict["input_1"] = galaxy_images_output

    eg = EvaluableGraphWithConditions(graph=mdf_graph, verbose=False)
    eg.evaluate(initializer=params_dict)

    assert np.allclose(
        output.detach().numpy(),
        eg.enodes["AveragePool_11"].evaluable_outputs["_11"].curr_value,
    )
