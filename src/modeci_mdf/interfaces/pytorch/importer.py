"""
Functions for converting from PyTorch TorchScript to MDF models.

This code was originally inspired by the following blog post:

    Mike He, "From Models to Computation Graphs (Part I)", https://ad1024.space/articles/22
"""
import inspect
import logging

from typing import Union, Dict, Any, Tuple, List, Callable
import onnx.defs


import torch

from modeci_mdf.mdf import Model, Graph, Node, Edge, InputPort, OutputPort, Parameter
from modeci_mdf.functions.onnx import onnx_opset_version as modeci_onnx_opset_version


logger = logging.getLogger(__name__)


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
    node: torch.Node, consts: Dict, port_mapper: "PortMapper"
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
    node: torch.Node, port_mapper: "PortMapper"
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Retrieve the argument names and attributes (parameters in MDF) for this Operation.

    Args:
        op: The TorchScript node containing the ONNX operation.
        port_mapper: The utitlity class for assigning TorchScript input output ids to Input Output Port ids.

    Returns:
        A two element tuple:
            - A dict representing argument names mapping to input port ids
            - A dict mapping parameters (ONNX attributes) names mapping to values
    """

    # Get the input node names
    inputs = [i.unique() for i in node.inputs()]

    # If this is an ONNX op, we need to get the schema from ONNX
    if "onnx::" in node.kind():
        try:
            schema = onnx.defs.get_schema(node.kind().split("::")[-1])

            schema_args = {}
            if len(schema.inputs) > 0:
                # If the first argument is variadic. Represent this as a list of input port names
                if schema.inputs[0].option.name == "Variadic":
                    schema_args = {
                        schema.inputs[0].name: str(
                            [
                                port_mapper.id_to_port(inp)
                                for i, inp in enumerate(inputs)
                            ]
                        )
                    }
                else:
                    schema_args = {
                        schema.inputs[i].name: port_mapper.id_to_port(inp)
                        for i, inp in enumerate(inputs)
                    }

        except onnx.onnx_cpp2py_export.defs.SchemaError:
            logger.warning(
                f"Could not find ONNX OpSchema for op {node.kind()}, using placeholder names for arguments."
            )
            schema_args = {
                f"arg{i}": port_mapper.id_to_port(inp) for i, inp in enumerate(inputs)
            }
    else:
        raise ValueError(f"Cannot process ONNX schema for non ONNX node: {node}")

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


class PortMapper:
    r"""
    A simple class that handles mapping TorchScript input\ouput ids to MDF InputPort\OutputPort ids. It keeps track of
    annoying details like graph level inputs and stuff.
    """

    def __init__(self, graph: torch.Graph, args: Tuple):

        # Keep generate special names for all the graph inputs and parameters
        self.graph_inputs = PortMapper._get_graph_inputs_dict(graph, args)

    def id_to_port(self, id: str):
        """Turn unique TorchScript output and input value names into valid MDF input and outport names"""

        # If this id is a graph input, use its debug name
        if id in self.graph_inputs:
            id = self.graph_inputs[id]

        new_name = str(id).replace(".", "_")

        # If the first character is a digit, precede with an underscore so this can never be interpreted
        # as number down the line.
        if new_name[0].isdigit():
            new_name = "_" + new_name

        return new_name

    def port_to_id(self, name: str):
        """Transform a port name back to is TorchScript ID"""

        # If first character is underscore, remove it
        id = name
        if name[0] == "_":
            id = name[1:]

        # Replace any remaining underscores with '.'
        id = id.replace("_", ".")

        # If this is a numeric id, make it an int again
        if id[0].isdigit():
            id = int(id)

        # If this id is actually a debugName from a graph input, use that
        for input_id, debug_name in self.graph_inputs.items():
            if debug_name == id:
                return input_id

        return id

    @staticmethod
    def _get_graph_inputs_dict(
        graph: torch.Graph, args: Tuple[torch.Tensor]
    ) -> Dict[str, str]:
        """
        Create a dict mapping graph input torch.Node ids to default names. The default names are just:
            - input1
            - input2
            - etc.

        Any parameters for the model will also be graph inputs but their node.debugName() will be used
        instead.
        """
        graph_inputs = {
            inp.unique(): inp.debugName() for i, inp in enumerate(graph.inputs())
        }

        # The first len(args) inputs should be the input arguments to the function or forward method. Lets
        # canonicalize them.
        input_ids = list(graph_inputs.keys())
        for i in range(len(args)):
            graph_inputs[input_ids[i]] = f"input{i + 1}"

        return graph_inputs


def torchnode_to_mdfnode(
    node: torch.Node,
    graph: torch.Graph,
    consts: Dict[str, Any],
    port_mapper: "PortMapper",
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

    # If we are dealing with a loop node, we need to recursively create a sub-graph for the loop body
    if op == "onnx::Loop":
        sub_mdf_graph = Graph(id=f"LoopSubgraph{make_node_id(node)}")
        block_graph = list(node.blocks())[0]
        translate_graph(
            graph=block_graph,
            mdf_graph=sub_mdf_graph,
            consts=consts,
            port_mapper=port_mapper,
        )
        return sub_mdf_graph

    outputs = [o.unique() for o in node.outputs()]
    inputs = [i.unique() for i in node.inputs()]

    # Get the argument names and parameter names and values for this Node's operation
    if "onnx::" in op:
        arguments, parameters = process_onnx_schema(node, port_mapper)
    else:
        arguments, parameters = process_torch_schema(node, consts, port_mapper)

    mdf_node = Node(id=make_node_id(node))
    for p in parameters:
        mdf_node.parameters.append(Parameter(id=p, value=parameters[p]))

    # Add any output ports
    for o in outputs:
        mdf_node.output_ports.append(
            OutputPort(id=port_mapper.id_to_port(o), value=make_func_id(node))
        )

    # Add any input ports to the node, exclude inputs from constant nodes, these are parameters now
    for inp_i, inp in enumerate(inputs):
        if inp not in consts:
            ip_name = port_mapper.id_to_port(inp)

            # Try to get the shape and type of the input port
            inp_type = node.inputsAt(inp_i).type()
            try:
                shape = str(inp_type.sizes()) if inp_type.sizes() else "(?)"
            except RuntimeError:
                shape = "(?)"

            mdf_node.input_ports.append(
                InputPort(id=ip_name, shape=shape, type=str(inp_type))
            )

    # Add Parameter
    if type(arguments) == list:
        arguments = {"arguments": arguments}
    f = Parameter(id=make_func_id(node), function=op, args=arguments)
    mdf_node.parameters.append(f)

    return mdf_node


def translate_graph(
    graph: Union[torch.Graph, torch.Block],
    mdf_graph: Graph,
    consts: Dict[str, Any],
    port_mapper: "PortMapper",
):
    """
    Go through a :class:`~torch.Graph` or :class:`~torch.Block` and translate the nodes and edges to MDF nodes and
    edges.

    Args:
        graph: The graph to translate.
        mdf_graph: The MDF graph to store the translation into.
        consts: Constant to use for parameters of nodes.
        port_mapper: A port mapper instance to handle translating names.

    Returns:

    """

    # For every node, cache its input edges. This will let us look this up quickly for
    # any node in the loop below.
    node_to_in_edge = {
        node: [i.unique() for i in node.inputs()] for node in graph.nodes()
    }

    for node in graph.nodes():

        mdf_node = torchnode_to_mdfnode(
            node=node, graph=graph, consts=consts, port_mapper=port_mapper
        )

        # If we are excluding this node from the MDF graph, skip it.
        if mdf_node is None:
            continue

        mdf_graph.nodes.append(mdf_node)

        if type(mdf_node) == Graph:
            continue

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


def pytorch_to_mdf(
    model: Union[Callable, torch.nn.Module, torch.ScriptFunction, torch.ScriptModule],
    args: Union[None, torch.Tensor, Tuple[torch.Tensor]] = None,
    example_outputs: Union[None, torch.Tensor, Tuple[torch.Tensor]] = None,
    trace: bool = False,
    use_onnx_ops: bool = True,
) -> Union[Model, Graph]:
    r"""
    Convert a PyTorch model to an MDF model. By default, this function will invoke `torch.jit.script` on the
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

    model_name, graph_name = make_model_graph_name(model)

    # Setup the MDF model and graph
    mdf_model = Model(id=model_name)
    mdf_graph = Graph(id=graph_name)
    mdf_model.graphs.append(mdf_graph)

    # Get all constant nodes in the graph
    consts = get_graph_constants(graph)

    # Get any inputs to the graph, and their debug names. Pass args so we know how
    # many original input arguments the graph has. ONNX lowering from _model_to_graph
    # makes all parameters to the model inputs.
    port_mapper = PortMapper(graph=graph, args=args)

    # Translate the TorchScript graph to and MDF graph object. This could be a recursive call
    translate_graph(
        graph=graph, mdf_graph=mdf_graph, consts=consts, port_mapper=port_mapper
    )

    # Replace in "." for "_" in parameter names. We have done this elsewhere when creating the input ports for these
    # parameters.
    params_dict = {port_mapper.id_to_port(k): v for k, v in params_dict.items()}

    # Set the ONNX opset version
    mdf_model.onnx_opset_version = _export_onnx_opset_version

    return mdf_model, params_dict


if __name__ == "__main__":

    def simple(x, y):
        return x + y

    mdf_model, param_dict = pytorch_to_mdf(
        simple,
        args=(torch.tensor(1.0), torch.tensor(2.0)),
        example_outputs=torch.tensor(0.0),
    )
