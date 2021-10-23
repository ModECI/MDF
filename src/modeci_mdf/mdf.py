r"""
    The main object-oriented implementation of the MDF schema, with each core component of the `MDF specification <../Specification.html>`_
    implemented as a :code:`class`. Instances of these objects can be composed to create a representation of
    an MDF model as Python objects. These models can then be serialized and deserialized to and from JSON or YAML,
    executed via the :mod:`~modeci_mdf.execution_engine` module, or imported and exported to supported external
    environments using the :mod:`~modeci_mdf.interfaces` module.
"""

import collections

import attr
import cattr
import json
import yaml
import onnx.defs
import sympy
import sys

from typing import List, Tuple, Dict, Set, Any, Union, Optional, ForwardRef

from modeci_mdf import MODECI_MDF_VERSION
from modeci_mdf import __version__


__all__ = [
    "Model",
    "Graph",
    "Node",
    "Function",
    "InputPort",
    "OutputPort",
    "Parameter",
    "Edge",
    "ConditionSet",
    "Condition",
]

converter = cattr.Converter()


@attr.define(eq=False)
class MdfBase:
    """
    Base class for all MDF core classes that implements common functionality.

    Args:
        metadata: Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.

    """

    metadata: Optional[Dict[str, Any]] = attr.field(kw_only=True, default=None)

    def to_dict(self):
        """Convert the model to a nested dict structure."""
        return converter.unstructure(self)

    def to_json(self) -> str:
        """
        Convert the model to a JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=4)


@attr.define(eq=False)
class Model(MdfBase):
    r"""
    The top level construct in MDF is Model, which may contain multiple :class:`.Graph` objects and model attribute(s)

    Args:
        id: A unique identifier for this Model
        graphs: The collection of graphs that make up the MDF model.
        format: Information on the version of MDF used in this file
        generating_application: Information on what application generated/saved this file
        onnx_opset_version: The ONNX opset used for any ONNX functions in this model.

    """
    id: str
    graphs: List["Graph"] = attr.field(factory=list)
    format: str = f"ModECI MDF v{MODECI_MDF_VERSION}"
    generating_application: str = f"Python modeci-mdf v{__version__}"
    onnx_opset_version: Optional[str] = None

    def to_json_file(
        self, filename: Optional[str] = None, include_metadata: bool = True
    ) -> str:
        """Convert the MDF model to JSON format and save to a file.

         .. note::
            JSON is standard file format uses human-readable text to store and transmit data objects consisting of attribute–value pairs and arrays

        Args:
            filename: The name of the file to save. If None, use the  (.json extension)
            include_metadata: Contains contact information, citations, acknowledgements, pointers to sample data,
                              benchmark results, and environments in which the specified model was originally implemented
        Returns:
            The name of the generated JSON file
        """

        if filename is None:
            filename = f"{self.id}.json"

        with open(filename, "w") as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

        return filename

    def to_yaml_file(self, filename: str, include_metadata: bool = True) -> str:
        """Convert file in MDF format to yaml format

        Args:
            filename: File in MDF format (Filename extension: .mdf )
            include_metadata: Contains contact information, citations, acknowledgements, pointers to sample data,
                              benchmark results, and environments in which the specified model was originally implemented
        Returns:
            The name of the generated yaml file
        """

        if filename is None:
            filename = f"{self.id}.yaml"

        with open(filename, "w") as outfile:
            yaml.dump(self.to_dict(), outfile)

        return filename

    def to_graph_image(
        self,
        engine: str = "dot",
        output_format: str = "png",
        view_on_render: bool = False,
        level: int = 2,
        filename_root: Optional[str] = None,
        only_warn_on_fail: bool = False,
    ):
        """Convert MDF graph to an image (png or svg) using the Graphviz export

        Args:
            engine: dot or other Graphviz formats
            output_format: e.g. png (default) or svg
            view_on_render: if True, will open generated image in system viewer
            level: 1,2,3, depending on how much detail to include
            filename_root: will change name of file generated to filename_root.png, etc.
            only_warn_on_fail: just give a warning if this fails, e.g. no dot executable. Useful for preventing errors in automated tests
        """
        from modeci_mdf.interfaces.graphviz.exporter import mdf_to_graphviz

        try:
            mdf_to_graphviz(
                self.graphs[0],
                engine=engine,
                output_format=output_format,
                view_on_render=view_on_render,
                level=level,
                filename_root=filename_root,
            )

        except Exception as e:
            if only_warn_on_fail:
                print(
                    "Failure to generate image! Ensure Graphviz executables (dot etc.) are installed on native system. Error: \n%s"
                    % e
                )
            else:
                raise (e)

    @classmethod
    def from_file(cls, filename: str) -> "Model":
        """
        Create a :class:`.Model` from its representation stored in a file. Auto-detect the correct deserialization code
        based on file extension. Currently supported formats are; JSON(.json) and YAML (.yaml or .yml)

        Args:
            filename: The name of the file to load.

        Returns:
            An MDF :class:`.Model` for this file.
        """
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            return cls.from_yaml_file(filename)
        elif filename.endswith(".json"):
            return cls.from_json_file(filename)
        else:
            raise ValueError(
                f"Cannot auto-detect MDF serialization format from filename ({filename}). The filename "
                f"must have one of the following extensions: .json, .yml, or .yaml."
            )

    @classmethod
    def from_json_file(cls, filename: str) -> "Model":
        """
        Create a :class:`.Model` from its JSON representation stored in a file.

        Args:
            filename: The file from which to load the JSON data.

        Returns:
            An MDF :class:`.Model` for this JSON
        """
        with open(filename) as infile:
            d = json.load(infile)
            return converter.structure(d, cls)

    @classmethod
    def from_yaml_file(cls, filename: str) -> "Model":
        """
        Create a :class:`.Model` from its YAML representation stored in a file.

        Args:
            filename: The file from which to load the YAML data.

        Returns:
            An MDF :class:`.Model` for this YAML
        """
        with open(filename) as infile:
            d = yaml.load(infile)
            return converter.structure(d, cls)


@attr.define(eq=False)
class Graph(MdfBase):
    r"""
    A directed graph consisting of Node(s) connected via Edge(s)

    Args:
        id: A unique identifier for this Graph
        parameters: Dictionary of global parameters for the Graph
        conditions: The ConditionSet stored as dictionary for scheduling of the Graph
    """
    id: str
    nodes: List["Node"] = attr.field(factory=list)
    edges: List["Edge"] = attr.field(factory=list)
    parameters: Optional[Dict[str, Any]] = None
    conditions: Optional["ConditionSet"] = None

    def get_node(self, id: str) -> Union["Node", None]:
        """Retrieve Node object corresponding to the given id

        Args:
            id: Unique identifier of Node object

        Returns:
            :class:`Node` object if the entered :code:`id` matches with the :code:`id` of node present in the
            :class:`~Graph`. :code:`None` if a node is not found with that id .
        """
        for node in self.nodes:
            if id == node.id:
                return node

        return None

    @property
    def dependency_dict(self) -> Dict["Node", Set["Node"]]:
        """Returns the dependency among nodes as dictionary

        Key: receiver, Value: Set of senders imparting information to the receiver

        Returns:
            Returns the dependency dictionary
        """
        # assumes no cycles, need to develop a way to prune if cyclic
        # graphs are to be supported
        dependencies = {n: set() for n in self.nodes}

        for edge in self.edges:
            sender = self.get_node(edge.sender)
            receiver = self.get_node(edge.receiver)

            dependencies[receiver].add(sender)

        return dependencies

    @property
    def inputs(self: "Graph") -> List[Tuple["Node", "InputPort"]]:
        """
        Enumerate all Node-InputPort pairs that specify no incoming edge.
        These are input ports for the graph itself and must be provided values to evaluate

        Returns:
            A list of Node, InputPort tuples
        """

        # Get all input ports
        all_ips = [(node.id, ip.id) for node in self.nodes for ip in node.input_ports]

        # Get all receiver ports
        all_receiver_ports = {(e.receiver, e.receiver_port) for e in self.edges}

        # Find any input ports that aren't receiving values from an edge
        return list(filter(lambda x: x not in all_receiver_ports, all_ips))


@attr.define(eq=False)
class Node(MdfBase):
    r"""
    A self contained unit of evaluation receiving input from other nodes on :class:`InputPort`\(s).
    The values from these are processed via a number of :class:`Function`\(s) and one or more final values
    are calculated on the :class:`OutputPort`\(s)

    Args:
        id: A unique identifier for the node.
        input_ports: Dictionary of the :class:`InputPort` objects in the Node
        parameters: Dictionary of :class:`Parameter`\(s) for the node
        functions: The :class:`Function`\(s) for computation the node
        output_ports: The :class:`OutputPort`\(s) containing evaluated quantities from the node
    """

    id: str
    input_ports: List["InputPort"] = attr.field(factory=list)
    functions: List["Function"] = attr.field(factory=list)
    parameters: List["Parameter"] = attr.field(factory=list)
    output_ports: List["Parameter"] = attr.field(factory=list)

    def get_parameter(self, id: str) -> Union["Parameter", None]:
        r"""Get a parameter by its string :code:`id`

        Args:
            id: The unique string id of the :class:`Parameter`

        Returns:
            The :class:`Parameter` object stored on this node. :code:`None` if not found.
        """
        for p in self.parameters:
            if p.id == id:
                return p

        return None


@attr.define(eq=False)
class Function(MdfBase):
    r"""
    A single value which is evaluated as a function of values on :class:`InputPort`\(s) and other Functions

    Args:
        id: The unique (for this Node) id of the function, which will be used in other :class:`~Function`s and
            the :class:`~OutputPort`s for its value
        function: Which of the in-build MDF functions (:code:`linear`, etc.). See supported functions:
            https://mdf.readthedocs.io/en/latest/api/MDF_function_specifications.html
        args: Dictionary of values for each of the arguments for the Function, e.g. if the in-built function
              is linear(slope),the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}
    """
    id: str
    function: Optional[str] = None
    args: Optional[Dict[str, Any]] = None


@attr.define(eq=False)
class InputPort(MdfBase):
    r"""
    The :class:`InputPort` is an attribute of a Node which allows external information to be input to the Node

    Args:
        id: The unique (for this Node) id of the input port,
        shape: The shape of the input or output of a port. This uses the same syntax as numpy ndarray shapes (e.g., numpy.zeros(<shape>) would produce an array with the correct shape
        type: The data type of the input received at a port or the output sent by a port
    """
    id: str
    shape: Optional[Tuple] = None
    type: Optional[str] = None


@attr.define(eq=False)
class OutputPort(MdfBase):
    r"""
    The :class:`OutputPort` is an attribute of a :class:`Node` which exports information to another :class:`Node`
    connected by an :class:`Edge`

    Args:
        id: Unique identifier for the output port.
        value: The value of the :class:`OutputPort` in terms of the :class:`InputPort`, :class:`Function` values, and
            :class:`Parameter` values.
    """
    id: str
    value: Optional[str] = None


@attr.define(eq=False)
class Parameter(MdfBase):
    r"""
    A parameter of the :class:`Node`, which can have a specific value (a constant or a string expression
    referencing other :class:`Parameter`\(s)), be evaluated by an inbuilt function with args, or change from a
    :code:`default_initial_value` with a :code:`time_derivative`.

    Args:
        value: The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values
        default_initial_value: The initial value of the parameter, only used when parameter is stateful.
        time_derivative: How the parameter changes with time, i.e. ds/dt. Units of time are seconds.
        function: Which of the in-build MDF functions (linear etc.) this uses, See
        https://mdf.readthedocs.io/en/latest/api/MDF_function_specifications.html
        args: Dictionary of values for each of the arguments for the function of the parameter,
            e.g. if the in-build function is :code:`linear(slope)`, the args here could be :code:`{"slope": 3}` or
            :code:`{"slope": "input_port_0 + 2"}`
    """

    id: str
    value: Optional[str] = None
    default_initial_value: Optional[str] = None
    time_derivative: Optional[str] = None
    function: Optional[str] = None
    args: Optional[Dict[str, Any]] = None

    def is_stateful(self) -> bool:
        """
        Is the parameter stateful?

        A parameter is considered stateful if it has a :code:`time_derivative`, :code:`defualt_initial_value`, or it's
        id is referenced in its value expression.

        Returns:
            :code:`True` if stateful, `False` if not.
        """
        from modeci_mdf.execution_engine import parse_str_as_list

        if self.time_derivative is not None:
            return True
        if self.default_initial_value is not None:
            return True
        if self.value is not None and type(self.value) == str:
            # If we are dealing with a list of symbols, each must treated separately
            if self.value[0] == "[" and self.value[-1] == "]":
                # Use the Python interpreter to parse this into a List[str]
                arg_expr_list = parse_str_as_list(self.value)
            else:
                arg_expr_list = [self.value]

            req_vars = []

            for e in arg_expr_list:
                param_expr = sympy.simplify(e)
                req_vars.extend([str(s) for s in param_expr.free_symbols])
            sf = self.id in req_vars
            print(
                "Checking whether %s is stateful, %s: %s"
                % (self, param_expr.free_symbols, sf)
            )
            return sf

        return False


@attr.define(eq=False)
class Edge(MdfBase):
    r"""
    An :class:`Edge` is an attribute of a :class:`Graph` that transmits computational results from a sender's
    :class:`OutputPort` to a receiver's :class:`InputPort`.

    Args:
        parameters: Dictionary of parameters for the Edge
        sender: The id of the Node which is the source of the Edge
        receiver: The id of the Node which is the target of the Edge
        sender_port: The id of the OutputPort on the sender Node, whose value should be sent to the receiver_port
        receiver_port: The id of the InputPort on the receiver Node
    """
    id: str
    sender: str
    receiver: str
    sender_port: str
    receiver_port: str
    parameters: Optional[Dict[str, Any]] = None


@attr.define(eq=False)
class ConditionSet(MdfBase):
    r"""
    Specifies the non-default pattern of execution of Nodes

    Args:
        node_specific: A dictionary mapping nodes to any non-default run conditions
        termination: A dictionary mapping time scales of model execution to conditions indicating when they end
    """
    node_specific: Optional[Dict[str, "Condition"]] = None
    termination: Optional[Dict[str, "Condition"]] = None


@attr.define(eq=False)
class Condition(MdfBase):
    r"""A set of descriptors which specifies conditional execution of Nodes to meet complex execution requirements.

    Args:
        type: The type of :class:`Condition` from the library
        kwargs: The dictionary of keyword arguments needed to evaluate the :class:`Condition`

    """
    type: str = attr.field()
    kwargs: Optional[Dict[str, Any]] = attr.field()

    def __init__(
        self,
        type: Optional[str] = None,
        **kwargs: Optional[Dict[str, Any]],
    ):
        # We need to right out own __init__ in this case because the current API for Condition requires saving
        # kwargs.
        self.__attrs_init__(type, kwargs)


# We need to register some structure hooks for the ForwardRef type annotations we have used.
# Haven't figured out a slicker way to do this. Cleanest way is to probably avoid most of
# the ForwardRef by defining the classes in opposite order.
# for type_name in __all__:
#     converter.register_structure_hook(ForwardRef(type_name),
#                                       lambda d, t: converter.structure(d, getattr(sys.modules[__name__], type_name)))

converter.register_structure_hook(
    ForwardRef("Model"), lambda d, t: converter.structure(d, Model)
)
converter.register_structure_hook(
    ForwardRef("Graph"), lambda d, t: converter.structure(d, Graph)
)
converter.register_structure_hook(
    ForwardRef("Node"), lambda d, t: converter.structure(d, Node)
)
converter.register_structure_hook(
    ForwardRef("Function"), lambda d, t: converter.structure(d, Function)
)
converter.register_structure_hook(
    ForwardRef("InputPort"), lambda d, t: converter.structure(d, InputPort)
)
converter.register_structure_hook(
    ForwardRef("OutputPort"), lambda d, t: converter.structure(d, OutputPort)
)
converter.register_structure_hook(
    ForwardRef("Parameter"), lambda d, t: converter.structure(d, Parameter)
)
converter.register_structure_hook(
    ForwardRef("Edge"), lambda d, t: converter.structure(d, Edge)
)
converter.register_structure_hook(
    ForwardRef("ConditionSet"), lambda d, t: converter.structure(d, ConditionSet)
)
converter.register_structure_hook(
    ForwardRef("Condition"), lambda d, t: converter.structure(d, Condition)
)

if __name__ == "__main__":
    model = Model(id="MyModel")
    mod_graph0 = Graph(id="Test", parameters={"speed": 4})
    model.graphs.append(mod_graph0)

    node = Node(id="N0")
    node.parameters.append(Parameter(id="rate", value=5))

    mod_graph0.nodes.append(node)

    print(mod_graph0)
    print("------------------")
    print(mod_graph0.to_json())
    print("==================")
    model.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=3,
        filename_root="test",
        only_warn_on_fail=True,
    )
