r"""
    The main object-oriented implementation of the MDF schema, with each core component of the `MDF specification <../Specification.html>`_
    implemented as a :code:`class`. Instances of these objects can be composed to create a representation of
    an MDF model as Python objects. These models can then be serialized and deserialized to and from JSON or YAML,
    executed via the :mod:`~modeci_mdf.execution_engine` module, or imported and exported to supported external
    environments using the :mod:`~modeci_mdf.interfaces` module.
"""

import collections
import onnx.defs
import sympy

from typing import List, Tuple, Dict, Optional, Set, Any, Union, Optional

# Currently based on elements of NeuroMLlite: https://github.com/NeuroML/NeuroMLlite/tree/master/neuromllite
#  Try: pip install neuromllite
from neuromllite.BaseTypes import Base
from neuromllite.BaseTypes import BaseWithId
from neuromllite import EvaluableExpression

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


class MdfBaseWithId(BaseWithId):
    """Override BaseWithId from nueromllite"""

    def __init__(self, **kwargs):
        self.allowed_fields.update(
            {"metadata": ("Dict of metadata for the model element", dict)}
        )
        super().__init__(**kwargs)


class MdfBase(Base):
    """Override Base from nueromllite"""

    def __init__(self, **kwargs):
        self.allowed_fields.update(
            {"metadata": ("Dict of metadata for the model element", dict)}
        )
        super().__init__(**kwargs)


class Model(MdfBaseWithId):
    r"""The top level construct in MDF is Model, which may contain multiple :class:`.Graph` objects and model attribute(s)

    Args:
        id: A unique identifier for this Model
        format: Information on the version of MDF used in this file
        generating_application: Information on what application generated/saved this file
    """
    _definition = "The top level Model containing _Graph_s consisting of _Node_s connected via _Edge_s."

    def __init__(self, **kwargs):
        self.allowed_children = collections.OrderedDict(
            [("graphs", ("The list of _Graph_s in this Model", Graph))]
        )

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "format",
                    ("Information on the version of MDF used in this file", str),
                ),
                (
                    "generating_application",
                    ("Information on what application generated/saved this file", str),
                ),
            ]
        )
        """The allowed fields for this type"""

        # Removed for now...
        """
        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs = {}
        kwargs["id"] = id
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass"""

        super().__init__(**kwargs)

    @property
    def graphs(self) -> List["Graph"]:
        """The graphs present in the model"""
        return self.__getattr__("graphs")

    def _include_metadata(self):
        """Information on the version of ModECI MDF"""

        from modeci_mdf import MODECI_MDF_VERSION
        from modeci_mdf import __version__

        self.format = "ModECI MDF v%s" % MODECI_MDF_VERSION
        self.generating_application = "Python modeci-mdf v%s" % __version__

    # Overrides BaseWithId.to_json_file
    def to_json_file(self, filename: str, include_metadata: bool = True) -> str:
        """Convert the file in MDF format to JSON format

         .. note::
            JSON is standard file format uses human-readable text to store and transmit data objects consisting of attribute–value pairs and arrays

        Args:
            filename: file in MDF format (.mdf extension)
            include_metadata: Contains contact information, citations, acknowledgements, pointers to sample data,
                              benchmark results, and environments in which the specified model was originally implemented
        Returns:
            The name of the generated JSON file
        """

        if include_metadata:
            self._include_metadata()

        new_file = super().to_json_file(filename)

        return new_file

    # Overrides BaseWithId.to_yaml_file
    def to_yaml_file(self, filename: str, include_metadata: bool = True) -> str:
        """Convert file in MDF format to yaml format

        Args:
            filename: File in MDF format (Filename extension: .mdf )
            include_metadata: Contains contact information, citations, acknowledgements, pointers to sample data,
                              benchmark results, and environments in which the specified model was originally implemented
        Returns:
            The name of the generated yaml file
        """

        if include_metadata:
            self._include_metadata()

        new_file = super().to_yaml_file(filename)

        return new_file

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


class Graph(MdfBaseWithId):
    r"""A directed graph consisting of Node(s) connected via Edge(s)

    Args:
        id: A unique identifier for this Graph
        parameters: Dictionary of global parameters for the Graph
        conditions: The ConditionSet stored as dictionary for scheduling of the Graph
    """
    _definition = "A directed graph consisting of _Node_s connected via _Edge_s."

    def __init__(self, **kwargs):

        self.allowed_children = collections.OrderedDict(
            [
                ("nodes", ("The _Node_s present in the Graph", Node)),
                ("edges", ("The _Edge_s between _Node_s in the Graph", Edge)),
            ]
        )

        self.allowed_fields = collections.OrderedDict(
            [
                ("parameters", ("Dict of global parameters for the Graph", dict)),
                (
                    "conditions",
                    ("The _ConditionSet_ for scheduling of the Graph", ConditionSet),
                ),
            ]
        )
        """The allowed fields for this type"""
        """
        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs = {}
        #kwargs["id"] = id
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass"""

        super().__init__(**kwargs)

    @property
    def nodes(self) -> List["Node"]:
        """Node(s) present in this graph"""
        return self.__getattr__("nodes")

    @property
    def edges(self) -> List["Edge"]:
        """Edge(s) present in this graph"""
        return self.__getattr__("edges")

    def get_node(self, id: str) -> "Node":
        """Retrieve Node object corresponding to the given id

        Args:
            id: Unique identifier of Node object

        Returns:
            Node object if the entered id matches with the id of Node present in the Graph
        """
        for node in self.nodes:
            if id == node.id:
                return node

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


class Node(MdfBaseWithId):
    r"""
    A self contained unit of evaluation receiving input from other nodes on :class:`InputPort`\(s).
    The values from these are processed via a number of :class:`Function`\(s) and one or more final values
    are calculated on the :class:`OutputPort`\(s)

    Args:
        input_ports: Dictionary of the :class:`InputPort` objects in the Node
        parameters: Dictionary of :class:`Parameter`\(s) for the node
        functions: The :class:`Function`\(s) for computation the node
        output_ports: The :class:`OutputPort`\(s) containing evaluated quantities from the node
    """

    _definition = (
        "A self contained unit of evaluation receiving input from other Nodes on _InputPort_s. "
        + "The values from these are processed via a number of Functions and one or more final values "
        "are calculated on the _OutputPort_s "
    )

    def __init__(self, **kwargs):

        self.allowed_children = collections.OrderedDict(
            [
                ("input_ports", ("The _InputPort_s into the Node", InputPort)),
                ("functions", ("The _Function_s for the Node", Function)),
                ("parameters", ("The _Parameter_s of the Node", Parameter)),
                (
                    "output_ports",
                    (
                        "The _OutputPort_s containing evaluated quantities from the Node",
                        OutputPort,
                    ),
                ),
            ]
        )
        """The allowed fields for this type"""

        """
        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs = {}
        kwargs["id"] = id
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass"""

        super().__init__(**kwargs)

    def get_parameter(self, id: str) -> "Parameter":
        r"""Get a parameter by its string :code:`id`

        Args:
            id: The unique string id of the :class:`Parameter`

        Returns:
            The :class:`Parameter` object stored on this node.
        """
        for p in self.parameters:
            if p.id == id:
                return p
        return None

    @property
    def input_ports(self) -> List["InputPort"]:
        r"""
        The InputPort(s) present in the Node

        Returns:
            A list of InputPort(s) at the given Node
        """
        return self.__getattr__("input_ports")

    @property
    def functions(self) -> List["Function"]:
        r"""
        The :class:`Function`\(s) define computation at the :class:`Node`.

        Returns:
            A list of :class:`Function`\ s at the given Node
        """
        return self.__getattr__("functions")

    @property
    def output_ports(self) -> List["OutputPort"]:
        r"""
        The :class:`OutputPort`\(s) present at the Node

        Returns:
            A list of OutputPorts at the given Node
        """
        return self.__getattr__("output_ports")


class Function(MdfBaseWithId):
    r"""A single value which is evaluated as a function of values on :class:`InputPort`\(s) and other Functions

    Args:
        id: The unique (for this Node) id of the function, which will be used in other Functions and the _OutputPort_s
            for its value
        function: Which of the in-build MDF functions (linear etc.) this uses
        args: Dictionary of values for each of the arguments for the Function, e.g. if the in-build function
              is linear(slope),the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}
    """
    _definition = "A single value which is evaluated as a function of values on _InputPort_s and other Functions"

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "function",
                    (
                        "Which of the in-build MDF functions (linear etc.) this uses",
                        str,
                    ),
                ),
                (
                    "value",
                    (
                        "evaluable expression",
                        str,
                    ),
                ),
                (
                    "args",
                    (
                        'Dictionary of values for each of the arguments for the Function, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}',
                        dict,
                    ),
                ),
                (
                    "id",
                    (
                        "The unique (for this _Node_) id of the function, which will be used in other Functions and the _OutputPort_s for its value",
                        str,
                    ),
                ),
            ]
        )
        """The allowed fields for this type"""

        """
        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs = {}
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass"""

        super().__init__(**kwargs)


class InputPort(MdfBaseWithId):
    r"""The :class:`InputPort` is an attribute of a Node which allows external information to be input to the Node

    Args:
        shape: The shape of the input or output of a port. This uses the same syntax as numpy ndarray shapes (e.g., numpy.zeros(<shape>) would produce an array with the correct shape
        type: The data type of the input received at a port or the output sent by a port
    """
    _definition = "The InputPort is an attribute of a _Node_ which allows external information to be input to the _Node_"

    def __init__(
        self,
        id: Optional[str] = None,
        shape: Optional[str] = None,
        type: Optional[str] = None,
        **kwargs,
    ):

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "shape",
                    (
                        "The shape of the variable (note: there is limited support for this so far...)",
                        str,
                    ),
                ),
                (
                    "type",
                    (
                        "The type of the variable (note: there is limited support for this so far ",
                        str,
                    ),
                ),
            ]
        )

        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs["id"] = id
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass

        super().__init__(**kwargs)


class OutputPort(MdfBaseWithId):
    r"""The OutputPort is an attribute of a Node which exports information to another Node connected by an Edge

    Args:
        id: Unique indentifier for the element
        value: The value of the :class:`OutputPort` in terms of the :class:`InputPort` and :class:`Function` values
    """
    _definition = "The OutputPort is an attribute of a _Node_ which exports information to another _Node_ connected by an _Edge_"

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "value",
                    (
                        "The value of the OutputPort in terms of the _InputPort_ and _Function_ values",
                        str,
                    ),
                ),
            ]
        )
        """The allowed fields for this type"""

        """
        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs = {}
        kwargs["id"] = id
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass"""

        super().__init__(**kwargs)


class Parameter(MdfBaseWithId):
    r"""A parameter of the :class:`Node`, which can have a specific value (a constant or a string expression
    referencing other :class:`Parameter`\(s)), be evaluated by an inbuilt function with args, or change from a
    :code:`default_initial_value` with a :code:`time_derivative`.

    Args:
        default_initial_value: The initial value of the parameter
        value: The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values
        time_derivative: How the parameter with time, i.e. ds/dt. Units of time are seconds.
        function: Which of the in-build MDF functions (linear etc.) this uses
        args: Dictionary of values for each of the arguments for the function of the parameter, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}
    """
    _definition = "A Parameter of the _Node_, which can have a specific value (a constant or a string expression referencing other Parameters), be evaluated by an inbuilt function with args, or change from a default_initial_value with a time_derivative"

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "default_initial_value",
                    ("The initial value of the parameter", str),
                ),
                (
                    "value",
                    (
                        "The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values",
                        EvaluableExpression,
                    ),
                ),
                (
                    "time_derivative",
                    (
                        "How the parameter with time, i.e. ds/dt. Units of time are seconds.",
                        str,
                    ),
                ),
                (
                    "function",
                    (
                        "Which of the in-build MDF functions (linear etc.) this uses",
                        str,
                    ),
                ),
                (
                    "args",
                    (
                        'Dictionary of values for each of the arguments for the function of the parameter, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}',
                        dict,
                    ),
                ),
            ]
        )

        super().__init__(**kwargs)

    def is_stateful(self) -> bool:
        """
        Is the parameter stateful?

        A parameter is considered stateful if it has a :code:`time_derivative`, :code:`defualt_initial_value`, or it's
        id is referenced in its value expression.

        Returns:
            :code:`True` if stateful, `False` if not.
        """

        if self.time_derivative is not None:
            return True
        if self.default_initial_value is not None:
            return True
        if self.value is not None and type(self.value) == str:
            param_expr = sympy.simplify(self.value)
            sf = self.id in [str(s) for s in param_expr.free_symbols]
            print(
                "Checking whether %s is stateful, %s: %s"
                % (self, param_expr.free_symbols, sf)
            )
            return sf
        return False


class Edge(MdfBaseWithId):
    r"""An :class:`Edge` is an attribute of a :class:`Graph` that transmits computational results from a sender's
    :class:`OutputPort` to a receiver's :class:`InputPort`.

    Args:
        parameters: Dictionary of parameters for the Edge
        sender: The id of the Node which is the source of the Edge
        receiver: The id of the Node which is the target of the Edge
        sender_port: The id of the OutputPort on the sender Node, whose value should be sent to the receiver_port
        receiver_port: The id of the InputPort on the receiver Node
    """
    _definition = "An Edge is an attribute of a _Graph_ that transmits computational results from a sender's _OutputPort_ to a receiver's _InputPort_"

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict(
            [
                ("parameters", ("Dict of parameters for the Edge", dict)),
                (
                    "sender",
                    ("The id of the _Node_ which is the source of the Edge", str),
                ),
                (
                    "receiver",
                    ("The id of the _Node_ which is the target of the Edge", str),
                ),
                (
                    "sender_port",
                    (
                        "The id of the _OutputPort_ on the sender _Node_, whose value should be sent to the receiver_port",
                        str,
                    ),
                ),
                (
                    "receiver_port",
                    ("The id of the _InputPort_ on the receiver _Node_", str),
                ),
            ]
        )
        """The allowed fields for this type"""

        """
        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs = {}
        kwargs["id"] = id
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass"""

        super().__init__(**kwargs)


class ConditionSet(MdfBase):
    r"""Specifies the non-default pattern of execution of Nodes

    Args:
        node_specific: A dictionary mapping nodes to any non-default run conditions
        termination: A dictionary mapping time scales of model execution to conditions indicating when they end
    """
    _definition = "Specifies the non-default pattern of execution of _Node_s"

    def __init__(
        self,
        node_specific: Optional[Dict[str, "Condition"]] = None,
        termination: Optional[Dict["str", "Condition"]] = None,
    ):

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "node_specific",
                    ("The _Condition_s corresponding to each _Node_", dict),
                ),
                (
                    "termination",
                    ("The _Condition_s that indicate when model execution ends", dict),
                ),
            ]
        )
        """The allowed fields for this type"""

        # FIXME: Reconstruct kwargs as neuromlite expects them
        kwargs = {}
        for f in self.allowed_fields:
            try:
                val = locals()[f]
                if val is not None:
                    kwargs[f] = val
            except KeyError:
                pass

        super().__init__(**kwargs)


class Condition(MdfBase):
    r"""A set of descriptors which specifies conditional execution of Nodes to meet complex execution requirements.

    Args:
        type: The type of :class:`Condition` from the library
        args: The dictionary of arguments needed to evaluate the :class:`Condition`

    """
    _definition = "A set of descriptors which specify conditional execution of _Node_s to meet complex execution requirements"

    def __init__(
        self,
        type: Optional[str] = None,
        **args: Optional[Any],
    ):

        self.allowed_fields = collections.OrderedDict(
            [
                ("type", ("The type of _Condition_ from the library", str)),
                (
                    "args",
                    (
                        "The dictionary of arguments needed to evaluate the _Condition_",
                        dict,
                    ),
                ),
            ]
        )

        super().__init__(type=type, args=args)


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
