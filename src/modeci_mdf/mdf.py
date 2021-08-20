"""
    Defines the structure of ModECI MDF - Work in progress!!!
"""

import collections
import onnx.defs

from typing import List, Tuple, Dict, Optional, Set

# Currently based on elements of NeuroMLlite: https://github.com/NeuroML/NeuroMLlite/tree/master/neuromllite
#  Try: pip install neuromllite
from neuromllite.BaseTypes import Base
from neuromllite.BaseTypes import BaseWithId
from neuromllite import EvaluableExpression

class MdfBaseWithId(BaseWithId):
    def __init__(self, **kwargs):
        self.allowed_fields.update({'metadata':("Dict of metadata for the Node", dict)})
        super().__init__(**kwargs)

class MdfBase(Base):
    def __init__(self, **kwargs):
        self.allowed_fields.update({'metadata':("Dict of metadata for the Node", dict)})
        super().__init__(**kwargs)


class Model(MdfBaseWithId):
    _definition = "The top level Model containing _Graph_s consisting of _Node_s connected via _Edge_s."

    def __init__(
        self,
        id: Optional[str] = None,
        format: Optional[str] = None,
        generating_application: Optional[str] = None,
        metadata: Optional[dict] = None
    ):
        """The top level construct in MDF is Model which consists of Graph's and model attributed
        Args:
            id: A unique identifier for this Model.
            format: Information on the version of MDF used in this file
            generating_application: Information on what application generated/saved this file
        """

        kwargs = {}
        if id is not None:
            kwargs["id"] = id
        if format is not None:
            kwargs["format"] = format
        if generating_application is not None:
            kwargs["generating_application"] = generating_application
        if metadata is not None:
            kwargs["metadata"] = metadata

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

        # Inheriting the init method from superclass BaseWithId
        super().__init__(**kwargs)

    @property
    def graphs(self) -> List["Graph"]:
        """
        The graphs this model contains.
        """
        return self.__getattr__("graphs")

    def _include_metadata(self):
        """Information on the version of ModECI_MDF"""

        from modeci_mdf import MODECI_MDF_VERSION
        from modeci_mdf import __version__

        self.format = "ModECI MDF v%s" % MODECI_MDF_VERSION
        self.generating_application = "Python modeci-mdf v%s" % __version__

    # Overrides BaseWithId.to_json_file
    def to_json_file(self, filename, include_metadata=True) -> str:
        """Convert MDF object to json file
        Args:
            filename: file in MDF format (.mdf extension)
            include_metadata: Contains contact information, citations, acknowledgements, pointers to sample data,
                              benchmark results, and environments in which the specified model was originally implemented
        Returns:
            The name of the JSON file generated.
        """

        if include_metadata:
            self._include_metadata()

        new_file = super().to_json_file(filename)

        return new_file

    # Overrides BaseWithId.to_yaml_file
    def to_yaml_file(self, filename, include_metadata=True):
        """Convert MDF object to yaml format
        Args:
            filename: file in MDF format (Filename extension: .mdf )
            include_metadata: Contains contact information, citations, acknowledgements, pointers to sample data,
                              benchmark results, and environments in which the specified model was originally implemented
        Returns:
            file in yaml format
        """

        if include_metadata:
            self._include_metadata()

        new_file = super().to_yaml_file(filename)

    def to_graph_image(
        self,
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=2,
        filename_root=None,
        only_warn_on_fail=False,
    ):
        """Convert MDF graph to an image (png or svg) using the Graphviz export
        Args:
            engine: dot or other Graphviz formats
            output_format: e.g. png (default) or svg
            view_on_render: if True, will open generated image in system viewer
            level: 1,2,3, depending on how much detail to include
            filename_root: will change name of file generated to filename_root.png, etc.
            only_warn_on_fail: just give a warning if this fails, e.g. no dot executable. Useful for preventing erros in automated tests
        """
        from modeci_mdf.interfaces.graphviz.importer import mdf_to_graphviz

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
    _definition = "A directed graph consisting of _Node_s connected via _Edge_s."

    def __init__(self, **kwargs):
        """A directed graph consisting of _Node_s connected via _Edge_s
        Args:
            nodes: Dictionary of Node objects in the Graph
            edges: Dictionary of Edge objects in the Graph
            parameters: Dictionary of global parameters for the Graph
            conditions: The _ConditionSet_ stored as dictionary for scheduling of the Graph
        """

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
                    ("The _ConditionSet_ for scheduling of the Graph", dict),
                ),
            ]
        )

        super().__init__(**kwargs)

    def get_node(self, id):
        """Retrieve Node object corresponding to the given id
        Args:
            id: Unique identifier of Node object
        Returns:
            Node object if the entered id matches with the id of node present in the graph
        """
        for node in self.nodes:
            if id == node.id:
                return node

    @property
    def dependency_dict(self) -> Dict["Node", Set["Node"]]:
        """Returns the dependency among nodes as dictionary
        Key: receiver, Value: set of senders imparting information to the receiver
        Returns:
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
        """Enumerate all Node, InputPort pairs that specify no incoming edge. These are input ports for the graph itself and must be provided values to evaluate
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
    _definition = (
        "A self contained unit of evaluation receiving input from other Nodes on _InputPort_s. "
        + "The values from these are processed via a number of _Function_s and one or more final values "
        "are calculated on the _OutputPort_s "
    )

    def __init__(self, **kwargs):
        """A self contained unit of evaluation receiving input from other Nodes on _InputPort_s.
        The values from these are processed via a number of _Function_s and one or more final values
        are calculated on the _OutputPort_
        Args:
            input_ports (obj): Dictionary of the InputPort objects in the Node
            functions (obj): The _Function_s for computation the Node
            states (obj): The _State_s of the Node
            output_ports (obj): The _OutputPort_s containing evaluated quantities from the Node
            parameters : Dictionary of parameters for the Node
        """

        self.allowed_children = collections.OrderedDict(
            [
                ("input_ports", ("The _InputPort_s into the Node", InputPort)),
                ("functions", ("The _Function_s for the Node", Function)),
                ("states", ("The _State_s of the Node", State)),
                ("stateful_parameters",("The Stateful Parameter s of the Node",Stateful_Parameter)),
                (
                    "output_ports",
                    (
                        "The _OutputPort_s containing evaluated quantities from the Node",
                        OutputPort,
                    ),
                ),
            ]
        )

        self.allowed_fields = collections.OrderedDict(
            [
                ("parameters", ("Dict of parameters for the Node", dict))
            ]
        )

        super().__init__(**kwargs)


class Function(MdfBaseWithId):
    _definition = "A single value which is evaluated as a function of values on _InputPort_s and other Functions"

    def __init__(self, **kwargs):
        """A single value which is evaluated as a function of values on _InputPort_s and other Functions
        Args:
            function (str): Which of the in-build MDF functions (linear etc.) this uses
            args : Dictionary of values for each of the arguments for the Function, e.g. if the in-build function
                  is linear(slope),the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}
            id (str): The unique (for this _Node_) id of the function, which will be used in other Functions and the _OutputPort_s
                for its value
        """

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
                    "args",
                    (
                        'Dictionary of values for each of the arguments for the Function, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}',
                        dict,
                    ),
                ),
            ]
        )

        super().__init__(**kwargs)

        self.allowed_fields["id"] = (
            "The unique (for this _Node_) id of the function, which will be used in other Functions and the _OutputPort_s for its value",
            str,
        )


class InputPort(MdfBaseWithId):
    def __init__(self, **kwargs):
        """The InputPort is an attribute of a Node which imports information to the Node object
        Args:
            shape (str): The shape of the input or output of a port. This uses the same syntax as numpy ndarray shapes (e.g., numpy.zeros(<shape>) would produce an array with the correct shape
            type (str): The data type of the input received at a port or the output sent by a port
        """
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
                )
            ]
        )

        super().__init__(**kwargs)


class OutputPort(MdfBaseWithId):
    def __init__(self, **kwargs):
        """The OutputPort is an attribute of a Node which exports information to the dependent Node object
        Args:
            value (str): The value of the OutputPort in terms of the _InputPort_ and _Function_ values
        """

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

        super().__init__(**kwargs)


class State(MdfBaseWithId):
    _definition = "A state variable of a _Node_, i.e. has a value that persists between evaluations of the _Node_."

    def __init__(self, **kwargs):
        """A state variable of a _Node_, i.e. has a value that persists between evaluations of the _Node_
        Args:
            default_initial_value (str): The initial value of the state variable
            value (str): The next value of the state variable, in terms of the inputs, functions and PREVIOUS state values
            time_derivative (str): How the state varies with time, i.e. ds/dt. Unit of time is second
        """

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "default_initial_value",
                    ("The initial value of the state variable", str),
                ),
                (
                    "value",
                    (
                        "The next value of the state variable, in terms of the inputs, functions and PREVIOUS state values",
                        str,
                    ),
                ),
                (
                    "time_derivative",
                    (
                        "How the state varies with time, i.e. ds/dt. Units of time are seconds.",
                        str,
                    ),
                ),
            ]
        )

        super().__init__(**kwargs)


class Stateful_Parameter(BaseWithId):
    _definition = "A stateful parameter of a _Node_, i.e. has a value that updates by functions between evaluations of the _Node_."

    def __init__(self, **kwargs):
        """A stateful parameter of a _Node_, i.e. has a value that updates by functions between evaluations of the _Node_

        Args:
            default_initial_value (str): The initial value of the stateful parameter
            value (str): The next value of the stateful parameter, in terms of the inputs, functions

        """

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "default_initial_value",
                    ("The initial value of the stateful parameter", str),
                ),
                (
                    "value",
                    (
                        "The next value of the stateful parameter, in terms of the inputs, functions",
                        str,
                    ),
                ),

            ]
        )

        super().__init__(**kwargs)



class Edge(MdfBaseWithId):
    def __init__(self, **kwargs):
        """Edge is an attribute of Graph object that transmits computational results from sender_port to receiver port
        Args:
            parameters: Dictionary of parameters for the Edge
            sender (str): The id of the _Node_ which is the source of the Edge
            receiver (str): The id of the _Node_ which is the target of the Edge
            sender_port (str): The id of the _OutputPort_ on the sender _Node_, whose value should be sent to the receiver_port
            receiver_port (str): The id of the _InputPort_ on the receiver _Node_
        """

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

        super().__init__(**kwargs)


class ConditionSet(MdfBase):
    def __init__(self, **kwargs):
        """Specify the non-default pattern of execution
        Args:
            node_specific: A dictionary mapping nodes to any non-default run conditions
            termination: A dictionary mapping time scales of model execution to conditions indicating when they end
        """

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

        super().__init__(**kwargs)


class Condition(MdfBase):
    def __init__(self, type=None, **kwargs):
        """A set of descriptors which specifies conditional execution of Nodes to meet complex execution requirements
        Args:
            type (str): The type of _Condition_ from the library
            args: The dictionary of arguments needed to evaluate the _Condition_
        """

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

        super().__init__(type=type, args=kwargs)


if __name__ == "__main__":
    model = Model(id="MyModel")
    mod_graph0 = Graph(id="Test", parameters={"speed": 4})
    model.graphs.append(mod_graph0)

    node = Node(id="N0", parameters={"rate": 5})

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
