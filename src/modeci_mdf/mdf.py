import collections

"""
    Defines the structure of ModECI MDF - Work in progress!!!
"""


# Currently based on elements of NeuroMLlite: https://github.com/NeuroML/NeuroMLlite/tree/master/neuromllite
#  Try: pip install neuromllite
from neuromllite.BaseTypes import Base
from neuromllite.BaseTypes import BaseWithId
from neuromllite import EvaluableExpression


class Model(BaseWithId):

    _definition = "The top level Model containing a number of _Graph_s of _Node_s connected via _Edge_s."

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

        super().__init__(**kwargs)

    def _include_metadata(self):

        from modeci_mdf import MODECI_MDF_VERSION
        from modeci_mdf import __version__

        self.format = "ModECI MDF v%s" % MODECI_MDF_VERSION
        self.generating_application = "Python modeci-mdf v%s" % __version__

    # Overrides BaseWithId.to_json_file
    def to_json_file(self, filename, include_metadata=True):

        if include_metadata:
            self._include_metadata()

        new_file = super().to_json_file(filename)

    # Overrides BaseWithId.to_yaml_file
    def to_yaml_file(self, filename, include_metadata=True):

        if include_metadata:
            self._include_metadata()

        new_file = super().to_yaml_file(filename)


class Graph(BaseWithId):

    _definition = "A directed graph of _Node_s connected via _Edge_s."

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
                    ("The _ConditionSet_ for scheduling of the Graph", dict),
                ),
            ]
        )

        super().__init__(**kwargs)

    def get_node(self, id):
        for node in self.nodes:
            if id == node.id:
                return node

    @property
    def dependency_dict(self):
        # assumes no cycles, need to develop a way to prune if cyclic
        # graphs are to be supported
        dependencies = {n: set() for n in self.nodes}

        for edge in self.edges:
            sender = self.get_node(edge.sender)
            receiver = self.get_node(edge.receiver)

            dependencies[receiver].add(sender)

        return dependencies


class Node(BaseWithId):

    _definition = (
        "A self contained unit of evaluation recieving input from other Nodes on _InputPort_s. "
        + "The values from these are processed via a number of _Function_s and one or more final values are calculated on the _OutputPort_s"
    )

    def __init__(self, **kwargs):

        self.allowed_children = collections.OrderedDict(
            [
                ("input_ports", ("The _InputPort_s into the Node", InputPort)),
                ("functions", ("The _Function_s for the Node", Function)),
                ("states", ("The _State_s of the Node", State)),
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
            [("parameters", ("Dict of parameters for the Node", dict))]
        )

        super().__init__(**kwargs)


class Function(BaseWithId):

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


class InputPort(BaseWithId):
    def __init__(self, **kwargs):
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

        super().__init__(**kwargs)


class OutputPort(BaseWithId):
    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict(
            [
                (
                    "value",
                    (
                        "The value of the OutputPort in terms of the _InputPort_ and _Function_ values",
                        str,
                    ),
                )
            ]
        )

        super().__init__(**kwargs)


class State(BaseWithId):

    _definition = "A state variable of a _Node_, i.e. has a value that persists between evaluations of the _Node_."

    def __init__(self, **kwargs):

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


class Edge(BaseWithId):
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
                    ("The id of the _InputPort_ on the sender _Node_", str),
                ),
            ]
        )

        super().__init__(**kwargs)


class ConditionSet(Base):
    def __init__(self, **kwargs):

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


class Condition(Base):
    def __init__(self, type=None, **kwargs):

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

    mod_graph0 = Graph(id="Test", parameters={"speed": 4})

    node = Node(id="N0", parameters={"rate": 5})

    mod_graph0.nodes.append(node)

    print(mod_graph0)
    print("------------------")
    print(mod_graph0.to_json())
    print("==================")
