import numpy as np

from modeci_mdf.mdf import (
    Model,
    Graph,
    Node,
    OutputPort,
    Function,
    Condition,
    ConditionSet,
    Parameter,
    Edge,
    InputPort,
)

from modeci_mdf.utils import load_mdf

import os
import pytest


def test_model_init_kwargs():
    m = Model(
        id="Test",
        format="test_format",
        generating_application="test_application",
        metadata={"info": "test_metadata"},
    )
    assert m.format == "test_format"
    assert m.generating_application == "test_application"
    assert m.metadata == {"info": "test_metadata"}
    assert m.id == "Test"


def test_graph_init_kwargs():
    g = Graph(
        id="Test_Graph", parameters={"test_parameters": 1}, conditions=ConditionSet()
    )
    assert g.parameters == {"test_parameters": 1}
    assert str(g.conditions) == str(ConditionSet())


def test_Node_init_kwargs():
    n = Node(id="test_node")
    print(n)
    print(n.id)
    assert n.id == "test_node"


def test_Function_init_kwargs():
    f = Function(id="Test_Function", function="Test_function", args={"Test_arg": 1})
    assert f.function == "Test_function"
    assert f.args == {"Test_arg": 1}


# def test_Function_init_kwargs():
#     f = Function(
#         id="Test_Function", function={"Test_function": "Linear"}, args={"Test_arg": 1}
#     )
#     assert f.function == {"Test_function": "Linear"}
#     # assert f.args == {"Test_arg": 1}


def test_InputPort_init_kwargs():
    ip = InputPort(id="Test_InputPort", shape=(1, 2), type="Test_type")
    assert ip.shape == (1, 2)
    assert ip.type == "Test_type"


def test_OutputPort_init_kwargs():
    op = OutputPort(id="test_OutputPort", value="test_value")
    assert op.value == "test_value"


def test_Edge_init_kwargs():
    e = Edge(
        id="test_Edge",
        parameters={"test_parameters": 3},
        sender="test_sender",
        receiver="test_receiver",
        sender_port="test_sender_port",
        receiver_port="test_receiver_port",
    )
    assert e.parameters == {"test_parameters": 3}
    assert e.sender == "test_sender"
    assert e.receiver == "test_receiver"
    assert e.sender_port == "test_sender_port"
    assert e.receiver_port == "test_receiver_port"
    assert e.id == "test_Edge"


def test_ConditionSet_init_kwargs():
    CS = ConditionSet(
        node_specific={"test_node_specific": 1}, termination={"test_termination": 3}
    )
    assert CS.node_specific == {"test_node_specific": 1}
    assert CS.termination == {"test_termination": 3}


def test_Condition_init_kwargs():
    """Check the working of Condition"""
    C = Condition(type="test_type", n="test_n", dependency="test_dependency")
    assert C.type == "test_type"
    assert C.kwargs == {"n": "test_n", "dependency": "test_dependency"}


def test_Condition_init_kwargs():
    C = Condition(type="test_type", n="test_n", dependencies="test_dependencies")
    assert C.type == "test_type"
    assert C.kwargs == {"n": "test_n", "dependencies": "test_dependencies"}


def test_model_graph_to_json():
    """Check if dumping a model to a simple JSON string works."""

    mod_graph0 = Graph(
        id="Test", parameters={"speed": 4}, metadata={"info": "mdf_model"}
    )

    node = Node(id="N0", metadata={"info": "mdf_Node"})
    node.parameters.append(Parameter(id="rate", value=5))

    node1 = Node(id="test_node", metadata={"info": "mdf_Node2"})
    node1.parameters.append(Parameter(id="level", value=3))

    condition = Condition(type="Always")
    condition.metadata = {"info": "mdf_condition"}

    mod_graph0.conditions = ConditionSet(node_specific={node1.id: condition})

    mod_graph0.nodes.append(node)
    mod_graph0.nodes.append(node1)

    # Export to JSON and see if we can load back in
    import json

    d = json.loads(mod_graph0.to_json())


def test_no_input_ports_to_json(tmpdir):
    """Test the edge case of exporting a model to JSON when it has a node with no input ports"""

    mod = Model(id="ABCD")
    mod_graph = Graph(id="abcd_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0")
    input_node.parameters.append(Parameter(id="input_level", value=10.0))
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    tmpfile = f"{tmpdir}/test.json"
    mod.to_json_file(tmpfile)

    lmod = load_mdf(tmpfile)

    assert lmod.graphs[0].nodes[0].parameters[0].value == 10.0


def test_include_metadata_to_json(tmpdir):
    """
    Test for serialization
    """

    mod = Model(id="ABCD", metadata={"info": "model_test"})
    mod_graph = Graph(
        id="abcd_example", metadata={"info": {"graph_test": {"environment_x": "xyz"}}}
    )
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0", metadata={"color": ".8 0 .8"})
    input_node.parameters.append(Parameter(id="input_level", value=10.0))
    op1 = OutputPort(id="out_port", metadata={"info": "value at OutputPort"})
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    tmpfile = f"{tmpdir}/test.json"
    mod.to_json_file(tmpfile)

    lmod = load_mdf(tmpfile)
    lmod.graphs[0].metadata

    assert lmod.graphs[0].metadata == {"info": {"graph_test": {"environment_x": "xyz"}}}
    assert lmod.graphs[0].get_node("input0").metadata == {"color": ".8 0 .8"}
    assert lmod.graphs[0].get_node("input0").output_ports[0].metadata == {
        "info": "value at OutputPort"
    }


def test_node_params_empty():
    """
    Test whether we get a TypeError when passing no id for a node
    """
    with pytest.raises(TypeError):
        Node()


def test_metadata_dict():
    """
    Test whether we get a serialization error when passing anything else from a dictionary
    """
    with pytest.raises(Exception):
        Graph(id="n0", metadata="info").to_json()


def test_param_args_empty_dict():
    """
    Test whether we don't a serialization error when passing empty dicts to Parameter args
    """
    Parameter(id="noargs", args={}).to_json()


def test_graph_inputs():
    r"""Test whether we can retrieve graph input node\ports via the inputs property."""
    mod = Model(id="ABCD")
    mod_graph = Graph(id="abcd_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0")
    input_node.parameters.append(Parameter(id="input_level", value=10.0))
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)


def test_graph_inputs_none(simple_model_mdf):
    """Test that the simple model with no input ports used has no graph inputs"""
    assert len(simple_model_mdf.graphs[0].inputs) == 0


def test_graph_types(tmpdir):
    r"""
    Test whether types saved in parameters are the same after reloading
    """
    mod = Model(id="Test0")
    mod_graph = Graph(id="test_example")
    mod.graphs.append(mod_graph)
    node0 = Node(id="node0")
    mod_graph.nodes.append(node0)

    p_int = 2
    node0.parameters.append(Parameter(id="p_int", value=p_int))
    p_float = 2.0
    node0.parameters.append(Parameter(id="p_float", value=p_float))
    p_bool = False
    node0.parameters.append(Parameter(id="p_bool", value=p_bool))
    p_str = "p_int + p_float"
    node0.parameters.append(Parameter(id="p_str", value=p_str))
    p_str2 = "2"
    node0.parameters.append(Parameter(id="p_str2", value=p_str2))
    p_list = ["2", 2, "two"]
    node0.parameters.append(Parameter(id="p_list", value=p_list))

    p_dict = {"a": 3, "w": {"b": 3, "x": True, "y": [2, 2, 2, 2]}}
    node0.parameters.append(Parameter(id="p_dict", value=p_dict))

    p_dict_tuple = {"y": (4, 44)}  # will change to {'y': [4, 44]}
    node0.parameters.append(Parameter(id="p_dict_tuple", value=p_dict_tuple))

    print(mod)
    tmpfile = f"{tmpdir}/test.json"
    mod.to_json_file(tmpfile)
    mod_graph2 = load_mdf(tmpfile)
    print(f"Saved to {tmpfile}: {mod_graph2}")
    new_node0 = mod_graph2.graphs[0].nodes[0]

    for p in [p.id for p in node0.parameters]:
        print(
            "Testing {}, is {} = {}?".format(
                p, new_node0.get_parameter(p).value, eval(p)
            )
        )

        assert type(new_node0.get_parameter(p).value) == type(eval(p))
        # Type will be same for tuple containing dict, but tuple will have been converetd to dict...
        if not p == "p_dict_tuple":
            assert new_node0.get_parameter(p).value == eval(p)


def test_arrays(tmpdir):
    """Test whether arrays are serialized and deserialized properly."""

    mod = Model(id="Arrays")
    mod_graph = Graph(id="array_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input_node")

    input_node.parameters.append(Parameter(id="input_level", value=[[1, 2.0], [3, 4]]))

    op1 = OutputPort(id="out_port", value="input_level")
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    middle_node = Node(id="middle_node")
    middle_node.parameters.append(Parameter(id="slope", value=0.5))
    middle_node.parameters.append(
        Parameter(id="intercept", value=np.array([[0, 1.0], [2, 2]]))
    )

    ip1 = InputPort(id="input_port1")
    middle_node.input_ports.append(ip1)
    mod_graph.nodes.append(middle_node)

    f1 = Parameter(
        id="linear_1",
        function="linear",
        args={"variable0": ip1.id, "slope": "slope", "intercept": "intercept"},
    )
    middle_node.parameters.append(f1)

    middle_node.output_ports.append(OutputPort(id="output_1", value="linear_1"))

    e1 = Edge(
        id="input_edge",
        parameters={"weight": [[1, 0], [0, 1]]},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=middle_node.id,
        receiver_port=ip1.id,
    )

    mod_graph.edges.append(e1)

    json_file = mod.to_json_file(os.path.join(tmpdir, f"{mod.id}.json"))
    yaml_file = mod.to_yaml_file(os.path.join(tmpdir, f"{mod.id}.yaml"))
    bson_file = mod.to_bson_file(os.path.join(tmpdir, f"{mod.id}.bson"))

    json_mod = load_mdf(json_file)
    yaml_mod = load_mdf(yaml_file)
    bson_mod = load_mdf(bson_file)

    # Check both the YAML, JSON and BSON loaded versions of the model
    for mod in [json_mod, yaml_mod, bson_mod]:
        assert (
            mod.graphs[0].edges[0].parameters["weight"] == np.array([[1, 0], [0, 1]])
        ).all()


def test_ndarray_json_metadata():
    model = Node(id="a", metadata={"b": np.array([0])})
    model.to_json()


def test_serialization_Function_metadata():
    model = Function(id="a", metadata={"b": 0}, value="0")
    new_model = model.from_dict(model.to_dict())

    assert new_model.metadata == model.metadata


@pytest.mark.parametrize(
    "function, args",
    [
        ("linear", {"slope": 1, "intercept": 1}),
        ({"linear": {"slope": 1, "intercept": 1}}, None),
    ],
)
def test_serialization_Function_function_args_formats(function, args):
    model = Function("a", function=function, args=args)
    new_model = model.from_dict(model.to_dict())

    assert new_model.function == "linear"
    assert new_model.args == {"slope": 1, "intercept": 1}


def test_serialization_nested_Conditions():
    model = Condition(
        type="All",
        kwargs={
            "args": [
                Condition(
                    type="Not",
                    kwargs={
                        "condition": Condition(
                            type="BeforeNCalls",
                            kwargs={
                                "dependency": "B",
                                "n": 5,
                                "time_scale": "TimeScale.ENVIRONMENT_STATE_UPDATE",
                            },
                        )
                    },
                )
            ]
        },
    )
    new_model = model.from_dict(model.to_dict())

    cond_not = new_model.kwargs["args"][0]
    assert isinstance(cond_not, Condition)

    cond_beforencalls = cond_not.kwargs["condition"]
    assert isinstance(cond_beforencalls, Condition)

    assert cond_beforencalls.kwargs["dependency"] == "B"
    assert cond_beforencalls.kwargs["n"] == 5
    assert (
        cond_beforencalls.kwargs["time_scale"] == "TimeScale.ENVIRONMENT_STATE_UPDATE"
    )

    assert new_model.to_dict() == model.to_dict()


if __name__ == "__main__":
    test_graph_types("/tmp")
