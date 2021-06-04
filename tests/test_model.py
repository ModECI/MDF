from modeci_mdf.mdf import (
    Model,
    Graph,
    Node,
    Function,
    InputPort,
    OutputPort,
    Edge,
    State,
    ConditionSet,
    Condition,
)


def test_model_init_kwargs():
    m = Model(
        id="Test", format="test_format", generating_application="test_application"
    )
    assert m.format == "test_format"
    assert m.generating_application == "test_application"


def test_graph_init_kwargs():
    g = Graph(
        id="Test_Graph", parameters="test_parameters", conditions="test_Condition"
    )
    assert g.parameters == "test_parameters"
    # assert g.condition == "test_Condition"


def test_Node_init_kwargs():
    n = Node(id="Test_Node", parameters="test_parameters")
    assert n.parameters == "test_parameters"


def test_Function_init_kwargs():
    f = Function(id="Test_Function", function="Test_function", args="Test_args")
    assert f.function == "Test_function"
    assert f.args == "Test_args"


def test_InputPort_init_kwargs():
    ip = InputPort(id="Test_InputPort", shape="Test_shape", type="Test_type")
    assert ip.shape == "Test_shape"
    assert ip.type == "Test_type"


def test_OutputPort_init_kwargs():
    op = OutputPort(id="test_OutputPort", value="test_value")
    assert op.value == "test_value"


def test_State_init_kwargs():
    st = State(
        id="test_State",
        default_initial_value="test_default_initial_value",
        value="test_value",
        time_derivative="test_time_derivative",
    )
    assert st.default_initial_value == "test_default_initial_value"
    assert st.value == "test_value"
    assert st.time_derivative == "test_time_derivative"


def test_Edge_init_kwargs():
    e = Edge(
        id="test_Edge",
        parameters="test_parameters",
        sender="test_sender",
        receiver="test_receiver",
        sender_port="test_sender_port",
        receiver_port="test_receiver_port",
    )
    assert e.parameters == "test_parameters"
    assert e.sender == "test_sender"
    assert e.receiver == "test_receiver"
    assert e.sender_port == "test_sender_port"
    assert e.receiver_port == "test_receiver_port"


def test_ConditionSet_init_kwargs():
    CS = ConditionSet(
        node_specific="test_node_specific", termination="test_termination"
    )
    assert CS.node_specific == "test_node_specific"
    assert CS.termination == "test_termination"


def test_Condition_init_kwargs():
    C = Condition(
        type="test_type",
        args="test_args",
        dependency="test_dependency",
        n="test_n",
        dependencies="test_dependencies",
    )
    assert C.type == "test_type"


def test_model_graph_to_json():
    """
    Check if dumping a model to a simple JSON string works.
    """

    mod_graph0 = Graph(id="Test", parameters={"speed": 4})

    node = Node(id="N0", parameters={"rate": 5})

    mod_graph0.nodes.append(node)

    # Export to JSON and see if we can load back in
    import json

    d = json.loads(mod_graph0.to_json())


def test_no_input_ports_to_json(tmpdir):
    """
    Test the edge case of exporting a model to JSON when it has a node with no input ports
    """

    mod = Model(id="ABCD")
    mod_graph = Graph(id="abcd_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0", parameters={"input_level": 10.0})
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    tmpfile = f"{tmpdir}/test.json"
    mod_graph.to_json_file(tmpfile)

    # FIXME: Doesn't seem like we have any methods for deserialization. Just do some quick and dirty checks
    # This should really be something like assert mod_graph == deserialized_mod_graph
    import json

    with open(tmpfile) as f:
        data = json.load(f)

    assert data["abcd_example"]["nodes"]["input0"]["parameters"]["input_level"] == 10.0


def test_node_params_empty_dict():
    """
    Test whether we don't a serialization error when passing empty dicts to Node parameters
    """
    Node(parameters={}).to_json()


def test_func_args_empty_dict():
    """
    Test whether we don't a serialization error when passing empty dicts to Function args
    """
    Function(args={}).to_json()


def test_graph_inputs():
    r"""
    Test whether we can retrieve graph input node\ports via the inputs property.
    """
    mod = Model(id="ABCD")
    mod_graph = Graph(id="abcd_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0", parameters={"input_level": 10.0})
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)


def test_graph_inputs_none(simple_model_mdf):
    """Test that the simple model with no input ports used has no graph inputs"""
    assert len(simple_model_mdf.graphs[0].inputs) == 0
