from modeci_mdf.mdf import Model, Graph, Node, OutputPort, Function, Condition, ConditionSet, Parameter, Edge, State, Stateful_Parameter, InputPort

from modeci_mdf.utils import load_mdf

def test_model_init_kwargs():
    m = Model(
        id="Test", format="test_format", generating_application="test_application"
    )
    m.metadata={"info":'test_metadata'}
    print(m)
    assert m.format == "test_format"
    assert m.generating_application == "test_application"
    assert m.metadata== {"info":'test_metadata'}
    assert m.id == 'Test'


def test_graph_init_kwargs():
    g = Graph(
        id="Test_Graph", parameters="test_parameters", conditions="test_Condition"
    )
    assert g.parameters == "test_parameters"
    assert g.conditions == "test_Condition"


def test_Node_init_kwargs():
    n = Node(id="test_node", parameters="test_parameters")

    assert n.id == "test_node"
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


def test_Stateful_Parameter_init_kwargs():
    sp = Stateful_Parameter(id="test_stateful", default_initial_value="0", value="6")
    assert sp.id == "test_stateful"
    assert sp.default_initial_value == "0"


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
    assert e.id == "test_Edge"


def test_ConditionSet_init_kwargs():
    CS = ConditionSet(
        node_specific="test_node_specific", termination="test_termination"
    )
    assert CS.node_specific == "test_node_specific"
    assert CS.termination == "test_termination"


def test_Condition_init_kwargs():
    """ Check the working of Condition"""
    C = Condition(type="test_type", n="test_n", dependency="test_dependency")
    assert C.type == "test_type"
    assert C.args == {"n": "test_n", "dependency": "test_dependency"}


def test_Condition_init_kwargs():
    C = Condition(type="test_type", n="test_n", dependencies="test_dependencies")
    assert C.type == "test_type"
    assert C.args == {"n": "test_n", "dependencies": "test_dependencies"}


def test_model_graph_to_json():
    """Check if dumping a model to a simple JSON string works."""


    mod_graph0 = Graph(id="Test", parameters={"speed": 4},metadata={'info':"mdf_model"})

    node = Node(id="N0",metadata={'info':"mdf_Node"})
    node.parameters.append(Parameter(id="rate", value=5))

    node1=Node(id='test_node', metadata={'info':"mdf_Node2"})
    node1.parameters.append(Parameter(id="level", value=3))

    condition=Condition(type='Always',metadata={'info':"mdf_condition"})

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

    # FIXME: Doesn't seem like we have any methods for deserialization. Just do some quick and dirty checks
    # This should really be something like assert mod_graph == deserialized_mod_graph
    import json

    with open(tmpfile) as f:
        data = json.load(f)
    print(data)

    assert data["ABCD"]["graphs"]["abcd_example"]["nodes"]["input0"]["parameters"]["input_level"]["value"] == 10.0

    mod_graph2 = load_mdf(tmpfile)

    print(mod_graph2)
    assert mod_graph2.graphs[0].nodes[0].parameters[0].value == 10.0


def test_include_metadata_to_json(tmpdir):
    """
    Test for serialization
    """

    mod = Model(id="ABCD",metadata={"info":"model_test"})
    mod_graph = Graph(id="abcd_example",metadata={"info":{"graph_test":{"environment_x":"xyz"}}})
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0", metadata={"color":".8 0 .8"})
    input_node.parameters.append(Parameter(id="input_level", value=10.0))
    op1 = OutputPort(id="out_port",metadata={"info":"value at OutputPort"})
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

    assert data["abcd_example"]["metadata"] == {"info":{"graph_test":{"environment_x":"xyz"}}}
    assert data["abcd_example"]["nodes"]["input0"]["metadata"] == {"color":".8 0 .8"}
    assert data["abcd_example"]["nodes"]["input0"]["output_ports"]["out_port"]["metadata"]=={"info":"value at OutputPort"}



def test_node_params_empty_dict():
    """
    Test whether we get a serialization error when passing empty dicts to Node parameters
    """
    Node().to_json()

def test_node_metadata_empty_dict():
    """
    Check for serialization error when passing empty dicts to Node metadata
    """
    Node(metadata={}).to_json()

def test_metadata_dict():
    """
    Test whether we get a serialization error when passing anything else from a dictionary
    """
    Graph(metadata='info').to_json()


def test_param_args_empty_dict():
    """
    Test whether we don't a serialization error when passing empty dicts to Parameter args
    """
    Parameter(id='noargs',args={}).to_json()


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
    p_str = 'p_int + p_float'
    node0.parameters.append(Parameter(id="p_str", value=p_str))
    p_str2 = '2'
    node0.parameters.append(Parameter(id="p_str2", value=p_str2))
    p_list = ['2',2,'two']
    node0.parameters.append(Parameter(id="p_list", value=p_list))

    p_dict = {'a':3,'w':{'b':3,'x':True,'y':[2,2,2,2]}}
    node0.parameters.append(Parameter(id="p_dict", value=p_dict))

    print(mod)
    tmpfile = f"{tmpdir}/test.json"
    mod.to_json_file(tmpfile)
    mod_graph2 = load_mdf(tmpfile)
    print('Saved to %s: %s'%(tmpfile,mod_graph2))
    new_node0 = mod_graph2.graphs[0].nodes[0]

    for p in [p.id for p in node0.parameters]:
        print('Testing %s, is %s = %s?'%(p,new_node0.get_parameter(p).value,eval(p)))
        assert new_node0.get_parameter(p).value == eval(p)
        assert type(new_node0.get_parameter(p).value) == type(eval(p))

if __name__ == '__main__':
    test_graph_types('/tmp')
