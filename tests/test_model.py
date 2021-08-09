from modeci_mdf.mdf import Model, Graph, Node, OutputPort, Function, Condition,ConditionSet


def test_model_init_kwargs():
    m = Model(
        id="Test", format="test_format", generating_application="test_application",metadata={"info":'test_metadata'}
    )
    assert m.format == "test_format"
    assert m.generating_application == "test_application"
    assert m.metadata== {"info":'test_metadata'}
    assert m.id == 'Test'


def test_model_graph_to_json():
    """
    Check if dumping a model to a simple JSON string works.
    """


    mod_graph0 = Graph(id="Test", parameters={"speed": 4},metadata={'info':"mdf_model"})

    node = Node(id="N0", parameters={"rate": 5},metadata={'info':"mdf_Node"})

    node1=Node(id='test_node',parameters={"level":3},metadata={'info':"mdf_Node2"})

    condition=Condition(type='Always',metadata={'info':"mdf_condition"})

    mod_graph0.conditions = ConditionSet(node_specific={node1.id: condition})


    mod_graph0.nodes.append(node)
    mod_graph0.nodes.append(node1)

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

def test_include_metadata_to_json(tmpdir):
    """
    Test for serialization
    """

    mod = Model(id="ABCD",metadata={"info":"model_test"})
    mod_graph = Graph(id="abcd_example",metadata={"info":{"graph_test":{"environment_x":"xyz"}}})
    mod.graphs.append(mod_graph)

    input_node = Node(id="input0", parameters={"input_level": 10.0},metadata={"color":".8 0 .8"})
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
    Node(parameters={}).to_json()

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
