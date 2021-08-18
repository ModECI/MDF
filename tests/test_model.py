from modeci_mdf.mdf import Model, Graph, Node, OutputPort, Function, Parameter

from modeci_mdf.utils import load_mdf

def test_model_init_kwargs():
    m = Model(
        id="Test", format="test_format", generating_application="test_application"
    )
    assert m.format == "test_format"
    assert m.generating_application == "test_application"


def test_model_graph_to_json():
    """
    Check if dumping a model to a simple JSON string works.
    """

    mod_graph0 = Graph(id="Test", parameters={"speed": 4})

    node = Node(id="N0")
    node.parameters.append(Parameter(id="rate", value=5))

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



def test_node_params_empty_dict():
    """
    Test whether we don't a serialization error with an empty Node parameters
    """
    Node().to_json()


def test_param_args_empty_dict():
    """
    Test whether we don't a serialization error when passing empty dicts to Function args
    """
    Parameter(args={}).to_json()


def test_graph_inputs():
    r"""
    Test whether we can retrieve graph input node\ports via the inputs property.
    """
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


    tmpfile = f"{tmpdir}/test.json"
    mod.to_json_file(tmpfile)

    mod_graph2 = load_mdf(tmpfile)
    print('Saved to %s: %s'%(tmpfile,mod_graph2))
    new_node0 = mod_graph2.graphs[0].nodes[0]


    assert new_node0.get_parameter('p_int').value == p_int
    assert new_node0.get_parameter('p_float').value == p_float
