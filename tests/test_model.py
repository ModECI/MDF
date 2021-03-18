from modeci_mdf.mdf import Model, ModelGraph, Node, OutputPort


def test_model_graph_to_json():
    """
    Check if dumping a model to a simple JSON string works.
    """

    mod_graph0 = ModelGraph(id='Test', parameters={'speed': 4})

    node = Node(id='N0', parameters={'rate': 5})

    mod_graph0.nodes.append(node)

    # Export to JSON and see if we can load back in
    import json
    d = json.loads(mod_graph0.to_json())


def test_no_input_ports_to_json(tmpdir):
    """
    Test the edge case of exporting a model to JSON when it has a node with no input ports
    """

    mod = Model(id='ABCD')
    mod_graph = ModelGraph(id='abcd_example')
    mod.graphs.append(mod_graph)

    input_node = Node(id='input0', parameters={'input_level': 10.0})
    op1 = OutputPort(id='out_port')
    op1.value = 'input_level'
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    tmpfile = f"{tmpdir}/test.json"
    mod_graph.to_json_file(tmpfile)

    # FIXME: Doesn't seem like we have any methods for deserialization. Just do some quick and dirty checks
    # This should really be something like assert mod_graph == deserialized_mod_graph
    import json
    with open(tmpfile) as f:
        data = json.load(f)

    assert data['abcd_example']['nodes'][0]['input0']['parameters']['input_level'] == 10.0
