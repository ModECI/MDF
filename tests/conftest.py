import pytest
from modeci_mdf.mdf import Model, Graph, Node, Edge, InputPort, OutputPort, Parameter

#
# def pytest_exception_interact(node, call, report):
#     excinfo = call.excinfo
#     if "script" in node.funcargs:
#         excinfo.traceback = excinfo.traceback.cut(path=node.funcargs["script"])
#     report.longrepr = node.repr_failure(excinfo)

backend_markers = {
    "ACT-R": "actr",
    "PyTorch": "pytorch",
    "NeuroML": "neuroml",
    "PsyNeuLink": "psyneulink",
    "TensorFlow": "tensorflow",
}


def pytest_collection_modifyitems(items):
    """Mark tests into whether they are core-mdf or they require specific backends to be installed."""
    for item in items:
        for pattern, marker in backend_markers.items():

            # Mark any test of an example that is in a backend folder as requiring that backend.
            if "tests/test_examples.py" in item.nodeid and pattern in item.nodeid:
                item.add_marker(pytest.mark.__getattr__(marker))
                break

            # Mark any test that is under interfaces with the appropriate backend.
            elif "tests/interfaces" in item.nodeid and marker in item.nodeid:
                item.add_marker(pytest.mark.__getattr__(marker))
                break

        # These are a couple examples that should probably be under PyTorch but are in the MDF folder
        if "abcd_torch.py" in item.nodeid or "rnn_pytorch.py" in item.nodeid:
            item.add_marker(pytest.mark.pytorch)

        # For some reason there are a bunch of examples in examples/ONNX that require pytorch. These
        # should probably be rewritten to not use ONNX or moved to examples/PyTorch/ONNX or something.
        if "tests/test_examples.py" in item.nodeid and "ONNX" in item.nodeid:
            item.add_marker(pytest.mark.pytorch)

        # All other tests should be marked core MDF by default if they don't have another backend marker.
        # Remember, some tests could be marked manually and not above.
        if (
            len([m for m in item.iter_markers() if m.name in backend_markers.values()])
            == 0
        ):
            item.add_marker(pytest.mark.coremdf)


@pytest.fixture
def simple_model_mdf():
    """
    A simple MDF model with two nodes. Input node has an input port with no receiving edge but it is not used
    because the output port uses a parameter instead.
    """
    mod = Model(id="Simple")
    mod_graph = Graph(id="simple_example")
    mod.graphs.append(mod_graph)

    input_node = Node(id="input_node")
    input_node.parameters.append(Parameter(id="input_level", value=0.5))
    op1 = OutputPort(id="out_port")
    op1.value = "input_level"
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

    processing_node = Node(id="processing_node")
    mod_graph.nodes.append(processing_node)

    processing_node.parameters.append(Parameter(id="lin_slope", value=0.5))
    processing_node.parameters.append(Parameter(id="lin_intercept", value=0))
    processing_node.parameters.append(Parameter(id="log_gain", value=3))

    ip1 = InputPort(id="input_port1")
    processing_node.input_ports.append(ip1)

    f1 = Parameter(
        id="linear_1",
        function="linear",
        args={"variable0": ip1.id, "slope": "lin_slope", "intercept": "lin_intercept"},
    )
    f2 = Parameter(
        id="logistic_1",
        function="logistic",
        args={"variable0": f1.id, "gain": "log_gain", "bias": 0, "offset": 0},
    )
    processing_node.parameters.append(f1)
    processing_node.parameters.append(f2)
    processing_node.output_ports.append(OutputPort(id="output_1", value="logistic_1"))

    e1 = Edge(
        id="input_edge",
        parameters={"weight": 0.55},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=processing_node.id,
        receiver_port=ip1.id,
    )

    mod_graph.edges.append(e1)

    return mod


@pytest.fixture
def create_model():
    def _create_model(nodes=None, edges=None, conditions=None):
        if nodes is None:
            nodes = []

        if edges is None:
            edges = []

        return Model(
            id="M",
            graphs=[
                Graph(
                    id="G",
                    nodes=nodes,
                    edges=edges,
                    conditions=conditions,
                )
            ],
        )

    return _create_model
