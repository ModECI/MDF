import modeci_mdf.mdf as mdf
from modeci_mdf.execution_engine import EvaluableGraph


def create_model(nodes=None, edges=None):
    if nodes is None:
        nodes = []

    if edges is None:
        edges = []

    return mdf.Model(
        id="M",
        graphs=[
            mdf.Graph(
                id="G",
                nodes=nodes,
                edges=edges,
            )
        ],
    )


def test_function_no_args_unordered():
    m = create_model(
        [
            mdf.Node(
                id="N",
                functions=[
                    mdf.Function(id="b", value="2 * a"),
                    mdf.Function(id="a", value="1"),
                ],
                output_ports=[mdf.OutputPort(id="output", value="b")],
            )
        ]
    )

    eg = EvaluableGraph(m.graphs[0])
    eg.evaluate()

    assert eg.enodes["N"].evaluable_outputs["output"].curr_value == 2
