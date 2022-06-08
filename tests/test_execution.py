import pytest

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


@pytest.mark.parametrize(
    "function, value, result",
    [
        ("linear", None, 6),
        # execution is in favor of function not value
        ("linear", "2 * slope * variable0 + intercept", 6),
        (None, "2 * slope * variable0 + intercept", 8),
        (None, "slope + intercept", 6),
    ],
)
@pytest.mark.parametrize(
    "args",
    [
        {"slope": 2, "intercept": 4, "variable0": 1},
        {"slope": 2, "intercept": 4, "variable0": "input"},
        {"slope": 2, "intercept": 4, "variable0": "input"},
        {"slope": "2 * input", "intercept": 4, "variable0": "input"},
        # expressions as arg values referencing other args is not currently supported
        pytest.param(
            {"slope": 2, "intercept": "2 * slope", "variable0": "input"},
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_single_function_variations(args, function, value, result):
    m = create_model(
        [
            mdf.Node(
                id="N",
                input_ports=[mdf.InputPort(id="input")],
                functions=[
                    mdf.Function(id="f", args=args, function=function, value=value)
                ],
                output_ports=[mdf.OutputPort(id="output", value="f")],
            )
        ]
    )

    eg = EvaluableGraph(m.graphs[0])
    eg.evaluate(initializer={"input": 1})

    assert eg.enodes["N"].evaluable_outputs["output"].curr_value == result
