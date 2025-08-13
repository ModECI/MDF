import graph_scheduler
import pytest

import modeci_mdf.mdf as mdf
from modeci_mdf.execution_engine import EvaluableGraph


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
        {"slope": 2, "intercept": "2 * slope", "variable0": "input"},
        {"slope": "math.sqrt(4)", "intercept": 4, "variable0": 1},
        {"slope": "numpy.sqrt(4)", "intercept": 4, "variable0": 1},
    ],
)
def test_single_function_variations(create_model, args, function, value, result):
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


@pytest.mark.parametrize(
    "node_specific, termination, result",
    [
        ({"A": mdf.Condition(type="Always")}, None, 1),
        (
            None,
            {
                graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE: mdf.Condition(
                    type="AfterNCalls", dependency="A", n=5
                )
            },
            5,
        ),
    ],
)
def test_condition_variations(create_model, node_specific, termination, result):
    A = mdf.Node(
        id="A",
        input_ports=[mdf.InputPort(id="A_input")],
        parameters=[
            mdf.Parameter(id="A_param", value="A_param + 1", default_initial_value=0)
        ],
        output_ports=[mdf.OutputPort(id="A_output", value="A_param")],
    )

    model = create_model(
        nodes=[A],
        conditions=mdf.ConditionSet(
            node_specific=node_specific, termination=termination
        ),
    )

    eg = EvaluableGraph(model.graphs[0])
    eg.evaluate(initializer={"A_input": 0})

    assert eg.enodes["A"].evaluable_outputs["A_output"].curr_value == result


def test_dependency_in_function_value(create_model):
    m = create_model(
        [
            mdf.Node(
                id="N",
                input_ports=[mdf.InputPort(id="input")],
                functions=[
                    mdf.Function(id="f", value="g"),
                    mdf.Function(id="g", value="1"),
                ],
                output_ports=[mdf.OutputPort(id="output", value="f")],
            )
        ]
    )

    eg = EvaluableGraph(m.graphs[0])
    eg.evaluate(initializer={"input": 1})


# NOTE: this is enabled by "don't include known args in required variable"
# but could not be tested until dependency checking for value
def test_available_arg_in_function_value(create_model):
    m = create_model(
        [
            mdf.Node(
                id="N",
                input_ports=[mdf.InputPort(id="input")],
                functions=[
                    mdf.Function(id="f", args={"g": 1}, value="g"),
                ],
                output_ports=[mdf.OutputPort(id="output", value="f")],
            )
        ]
    )

    eg = EvaluableGraph(m.graphs[0])
    eg.evaluate(initializer={"input": 1})
