import graph_scheduler
import numpy as np
import pytest

import modeci_mdf.mdf as mdf
from modeci_mdf.execution_engine import EvaluableGraph


def test_execution_engine_main(tmpdir):

    import modeci_mdf.execution_engine
    from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW
    import numpy as np

    mdf_formats = ["json", "yaml"]
    array_formats = [FORMAT_NUMPY, FORMAT_TENSORFLOW]

    # For now, don't make tensorflow a requiremnt...
    try:
        import tensorflow
    except:
        array_formats = [FORMAT_NUMPY]

    for mdf_format in mdf_formats:
        for array_format in array_formats:

            eg = modeci_mdf.execution_engine.main(
                "examples/MDF/Simple.%s" % mdf_format, array_format=array_format
            )
            output = (
                eg.enodes["processing_node"].evaluable_outputs["output_1"].curr_value
            )
            assert output == 0.6016871801828567

            eg = modeci_mdf.execution_engine.main(
                "examples/MDF/ABCD.%s" % mdf_format, array_format=array_format
            )
            output = eg.enodes["D"].evaluable_outputs["out_port"].curr_value
            assert -1.7737500239216304 - output < 1e-9

            eg = modeci_mdf.execution_engine.main(
                "examples/MDF/Arrays.%s" % mdf_format, array_format=array_format
            )
            output = eg.enodes["middle_node"].evaluable_outputs["output_1"].curr_value
            assert output[0, 0] == 0.5
            assert output[1, 1] == 4

            eg = modeci_mdf.execution_engine.main(
                "examples/MDF/States.%s" % mdf_format, array_format=array_format
            )
            output = eg.enodes["counter_node"].evaluable_outputs["out_port"].curr_value
            assert output == 1
            output = eg.enodes["sine_node"].evaluable_outputs["out_port"].curr_value
            assert output == 0


def test_execution_engine_onnx(tmpdir):

    import modeci_mdf.execution_engine
    import numpy as np

    mdf_formats = ["json", "yaml"]
    from modelspec.utils import FORMAT_NUMPY

    array_format = FORMAT_NUMPY

    for mdf_format in mdf_formats:

        eg = modeci_mdf.execution_engine.main(
            "examples/ONNX/ab.%s" % mdf_format, array_format=array_format
        )
        output = eg.enodes["Mul_3"].evaluable_outputs["_4"].curr_value
        assert np.array_equal(output, np.full((2, 3), 5))


_abc_conditions_expected_output = [
    {"A"},
    {"A"},
    {"B"},
    {"A"},
    {"C"},
    {"A"},
    {"B"},
    {"A"},
    {"A"},
    {"C", "B"},
    {"A"},
]


@pytest.mark.parametrize(
    "fi, expected_output",
    [
        ("examples/MDF/abc_conditions.json", _abc_conditions_expected_output),
        ("examples/MDF/abc_conditions.yaml", _abc_conditions_expected_output),
    ],
)
def test_condition_scheduler_main(fi, expected_output):

    import modeci_mdf.execution_engine

    eg = modeci_mdf.execution_engine.main(fi)
    output = [{n.id for n in nodes} for nodes in eg.scheduler.execution_list[None]]
    assert output == expected_output

    assert eg.enodes["A"].evaluable_parameters["count_A"].curr_value == 7
    assert eg.enodes["B"].evaluable_parameters["count_B"].curr_value == 3


def test_nested_conditions(create_model):
    A = mdf.Node(
        id="A",
        input_ports=[mdf.InputPort(id="A_input")],
        parameters=[
            mdf.Parameter(id="A_param", value="A_param + 1", default_initial_value=0)
        ],
        output_ports=[mdf.OutputPort(id="A_output", value="A_param")],
    )
    B = mdf.Node(
        id="B",
        input_ports=[mdf.InputPort(id="B_input")],
        parameters=[
            mdf.Parameter(id="B_param", value="B_param + 1", default_initial_value=0)
        ],
        output_ports=[mdf.OutputPort(id="B_output", value="B_param")],
    )
    C = mdf.Node(
        id="C",
        input_ports=[mdf.InputPort(id="C_input")],
        parameters=[
            mdf.Parameter(id="C_param", value="C_param + 1", default_initial_value=0)
        ],
        output_ports=[mdf.OutputPort(id="C_output", value="C_param")],
    )

    m = create_model(
        nodes=[A, B, C],
        conditions=mdf.ConditionSet(
            node_specific={
                "B": mdf.Condition(
                    type="EveryNCalls", kwargs={"dependency": A, "n": 2}
                ),
                # condition is simple EveryNCalls(B, 2) within two Not conditions
                "C": mdf.Condition(
                    type="Not",
                    kwargs={
                        "condition": mdf.Condition(
                            type="Not",
                            kwargs={
                                "condition": mdf.Condition(
                                    type="EveryNCalls", kwargs={"dependency": B, "n": 2}
                                )
                            },
                        )
                    },
                ),
            },
            termination={
                graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE: mdf.Condition(
                    type="AfterNCalls", kwargs={"dependency": C, "n": 2}
                )
            },
        ),
    )

    m2 = mdf.Model.from_dict(m.to_dict())
    m3 = mdf.Model.from_dict(m2.to_dict())

    for model in [m, m2, m3]:
        eg = EvaluableGraph(model.graphs[0])
        eg.evaluate(initializer={"A_input": 0})

        assert eg.enodes["A"].evaluable_outputs["A_output"].curr_value == 8
        assert eg.enodes["B"].evaluable_outputs["B_output"].curr_value == 4
        assert eg.enodes["C"].evaluable_outputs["C_output"].curr_value == 2


@pytest.mark.skipif(
    graph_scheduler.__version__ < "1.1.0", reason="Threshold added in 1.1.0"
)
def test_threshold(create_model):
    A = mdf.Node(
        id="A",
        input_ports=[mdf.InputPort(id="A_input")],
        parameters=[
            mdf.Parameter(id="A_param", value="A_param + 1", default_initial_value=0)
        ],
        output_ports=[mdf.OutputPort(id="A_output", value="A_param")],
    )

    m = create_model(
        nodes=[A],
        conditions=mdf.ConditionSet(
            termination={
                graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE: mdf.Condition(
                    type="Threshold",
                    dependency=A,
                    parameter="A_param",
                    threshold=5,
                    comparator=">=",
                )
            },
        ),
    )

    m2 = mdf.Model.from_dict(m.to_dict())
    m3 = mdf.Model.from_dict(m2.to_dict())

    for model in [m, m2, m3]:
        eg = EvaluableGraph(model.graphs[0])
        eg.evaluate(initializer={"A_input": 0})

        assert eg.enodes["A"].evaluable_outputs["A_output"].curr_value == 5


@pytest.mark.skipif(
    graph_scheduler.__version__ < "1.1.0", reason="Threshold added in 1.1.0"
)
@pytest.mark.parametrize(
    "threshold_dependency, threshold_parameter",
    [
        ("A", "A_output"),
        ("A", "A_param_2"),
        ("B", "B_input"),
    ],
)
@pytest.mark.parametrize(
    "threshold_indices, result",
    [
        ((0,), np.array([5, 6])),
        ((1,), np.array([4, 5])),
    ],
)
def test_threshold_nonscalar_values(
    create_model, threshold_dependency, threshold_parameter, threshold_indices, result
):
    A = mdf.Node(
        id="A",
        input_ports=[mdf.InputPort(id="A_input")],
        parameters=[
            mdf.Parameter(id="A_param", value="A_param + 1", default_initial_value=0),
            mdf.Parameter(
                id="A_param_2",
                value="[A_param_2[0] + 1, A_param_2[1] + 1]",
                default_initial_value=np.array([0, 1]),
            ),
        ],
        output_ports=[mdf.OutputPort(id="A_output", value="A_param_2", shape=(2,))],
    )
    B = mdf.Node(
        id="B",
        input_ports=[mdf.InputPort(id="B_input", shape=(2,))],
        output_ports=[mdf.OutputPort(id="B_output", value="B_input", shape=(2,))],
    )

    m = create_model(
        nodes=[A, B],
        edges=[
            mdf.Edge(
                id="A->B",
                sender="A",
                receiver="B",
                sender_port="A_output",
                receiver_port="B_input",
            )
        ],
        conditions=mdf.ConditionSet(
            termination={
                # include JustRan condition so that
                # ENVIRONMENT_STATE_UPDATE isn't terminated immediately
                # after A reaches threshold when it is the condition's
                # dependency since we are testing B's output. This
                # ensures B will have its input and output updated
                graph_scheduler.TimeScale.ENVIRONMENT_STATE_UPDATE: mdf.Condition(
                    type="And",
                    dependencies=[
                        mdf.Condition(
                            type="Threshold",
                            dependency=threshold_dependency,
                            parameter=threshold_parameter,
                            threshold=5,
                            comparator=">=",
                            indices=threshold_indices,
                        ),
                        mdf.Condition(type="JustRan", dependency="B"),
                    ],
                )
            },
        ),
    )

    eg = EvaluableGraph(m.graphs[0])
    eg.evaluate(initializer={"A_input": 0})

    assert np.array_equal(
        eg.enodes["B"].evaluable_outputs["B_output"].curr_value, result
    )
