import pytest
import numpy
from modeci_mdf.mdf import (
    Parameter,
    ParameterCondition,
)
from modeci_mdf.execution_engine import (
    EvaluableParameter,
)
from modelspec.utils import _params_info


@pytest.mark.parametrize(
    "value, parameters, result, cond_test, cond_value, default_initial_value",
    [
        ("3+4", {}, 7, None, None, None),
        ("3+x", {"x": 3}, 6.0, None, None, None),
        # ("p1+1", {"x": 3}, 3, None, None, None),
        ("[x,4]", {"x": 3}, [3, 4], None, None, None),
        ([1, 2], {"x": 3}, [1, 2], None, None, None),
        # ('a*b', {"a": 3, 'b':[1, 2]}, [3, 6]),
        (
            "a+b",
            {"a": numpy.array([1, 2]), "b": numpy.array([1, 2])},
            [2, 4],
            None,
            None,
            None,
        ),
        (numpy.array([1, 2]), {"x": 2}, numpy.array([1, 2]), None, None, None),
        (
            "x*y",
            {"x": 2, "y": numpy.array([1, 2])},
            numpy.array([2, 4]),
            None,
            None,
            None,
        ),
        (
            "x*y",
            {"x": numpy.array([1, 2]), "y": numpy.array([1, 2])},
            numpy.array([1, 4]),
            None,
            None,
            None,
        ),
        ### Conditions
        (0, {"x": 3}, 1, "x<4", 1, None),
        (0, {"x": 3}, 0, "x<1", 1, None),
        (0, {"x": 3}, 1, "x==3", 1, None),
        (0, {"x": 3}, 1, "x==3 and x<=5", 1, None),
        (2, {"x": 3}, 3, "x<4", 3, None),
        (0.6, {"x": 3}, 0.6, "x>=4", 1.6, None),
        ("y+z", {"x": 3, "y": 2, "z": 3}, 3, "x>y", "z", None),
        (
            numpy.array([2, 2]),
            {"x": numpy.array([1, 0]), "y": numpy.array([0, 1])},
            numpy.array([3, 2]),
            "x>y",
            [3, 3],
            None,
        ),
        (
            numpy.array([0, 0]),
            {"x": numpy.array([1, 1]), "y": numpy.array([0, 0])},
            numpy.array([1, 1]),
            "y<x",
            [1, 1],
            None,
        ),
        # ([0,0], {"x": 3}, 1, 'x<4', 1, None),
        # (0, {}, 1, 'p1>2', 1, None),
        # ("x", {"x": 'abc'}, 'abc'),
    ],
)
def test_evaluable_parameter(
    value, parameters, result, cond_test, cond_value, default_initial_value
):

    p = Parameter(id="p1", value=value)
    if default_initial_value:
        p.default_initial_value = default_initial_value
    if cond_test:
        pc = ParameterCondition(id="pc", test=cond_test, value=cond_value)
        p.conditions.append(pc)

    print(f"---\nEvaluating the {p.summary()} with {parameters}...")

    ep = EvaluableParameter(p, verbose=True)
    evl = ep.evaluate(parameters)

    print(f"Evaluated {value} with {_params_info(parameters)} to be: {evl}")

    assert numpy.all(evl == result)
    assert numpy.all(
        ep.get_current_value(parameters) == result
    )  # should still remain the same, i.e. not evaluated again.
