import pytest
import numpy
from modeci_mdf.mdf import (
    Parameter,
)
from modeci_mdf.execution_engine import (
    EvaluableParameter,
)


@pytest.mark.parametrize(
    "value, parameters, result",
    [
        ("3+4", {}, 7),
        ("3+x", {"x": 3}, 6.0),
        ("[x,4]", {"x": 3}, [3, 4]),
        ([1, 2], {"x": 3}, [1, 2]),
        # ('a*b', {"a": 3, 'b':[1, 2]}, [3, 6]),
        ("a+b", {"a": numpy.array([1, 2]), "b": numpy.array([1, 2])}, [2, 4]),
        (numpy.array([1, 2]), {"x": 2}, numpy.array([1, 2])),
        ("x*y", {"x": 2, "y": numpy.array([1, 2])}, numpy.array([2, 4])),
        (
            "x*y",
            {"x": numpy.array([1, 2]), "y": numpy.array([1, 2])},
            numpy.array([1, 4]),
        ),
    ],
)
def test_evaluable_parameter(value, parameters, result):

    p = Parameter(id="p1", value=value)
    ep = EvaluableParameter(p, verbose=True)
    evl = ep.evaluate(parameters)
    print(f"Evaluated {value} with {parameters} to be: {evl}")

    assert numpy.all(evl == result)
    assert numpy.all(ep.get_current_value(parameters) == result)
