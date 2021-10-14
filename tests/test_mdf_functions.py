import math

import numpy
import pytest

import modeci_mdf.functions.standard as stdf


@pytest.mark.parametrize(
    "name, parameters, expected_result",
    [
        ("linear", {"variable0": 1, "slope": 2, "intercept": 3}, 5),
        (
            "logistic",
            {"variable0": 1, "gain": 2, "bias": 3, "offset": 4},
            0.9820137900379085,
        ),
        (
            "exponential",
            {"variable0": 1, "scale": 2, "rate": 3, "bias": 4, "offset": 5},
            2198.266316856917,
        ),
        ("sin", {"variable0": math.pi / 2, "scale": 2}, 2.0),
        ("cos", {"variable0": math.pi, "scale": 2}, -2.0),
        (
            "MatMul",
            {"A": numpy.array([[1, 2], [3, 4]]), "B": numpy.array([[1, 2], [3, 4]])},
            numpy.array([[7, 10], [15, 22]]),
        ),
        ("Relu", {"A": 1}, 1),
        ("Relu", {"A": -1}, 0),
    ],
)
def test_std_functions(name, expected_result, parameters):
    try:
        assert stdf.mdf_functions[name]["function"](**parameters) == expected_result
    except ValueError:
        assert numpy.array_equal(
            stdf.mdf_functions[name]["function"](**parameters), expected_result
        )
