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
        ('3+4', {}, 7),
        ('3+x', {'x':3}, 6.),
        ('[x,4]', {'x':3}, [3,4]),
        ([1,2], {'x':3}, [1,2]),
        (numpy.array([1,2]), {'x':2}, numpy.array([1,2])),
    ],
)
def test_evaluable_parameter(value, parameters, result):

    p = Parameter(id='p1',value=value)
    ep = EvaluableParameter(p)
    evl = ep.evaluate(parameters)
    print('Evaluated %s with %s to be: %s'%(value, parameters, evl))

    assert numpy.all(evl==result)
    assert numpy.all(ep.get_current_value(parameters)==result)
