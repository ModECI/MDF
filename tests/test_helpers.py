import pytest
from modeci_mdf.execution_engine import get_required_variables_from_expression


@pytest.mark.parametrize(
    "expression, expected_variables",
    [
        (0, []),
        ((), []),
        ("", []),
        ("x", ["x"]),
        ("x.y", ["x"]),
        ("3*x + y", ["x", "y"]),
        ("x[y]", ["x", "y"]),
        ("x[y[z]]", ["x", "y", "z"]),
        ("x[y[z + 1]]", ["x", "y", "z"]),
        ("x[y] + z[a]", ["x", "y", "z", "a"]),
        ("x[y[z]] + a[b]", ["x", "y", "z", "a", "b"]),
        ("x[y[z[1]]] + a[0]", ["x", "y", "z", "a"]),
        ("x[y[z] + 1]", ["x", "y", "z"]),
        ("x[y[z[1 + a] + b[c]]] + d[0]", ["x", "y", "z", "a", "b", "c", "d"]),
        ("[x, 0, 1]", ["x"]),
    ],
)
def test_expression_parsing(expression, expected_variables):
    assert set(get_required_variables_from_expression(expression)) == set(
        expected_variables
    )
    assert set(get_required_variables_from_expression(f"[{expression}]")) == set(
        expected_variables
    )
