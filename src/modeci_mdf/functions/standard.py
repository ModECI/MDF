"""
Implementation of core MDF function ontology.

This module implements and registers all builtin MDF functions.

"""
import warnings

from typing import List, Dict, Callable

from docstring_parser import parse


# Make sure we import math and numpy for Python expression strings. These imports
# are important, do not remove even though they appear unused.
import math
import numpy


"""
A dict that stores all registered MDF functions.
"""
mdf_functions = {}


def add_mdf_function(
    name: str = None,
    description: str = None,
    arguments: List[str] = None,
    expression_string: str = None,
):

    """Register a function with MDF function ontology.

    Adds a function to the registered list of available MDF functions.

    Args:
        name: name of the function e.g.'sin','cos','linear'
        description: Information about the function
        arguments: Inputs provided to obtain the result of function
        expression_string: Function expression in string format

    Returns:
        Updates mdf_functions

    """

    mdf_functions[name] = {}

    mdf_functions[name]["description"] = description
    mdf_functions[name]["arguments"] = arguments
    mdf_functions[name]["expression_string"] = expression_string
    try:
        mdf_functions[name]["function"] = create_python_function(
            name, expression_string, arguments
        )
    except SyntaxError:
        # invalid syntax in some onnx functions (e.g. onnx_ops.or)
        mdf_functions[name]["function"] = None


def create_python_expression(expression_string: str = None) -> str:
    """Converts the mathematical representation of function into function expression in python

    Args:
        expression_string: Mathematical expression of function in string format

    Returns:
        function expression in python
    """

    for func in ["exp", "sin", "cos", "tan", "sinh", "cosh", "tanh"]:
        if "numpy." + func not in expression_string:

            expression_string = expression_string.replace(
                "%s(" % func, "numpy.%s(" % func
            )
    for func in ["maximum"]:
        expression_string = expression_string.replace("%s(" % func, "numpy.%s(" % func)
    """for func in ["max"]:
        expression_string = expression_string.replace(
            "%s(" % func, "numpy.%simum(" % func
        )"""

    return expression_string


def substitute_args(expression_string: str = None, args: Dict[str, str] = None) -> str:
    """Substitute arg with the value in args dict

    Args:
        expression_string: function expression
        args: Dictionary of arguments
    Returns:
        modified expression string after substitution

    """
    # TODO, better checks for string replacement
    for arg in args:
        expression_string = expression_string.replace(arg, str(args[arg]))
    return expression_string


def create_python_function(
    name: str = None,
    expression_string: str = None,
    arguments: List[str] = None,
) -> "types.FunctionType":
    """Create a Python function e.g. linear, exponential, sin, cos, ReLu

    Args:
        name: name of the function e.g.'sin','cos','linear'
        expression_string: Function expression in string format
        arguments: list of inputs provided to obtain result from the function


    Returns:
        A function object

    """

    # assumes expression is one line
    name = name.replace(":", "_")
    expr = create_python_expression(expression_string)
    func_str = f"def {name}({','.join(arguments)}):\n\treturn {expr}"

    res = {}
    exec(func_str, globals(), res)
    return res[name]


def parse_description_and_args(f: Callable):
    """Parse the description and arguments from a callable."""

    # Parse the docstring into markdown format
    p = parse(f.__doc__)

    # Extract the description, use the long description if available.
    # "short_description" only parse the first non-empty line and
    # "long_description" parse the rest of the docstring i.e.
    # it skips the first non-empty line and parse the rest of the docstring
    if p.long_description:
        description = f"{p.short_description} {p.long_description}"
    else:
        description = p.short_description

    args = [p.arg_name for p in p.params]

    return description, args


def add_function_from_callable(f: Callable, module_alias: str = None):
    """Adds a standard function from a callable.

    Args:
        f: A callable object.
        module_alias: A string to prepend to the function name.

    Returns:
        None
    """
    description, args = parse_description_and_args(f)

    expression_string = f.__name__ + "(" + ",".join(args) + ")"

    if module_alias:
        expression_string = module_alias + "." + expression_string

    add_mdf_function(
        name=f.__name__,
        description=description,
        arguments=args,
        expression_string=expression_string,
    )


def add_public_functions_from_module(module, module_alias: str = None):
    """Adds all public functions from a module to MDF standard functions.

    Args:
        module: A module object.
        module_alias: A string to prepend to the function names.

    Returns:
        None
    """

    try:
        for name in module.__all__:
            if callable(module.__dict__[name]):
                add_function_from_callable(
                    module.__dict__[name], module_alias=module_alias
                )

    except AttributeError:
        warnings.warn(
            f"Module {module.__name__} does not have an __all__ attribute. "
            f"No MDF standard functions were found."
        )


# Populate the list of known functions

if len(mdf_functions) == 0:

    STANDARD_ARG_0 = "variable0"
    STANDARD_ARG_1 = "variable1"

    add_mdf_function(
        "linear",
        description="A linear function, calculated from a slope and an intercept",
        arguments=[STANDARD_ARG_0, "slope", "intercept"],
        expression_string="(%s * slope + intercept)" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "logistic",
        description="Logistic function",
        arguments=[STANDARD_ARG_0, "gain", "bias", "offset"],
        expression_string="1/(1 + exp(-1*gain*(%s + bias) + offset))"
        % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "exponential",
        description="Exponential function",
        arguments=[STANDARD_ARG_0, "scale", "rate", "bias", "offset"],
        expression_string="scale * exp((rate * %s) + bias) + offset" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "sin",
        description="Sine function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * sin(%s)" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "cos",
        description="Cosine function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * cos(%s)" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "tan",
        description="Tangent function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * tan(%s)" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "sinh",
        description="Hyperbolic sine function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * sinh(%s)" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "cosh",
        description="Hyperbolic cosine function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * cosh(%s)" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "tanh",
        description="Hyperbolic tangent function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * tanh(%s)" % (STANDARD_ARG_0),
    )

    add_mdf_function(
        "MatMul",
        description="Matrix multiplication (work in progress...)",
        arguments=["A", "B"],
        expression_string="A @ B",
    )

    add_mdf_function(
        "Relu",
        description="Rectified linear function (work in progress...)",
        arguments=["A"],
        expression_string="A * (A > 0)",
    )

    # Enumerate all available ONNX operators and add them as MDF functions.
    from modeci_mdf.functions.onnx import get_onnx_ops

    for mdf_func_spec in get_onnx_ops():
        add_mdf_function(**mdf_func_spec)

    # Add the ACT-R functions.
    import modeci_mdf.functions.actr as actr

    add_public_functions_from_module(actr, module_alias="actr")

    # Add the DDM functions.
    import modeci_mdf.functions.ddm as ddm

    add_public_functions_from_module(ddm, module_alias="ddm")

if __name__ == "__main__":

    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(mdf_functions)
