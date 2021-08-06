import math

import numpy

import modeci_mdf.onnx_functions as onnx_ops

mdf_functions = {}


def _add_mdf_function(name, description, arguments, expression_string):

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


def create_python_expression(expression_string):

    for func in ["exp", "sin", "cos"]:
        if "math."+func not in expression_string:
      
            expression_string = expression_string.replace("%s(" % func, "math.%s(" % func)
    for func in ["maximum"]:
        expression_string = expression_string.replace("%s(" % func, "numpy.%s(" % func)
    
    return expression_string


def substitute_args(expression_string, args):
    # TODO, better checks for string replacement
    for arg in args:
        expression_string = expression_string.replace(arg, str(args[arg]))
    return expression_string


def create_python_function(name, expression_string, arguments):
    # assumes expression is one line
    name = name.replace(":", "_")
    expr = create_python_expression(expression_string)
    func_str = f"def {name}({','.join(arguments)}):\n\treturn {expr}"

    res = {}
    exec(func_str, globals(), res)
    return res[name]


# Populate the list of known functions

if len(mdf_functions) == 0:

    STANDARD_ARG_0 = "variable0"
    STANDARD_ARG_1 = "variable1"

    _add_mdf_function(
        "linear",
        description="A linear function, calculated from a slope and an intercept",
        arguments=[STANDARD_ARG_0, "slope", "intercept"],
        expression_string="(%s * slope + intercept)" % (STANDARD_ARG_0),
    )

    _add_mdf_function(
        "logistic",
        description="Logistic function",
        arguments=[STANDARD_ARG_0, "gain", "bias", "offset"],
        expression_string="1/(1 + exp(-1*gain*(%s + bias) + offset))"
        % (STANDARD_ARG_0),
    )

    _add_mdf_function(
        "exponential",
        description="Exponential function",
        arguments=[STANDARD_ARG_0, "scale", "rate", "bias", "offset"],
        expression_string="scale * exp((rate * %s) + bias) + offset" % (STANDARD_ARG_0),
    )

    _add_mdf_function(
        "sin",
        description="Sine function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * sin(%s)" % (STANDARD_ARG_0),
    )

    _add_mdf_function(
        "cos",
        description="Cosine function",
        arguments=[STANDARD_ARG_0, "scale"],
        expression_string="scale * cos(%s)" % (STANDARD_ARG_0),
    )

    _add_mdf_function(
        "MatMul",
        description="Matrix multiplication (work in progress...)",
        arguments=["A", "B"],
        expression_string="A @ B",
    )

    _add_mdf_function(
        "Relu",
        description="Rectified linear function (work in progress...)",
        arguments=["A"],
        expression_string="maximum(A,0)",
    )

    _add_mdf_function(
        "time_derivative_FN_V",
        description="time_derivative fitzhugh nagumo parameter V",
        arguments=[STANDARD_ARG_0,STANDARD_ARG_1, "a_v", "threshold", "b_v", "c_v", "d_v", "e_v", "f_v","Iext", "time_constant_v", "MSEC"],
        expression_string="(a_v*%s*%s*%s + (1+threshold)*b_v*%s*%s + (-1*threshold)*c_v*%s + d_v + e_v*%s + f_v*Iext)/(time_constant_v * MSEC)"
        % (STANDARD_ARG_0, STANDARD_ARG_0, STANDARD_ARG_0, STANDARD_ARG_0, STANDARD_ARG_0, STANDARD_ARG_0, STANDARD_ARG_1),
    )

    _add_mdf_function(
        "time_derivative_FN_W",
        description="time_derivative fitzhugh nagumo parameter W",
        arguments=[STANDARD_ARG_0,STANDARD_ARG_1, "a_w", "b_w", "c_w", "mode","uncorrelated_activity", "time_constant_w", "MSEC"],
        expression_string="(mode*a_w*%s + b_w*%s + c_w + (1-mode)*uncorrelated_activity )/(time_constant_w * MSEC)"
        % (STANDARD_ARG_0, STANDARD_ARG_1),
    )


    # Enumerate all available ONNX operators and add them as MDF functions.
    from modeci_mdf.onnx_functions import get_onnx_ops

    for mdf_func_spec in get_onnx_ops():
        _add_mdf_function(**mdf_func_spec)

    # Add the ACT-R functions.
    from modeci_mdf.actr_functions import get_actr_functions

    for mdf_func_spec in get_actr_functions():
        _add_mdf_function(**mdf_func_spec)


if __name__ == "__main__":

    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(mdf_functions)
