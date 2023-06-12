"""The reference implementation of the MDF execution engine; allows for executing :class:`~modeci.mdf.Graph`
objects in Python.

This module implements a set of classes for executing loaded MDF models in Python.
The implementation is organized such that each class present in :mod:`~modeci_mdf.mdf`
has a corresponding :code:`Evaluable` version of the class. Each of these classes implements
the execution of these components and tracks their state during execution. The organization of the entire execution of
the model is implemented at the top-level :func:`~modeci_mdf.execution_engine.EvaluableGraph.evaluate` method
of the :class:`EvaluableGraph` class. The external library `graph-scheduler
<https://pypi.org/project/graph-scheduler/>`_ is used to implement the scheduling of nodes under declarative
conditional constraints.

"""
import ast
import builtins
import copy
import functools
import inspect
import itertools
import os
import re
import sys
import math

import attr
import numpy as np

import graph_scheduler
import onnxruntime

from modeci_mdf.functions.standard import mdf_functions, create_python_expression


from modelspec.utils import evaluate as evaluate_params_modelspec
from modelspec.utils import _params_info, _val_info
from modelspec.utils import FORMAT_NUMPY

from collections import OrderedDict
from typing import Union, List, Dict, Optional, Any, Tuple
from modeci_mdf.mdf import (
    Function,
    Graph,
    Condition,
    Edge,
    OutputPort,
    InputPort,
    Node,
    Parameter,
)

import modeci_mdf.functions.onnx as onnx_ops
import modeci_mdf.functions.actr as actr_funcs
import modeci_mdf.functions.ddm as ddm_funcs


FORMAT_DEFAULT = FORMAT_NUMPY

KNOWN_PARAMETERS = ["constant", "math", "numpy"] + dir(builtins)


time_scale_str_regex = r"(TimeScale)?\.(.*)"


def evaluate_expr(
    expr: Union[str, List[str], np.ndarray, "tf.tensor"] = None,
    func_params: Dict[str, Any] = None,
    array_format: str = FORMAT_DEFAULT,
    allow_strings_returned: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> np.ndarray:

    """Evaluates an expression given in string format and a :code:`dict` of parameters.

    Args:
        expr: Expression or list of expressions to be evaluated
        func_params: A dict of parameters (e.g. :code:`{'weight': 2}`)
        array_format: It can be a n-dimensional array or a tensor
        allow_strings_returned: Don't throw an error if the expression evaluates to a string
        verbose: If set to True provides in-depth information else verbose message is not displayed

    Returns:
        n-dimensional array

    """

    e = evaluate_params_modelspec(
        expr, func_params, array_format=array_format, verbose=verbose
    )
    if type(e) == str and e not in KNOWN_PARAMETERS and not allow_strings_returned:
        raise Exception(
            "Error! Could not evaluate expression [%s] with params %s, returned [%s] which is a %s"
            % (expr, _params_info(func_params, multiline=True), e, type(e))
        )
    return e


def evaluate_onnx_expr(
    expr: str,
    base_parameters: Dict[str, Any],
    evaluated_parameters: Dict,
    verbose: bool = False,
) -> Any:
    """Evaluates a simple expression in string format representing an
    ONNX function call

    Args:
        expr (str): Expression to be evaluated
        base_parameters (Dict[str, Any]): A dict of parameters that may contain variables
        evaluated_parameters (Dict): A dict mapping variables used in **base_parameters** to actual values
        verbose (bool, optional): If set to True provides in-depth information else verbose message is not displayed. Defaults to False.

    Returns:
        Any: the return value of **expr**
    """
    # Get the ONNX function
    onnx_name = expr.split("(")[0].split(".")[-1]
    onnx_function = getattr(onnx_ops, onnx_name)
    onnx_schema = onnx_ops.get_onnx_schema(onnx_name)
    onnx_arguments = set(
        list(onnx_schema.attributes.keys()) + [i.name for i in onnx_schema.inputs]
    )
    # used to attempt to match inputs to expected onnx input types
    onnx_typecast_mappings = {
        onnx_schema.AttrType.INT: int,
        onnx_schema.AttrType.FLOAT: float,
        onnx_schema.AttrType.STRING: str,
        onnx_schema.AttrType.INTS: functools.partial(np.array, dtype=int),
        onnx_schema.AttrType.FLOATS: functools.partial(np.array, dtype=float),
        onnx_schema.AttrType.STRINGS: functools.partial(np.array, dtype=str),
        # TODO: add tensor and graph types?
    }

    try:
        has_variadic = (
            onnx_schema.inputs[0].option == onnx_schema.FormalParameterOption.Variadic
        )
    except IndexError:
        has_variadic = False

    # ONNX functions expect input args or kwargs first, followed by parameters (called attributes in ONNX) as
    # kwargs. Lets construct this.
    kwargs_for_onnx = {}
    for kw, arg_expr in base_parameters.items():
        if isinstance(arg_expr, str):
            arg_expr_list = get_required_variables_from_expression(arg_expr)
            for a in arg_expr_list:
                try:
                    kwargs_for_onnx[a] = evaluated_parameters[a]
                except KeyError:
                    pass

            try:
                if arg_expr[0] == "[" and arg_expr[-1] == "]":
                    # matches previous behavior
                    continue
            except IndexError:
                pass

        kwargs_for_onnx[kw] = evaluated_parameters[kw]

    kwargs_for_onnx = {
        k: v
        for k, v in kwargs_for_onnx.items()
        if (
            (k in onnx_arguments or has_variadic)
            and "onnx_" not in k  # filter Evaluable__ class names
        )
    }

    # attempt to cast attributes to what onnx_function expects
    for k, v in kwargs_for_onnx.items():
        try:
            onnx_attr = onnx_schema.attributes[k]
        except KeyError:
            continue

        try:
            cast_type = onnx_typecast_mappings[onnx_attr.type]
        except KeyError:
            continue

        try:
            kwargs_for_onnx[k] = cast_type(v)
        except (TypeError, ValueError):
            pass

    if verbose:
        print(f"Evaluating ONNX function {onnx_name} with {kwargs_for_onnx}")

    try:
        result = onnx_function(**kwargs_for_onnx)
    except (
        onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented,
        onnxruntime.capi.onnxruntime_pybind11_state.Fail,
    ) as e:
        err = str(e)
        if (
            "bound to different types (tensor(double) and tensor(float)" not in err
            and "Could not find an implementation for the node" not in err
        ):
            raise

        # assume this is related to lack of support for float64/double
        # for Cos, Relu (and likely others) on onnx CPUExecutionProvider
        result = onnx_function(
            **{
                k: v.astype(np.float32)
                if hasattr(v, "dtype") and v.dtype == np.float64
                else v
                for k, v in kwargs_for_onnx.items()
            }
        )

    try:
        if result.dtype == np.float32:
            result = result.astype(np.float64)
    except AttributeError:
        pass

    return result


def get_required_variables_from_expression(expr: str) -> List[str]:
    """Produces a list containing variable symbols in **expr**"""

    def recursively_extract_subscripted_values(s):
        res = []
        subscript_indices = []
        depth = 0

        len_s = len(s)
        for i in range(len_s):
            if s[i] == "[":
                if depth == 0:
                    subscript_indices.append([i, None])
                depth += 1

            if s[i] == "]":
                depth -= 1

                if depth == 0:
                    subscript_indices[-1][1] = i

        # s contains no subscripts, so it won't be added in below loop
        if len(subscript_indices) == 0 and len_s > 0:
            res.append(s)

        last = 0
        for start, end in subscript_indices:
            if end is None:
                end = len_s

            res.extend(recursively_extract_subscripted_values(s[start + 1 : end]))

            # add expression being subscripted
            if last != start:
                res.append(s[last:start])

            last = end + 1

        return res

    if not isinstance(expr, str):
        return []

    result = []

    for e in recursively_extract_subscripted_values(expr):
        result.extend(
            [
                str(elem.id)
                for elem in ast.walk(
                    ast.parse(e.strip(" ,+-*/%^&").lstrip("])").rstrip("[("))
                )
                if isinstance(elem, ast.Name)
            ]
        )
    return result


class EvaluableFunction:
    """Evaluates a :class:`~modeci_mdf.mdf.Function` value during MDF graph execution.

    Args:
        function: :func:`~modeci_mdf.mdf.Function` to be evaluated e.g. mdf standard function
        verbose: If set to True Provides in-depth information else verbose message is not displayed
    """

    def __init__(self, function: Function = False, verbose: Optional[bool] = False):
        self.verbose = verbose
        self.function = function

    def evaluate(
        self,
        parameters: Dict[str, Any] = None,
        array_format: str = FORMAT_DEFAULT,
    ) -> Dict[str, Any]:

        r"""Performs evaluation on the basis of given parameters and array_format

        Args:
            parameters: A dictionary of function parameters,e.g.logistic, parameters={'gain': 2,"bias": 3,"offset": 1}
            array_format: It can be a n-dimensional array or a tensor

        Returns:
             value of function after evaluation in Dictionary

        """

        expr = None

        # print("functions value and function>>>", self.function.value, self.function.function)

        # func_val  = self.function.value

        if self.function.function:
            for f in mdf_functions:
                if f == self.function.function:
                    expr = create_python_expression(
                        mdf_functions[f]["expression_string"]
                    )
                    break

        if expr is None:
            expr = self.function.value
        #     #raise "Unknown function: {}. Known functions: {}".format(
        #     #    self.function.function,
        #     #    mdf_functions.keys,
        #     #)

        func_params = {}
        func_params.update(parameters)
        if self.verbose:
            print(
                "    Evaluating %s with %s, i.e. [%s]"
                % (self.function, _params_info(func_params), expr)
            )

        if self.function.args is not None:
            for arg in self.function.args:
                func_params[arg] = evaluate_expr(
                    self.function.args[arg],
                    func_params,
                    verbose=False,
                    array_format=array_format,
                )
                if self.verbose:
                    print(
                        "      Arg: {} became: {}".format(
                            arg, _val_info(func_params[arg])
                        )
                    )

        # If this is an ONNX operation, evaluate it without modelspec.

        if "onnx_ops." in expr:
            if self.verbose:
                print(f"{self.function.id} is evaluating ONNX function {expr}")
            self.curr_value = evaluate_onnx_expr(
                expr,
                # parameters get overridden by self.function.args
                {**parameters, **self.function.args},
                func_params,
                self.verbose,
            )
        elif "actr." in expr:
            actr_function = getattr(actr_funcs, expr.split("(")[0].split(".")[-1])
            self.curr_value = actr_function(
                *[func_params[arg] for arg in self.function.args]
            )
        elif "ddm." in expr:
            actr_function = getattr(ddm_funcs, expr.split("(")[0].split(".")[-1])
            self.curr_value = ddm_function(
                *[func_params[arg] for arg in self.function.args]
            )
        else:
            self.curr_value = evaluate_expr(
                expr, func_params, verbose=self.verbose, array_format=array_format
            )

        if self.verbose:
            print(
                "    Evaluated %s with %s =\t%s"
                % (self.function, _params_info(func_params), _val_info(self.curr_value))
            )
        return self.curr_value


class EvaluableParameter:
    """
    Evaluates the current value of a :class:`~modeci_mdf.mdf.Parameter` during the MDF graph execution.

    Args:
        parameter: The parameter to evaluate during execution.
        verbose: Whether to print output of parameter calculations.

    """

    DEFAULT_INIT_VALUE = 0.0  # Temporary!

    def __init__(self, parameter: Parameter, verbose: bool = False):

        self.verbose = verbose
        self.parameter = parameter
        self.curr_value = None

    def get_current_value(
        self, parameters: Dict[str, Any], array_format: str = FORMAT_DEFAULT
    ) -> Any:
        """
        Get the current value of the parameter; evaluates the expression if the current value has not yet been set. Note:
        this is different from :code:`'evaluate'`, as calling that method multiple times can change the state of the parameter,
        but calling this should not reevaluate the parameter if it has a current value.

        Args:
            parameters: a dictionary of parameters and their values that may or may not be needed to evaluate this
                parameter.
            array_format: The array format to use (either :code:`'numpy'` or :code:`tensorflow'`).

        Returns:
            The evaluated value of the parameter.

        """
        # FIXME: Shouldn't this just call self.evaluate, seems like there is redundant code here?
        if self.curr_value is None:
            if (
                self.parameter.value is not None
                or self.parameter.default_initial_value is not None
            ):
                if self.parameter.is_stateful():
                    if self.verbose:
                        print(f"    Initial eval of <{self.parameter.summary()}>  ")

                    if self.parameter.default_initial_value is not None:
                        return evaluate_expr(
                            self.parameter.default_initial_value,
                            parameters,
                            verbose=self.verbose,
                            array_format=array_format,
                        )
                    else:
                        return self.DEFAULT_INIT_VALUE
                else:
                    ips = {}
                    ips.update(parameters)
                    ips[self.parameter.id] = self.DEFAULT_INIT_VALUE
                    self.curr_value = evaluate_expr(
                        self.parameter.value,
                        ips,
                        verbose=self.verbose,
                        array_format=array_format,
                    )
                    if self.verbose:
                        print(
                            "    Initial eval of <{}> = {} ".format(
                                self.parameter, self.curr_value
                            )
                        )

        return self.curr_value

    def evaluate(
        self,
        parameters: Dict[str, Any],
        time_increment: Optional[float] = None,
        array_format: str = FORMAT_DEFAULT,
    ) -> Any:
        """
        Evaluate the parameter and store the result in the :code:`curr_value` attribute.

        Args:
            parameters: a dictionary of parameters and their values that may or may not be needed to evaluate this
                parameter.
            time_increment: a floating point value specifying the timestep size, only used for :code:`time_derivative`
                parameters
            array_format: The array format to use (either :code:`'numpy'` or :code:`tensorflow'`).

        Returns:
            The current value of the parameter.

        """
        if self.verbose:
            print(
                "    Evaluating {} with {} ".format(
                    self.parameter.summary(), _params_info(parameters)
                )
            )

        if self.parameter.value is not None:

            self.curr_value = evaluate_expr(
                self.parameter.value,
                parameters,
                verbose=False,
                array_format=array_format,
            )

        elif self.parameter.function:
            expr = None
            for f in mdf_functions:
                if self.parameter.function == f:
                    expr = create_python_expression(
                        mdf_functions[f]["expression_string"]
                    )
            if not expr:

                expr = self.parameter.function
                # raise "Unknown function: {}. Known functions: {}".format(
                #    self.parameter.function,
                #    mdf_functions.keys,
                # )

            func_params = {}
            func_params.update(parameters)
            if self.verbose:
                print(
                    "    Evaluating %s with %s, i.e. [%s]"
                    % (self.parameter, _params_info(func_params), expr)
                )
            for arg in self.parameter.args:
                func_params[arg] = evaluate_expr(
                    self.parameter.args[arg],
                    func_params,
                    verbose=False,
                    array_format=array_format,
                )
                if self.verbose:
                    print(
                        "      Arg: {} became: {}".format(
                            arg, _val_info(func_params[arg])
                        )
                    )

            # If this is an ONNX operation, evaluate it without modelspec.
            if "onnx_ops." in expr:
                if self.verbose:
                    print(f"{self.parameter.id} is evaluating ONNX function {expr}")
                self.curr_value = evaluate_onnx_expr(
                    expr,
                    # parameters get overridden by self.parameter.args
                    {**parameters, **self.parameter.args},
                    func_params,
                    self.verbose,
                )
            elif "actr." in expr:
                actr_function = getattr(actr_funcs, expr.split("(")[0].split(".")[-1])
                self.curr_value = actr_function(
                    *[func_params[arg] for arg in self.parameter.args]
                )
            else:
                self.curr_value = evaluate_expr(
                    expr,
                    func_params,
                    verbose=self.verbose,
                    array_format=array_format,
                )

        elif self.parameter.time_derivative is not None:

            if time_increment == None:
                self.curr_value = evaluate_expr(
                    self.parameter.default_initial_value,
                    parameters,
                    verbose=self.verbose,
                    array_format=array_format,
                )

            else:
                td = evaluate_expr(
                    self.parameter.time_derivative,
                    parameters,
                    verbose=self.verbose,
                    array_format=array_format,
                )
                if self.verbose:
                    print(
                        f"Incrementing {self.parameter.id} from {self.curr_value} by {td} over time {time_increment}"
                    )

                self.curr_value = np.add(
                    self.curr_value, td * time_increment, casting="safe"
                )

        cond_mask = None
        val_if_true = None

        if len(self.parameter.conditions) > 0:
            for condition in self.parameter.conditions:
                cond_mask = evaluate_expr(
                    condition.test,
                    parameters,
                    verbose=False,
                    array_format=array_format,
                )

                val_if_true = evaluate_expr(
                    condition.value,
                    parameters,
                    verbose=False,
                    array_format=array_format,
                )

                if self.verbose:
                    print(
                        " --- Condition: %s: %s = %s: true? %s"
                        % (condition.id, condition.test, val_if_true, cond_mask)
                    )

                # e.g. if the parameter value is set only by a set of conditions...
                if self.curr_value is None:
                    self.curr_value = self.DEFAULT_INIT_VALUE

                self.curr_value = np.where(cond_mask, val_if_true, self.curr_value)

        if self.verbose:
            print(
                "    Evaluated this: %s with %s \n       =\t%s"
                % (
                    self.parameter.summary(),
                    _params_info(parameters),
                    _val_info(self.curr_value),
                )
            )

        return self.curr_value


class EvaluableOutput:
    r"""Evaluates the current value of an :class:`~modeci_mdf.mdf.OutputPort` during MDF graph execution.

    Args:
        output_port: Attribute of a Node which exports information to the dependent Node object
        verbose: If set to True Provides in-depth information else verbose message is not displayed
    """

    def __init__(self, output_port: OutputPort, verbose: Optional[bool] = False):
        self.verbose = verbose
        self.output_port = output_port
        self.curr_value = None

    def evaluate(
        self,
        parameters: Dict[str, Any] = None,
        array_format: str = FORMAT_DEFAULT,
    ) -> Union[int, np.ndarray]:

        """Evaluate the value at the output port on the basis of parameters and array_format

        Args:
            parameters: Dictionary of global parameters of the Output Port
            array_format: It is a n-dimensional array

        Returns:
            value at output port
        """
        if self.verbose:
            print(
                "    Evaluating %s with %s "
                % (self.output_port, _params_info(parameters))
            )
        self.curr_value = evaluate_expr(
            self.output_port.value, parameters, verbose=False, array_format=array_format
        )

        if self.verbose:
            print(
                "    Evaluated %s with %s \n       =\t%s"
                % (
                    self.output_port,
                    _params_info(parameters),
                    _val_info(self.curr_value),
                )
            )
        return self.curr_value


class EvaluableInput:
    """Evaluates input value at the :class:`~modeci_mdf.mdf.InputPort` of the node during MDF graph execution.

    Args:
        input_port: The :class:`~modeci_mdf.mdf.InputPort` is an attribute of a Node which imports information to the
            :class:`~modeci_mdf.mdf.Node`
        verbose: If set to True Provides in-depth information else verbose message is not displayed
    """

    def __init__(self, input_port: InputPort, verbose: Optional[bool] = False):
        self.verbose = verbose
        self.input_port = input_port
        default = 0
        if input_port.type and "float" in input_port.type:
            default = 0.0
        self.curr_value = np.full(input_port.shape, default)

    def set_input_value(self, value: Union[str, int, np.ndarray]):
        """Set a new value at input port

        Args:
            value: Value to be set at Input Port
        """
        if self.verbose:
            print(f"    Input value in {self.input_port.id} set to {_val_info(value)}")
        self.curr_value = value

    def evaluate(
        self, parameters: Dict[str, Any] = None, array_format: str = FORMAT_DEFAULT
    ) -> Union[int, np.ndarray]:

        """Evaluates value at Input port based on parameters and array_format

        Args:
            parameters: Dictionary of  parameters
            array_format: It is a n-dimensional array

        Returns:
            value at Input port
        """
        if self.verbose:
            print(
                "    Evaluated %s with params %s =\t%s"
                % (
                    self.input_port,
                    _params_info(parameters),
                    _val_info(self.curr_value),
                )
            )
        return self.curr_value


class EvaluableNode:
    r"""Evaluates a :class:`~modeci_mdf.mdf.Node` during MDF graph execution.

    Args:
        node: A self contained unit of evaluation receiving input from other :class:`~modeci_mdf.mdf.Node`\(s) on
            :class:`~modeci_mdf.mdf.InputPort`\(s).
        verbose: If set to True Provides in-depth information else verbose message is not displayed
    """

    def __init__(self, node: Node, verbose: Optional[bool] = False):
        self.verbose = verbose
        self.node = node
        self.evaluable_inputs = {}
        self.evaluable_parameters = OrderedDict()
        self.evaluable_functions = OrderedDict()

        self.evaluable_outputs = {}

        all_known_vars = []

        all_known_vars += KNOWN_PARAMETERS

        for ip in node.input_ports:
            rip = EvaluableInput(ip, self.verbose)
            self.evaluable_inputs[ip.id] = rip
            all_known_vars.append(ip.id)
            # params_init[ip] = ip.curr_value

        for p in node.parameters:
            all_known_vars.append(p.id)

        """
        for p in node.parameters:
            ep = EvaluableParameter(p, self.verbose)
            self.evaluable_parameters[p.id] = ep
            all_known_vars.append(p.id)
            # params_init[s] = s.curr_value"""

        # TODO: the below checks for evaluability of functions and
        # parameters using known variables are very similar and could be
        # simplified with a function
        all_funcs = [f for f in node.functions]
        num_funcs_remaining = {f.id: None for f in node.functions}
        func_missing_vars = {f.id: [] for f in node.functions}

        # Order the functions into the correct sequence
        while len(all_funcs) > 0:
            f = all_funcs.pop(0)  # pop first off list
            if verbose:
                print(
                    "    Checking whether function: %s with args %s is sufficiently determined by known vars %s"
                    % (f.id, f.args, all_known_vars)
                )
            all_req_vars = []
            if f.args:
                for arg in f.args:
                    arg_expr = f.args[arg]

                    # If we are dealing with a list of symbols, each must treated separately
                    all_req_vars.extend(
                        [
                            v
                            for v in get_required_variables_from_expression(arg_expr)
                            if v not in f.args
                        ]
                    )
            if f.value is not None:
                all_req_vars.extend(
                    [
                        v
                        for v in get_required_variables_from_expression(f.value)
                        if f.args is None or v not in f.args
                    ]
                )

            all_present = [v in all_known_vars for v in all_req_vars]
            func_missing_vars[f.id] = {
                v for v in all_req_vars if v not in all_known_vars
            }

            if verbose:
                print(
                    "    Are all of %s in %s? %s"
                    % (all_req_vars, all_known_vars, all_present)
                )
            if all(all_present):
                rf = EvaluableFunction(f, self.verbose)
                self.evaluable_functions[f.id] = rf
                all_known_vars.append(f.id)
            #     params_init[f] = self.evaluable_functions[f.id].evaluate(
            #     params_init, array_format=FORMAT_DEFAULT
            # )
            else:
                # track the number of remaining functions each time f
                # is examined. If it's the same as last time, we know
                # every function was examined and nothing changed, so
                # we can stop because otherwise it will just infinitely
                # loop
                if num_funcs_remaining[f.id] == len(all_funcs):
                    func_missing_vars = {
                        f: ", ".join(v) for f, v in func_missing_vars.items()
                    }
                    raise ValueError(
                        "Error! Could not evaluate functions using known vars. The following vars are missing:\n\t"
                        + "\n\t".join(
                            f"{f}: {v}"
                            for f, v in func_missing_vars.items()
                            if len(v) > 0
                        )
                    )
                else:
                    num_funcs_remaining[f.id] = len(all_funcs)
                    all_funcs.append(f)

        all_params_to_check = [p for p in node.parameters]
        num_params_remaining = {p.id: None for p in node.parameters}
        param_missing_vars = {f.id: [] for f in node.parameters}

        if self.verbose:
            print("all_params_to_check: %s" % all_params_to_check)

        # Order the parameters into the correct sequence
        while len(all_params_to_check) > 0:
            p = all_params_to_check.pop(0)  # pop first off list

            if verbose:
                print(
                    "    Checking whether parameter: %s with args: %s, value: %s (%s) is sufficiently determined by known vars %s"
                    % (p.id, p.args, p.value, type(p.value), all_known_vars)
                )
            all_req_vars = []

            if p.value is not None and type(p.value) == str:
                all_req_vars.extend(
                    [
                        v
                        for v in get_required_variables_from_expression(p.value)
                        if p.args is None or v not in p.args
                    ]
                )

            if p.args is not None:
                for arg in p.args:
                    arg_expr = p.args[arg]
                    if isinstance(arg_expr, str):
                        all_req_vars.extend(
                            [
                                v
                                for v in get_required_variables_from_expression(
                                    arg_expr
                                )
                                if v not in p.args
                            ]
                        )

            all_known_vars_plus_this = all_known_vars + [p.id]
            all_present = [v in all_known_vars_plus_this for v in all_req_vars]
            param_missing_vars[p.id] = {
                v for v in all_req_vars if v not in all_known_vars
            }

            if verbose:
                print(
                    "    Are all of %s in %s? %s, i.e. %s"
                    % (
                        all_req_vars,
                        all_known_vars_plus_this,
                        all_present,
                        all(all_present),
                    )
                )
            if all(all_present):
                ep = EvaluableParameter(p, self.verbose)
                self.evaluable_parameters[p.id] = ep
                all_known_vars.append(p.id)

            else:
                if num_params_remaining[p.id] == len(all_params_to_check):
                    param_missing_vars = {
                        p: ", ".join(v) for p, v in param_missing_vars.items()
                    }
                    raise ValueError(
                        "Error! Could not evaluate parameters using known vars. The following vars are missing:\n\t"
                        + "\n\t".join(
                            f"{p}: {v}"
                            for p, v in param_missing_vars.items()
                            if len(v) > 0
                        )
                    )
                else:
                    num_params_remaining[p.id] = len(all_params_to_check)
                    all_params_to_check.append(p)  # Add back to end of list...

        for op in node.output_ports:
            rop = EvaluableOutput(op, self.verbose)
            self.evaluable_outputs[op.id] = rop

    def evaluate(
        self,
        time_increment: Union[int, float] = None,
        array_format: str = FORMAT_DEFAULT,
    ):
        """
        Evaluate the Node for one time-step

        Args:
            time_increment: The time-increment to use for this evaluation.
            array_format: The format to use for arrays.

        """

        if self.verbose:
            print(
                "\n  ---------------\n  Evaluating Node: %s with %s"
                % (self.node.id, [p.id for p in self.node.parameters])
            )
        curr_params = {}

        for eip in self.evaluable_inputs:
            i = self.evaluable_inputs[eip].evaluate(
                curr_params, array_format=array_format
            )
            curr_params[eip] = i

        # First set params to previous parameter values for use in funcs and states...
        for ep in self.evaluable_parameters:

            curr_params[ep] = self.evaluable_parameters[ep].get_current_value(
                curr_params, array_format=array_format
            )

        for ef in self.evaluable_functions:
            curr_params[ef] = self.evaluable_functions[ef].evaluate(
                curr_params, array_format=array_format
            )

        # Now evaluate and set params to new parameter values for use in output...
        for ep in self.evaluable_parameters:
            curr_params[ep] = self.evaluable_parameters[ep].evaluate(
                curr_params, time_increment=time_increment, array_format=array_format
            )

        for eop in self.evaluable_outputs:
            self.evaluable_outputs[eop].evaluate(curr_params, array_format=array_format)

    def get_output(self, id: str = None) -> Union[int, np.ndarray, Tuple]:
        """Get value at output port for given output port's id

        Args:
            id: Unique identifier of the output port. If None, return a tuple for all output ports.

        Returns:
            value at the output port. If id is None, return all outputs as a tuple. If there is only
            one output, return just its value.

        """
        if id is not None:
            for rop in self.evaluable_outputs:
                if rop == id:
                    return self.evaluable_outputs[rop].curr_value
        else:
            outputs = tuple(
                self.evaluable_outputs[rop].curr_value for rop in self.evaluable_outputs
            )
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs


class EvaluableGraph:
    r"""
    Evaluates a :class:`~modeci_mdf.mdf.Graph` with the MDF execution engine. This is the top-level interface to the execution engine.

    Args:
        graph: A directed graph consisting of :class:`~modeci_mdf.mdf.Node`\(s) connected via :class:`~modeci_mdf.mdf.Edge`\(s)
        verbose: If set to True Provides in-depth information else verbose message is not displayed

    """

    def __init__(self, graph: Graph, verbose: Optional[bool] = False):
        self.verbose = verbose
        print("\nInit graph: %s" % graph.id)
        self.graph = graph
        self.enodes = {}
        self.root_nodes = []
        self.output_nodes = []
        self.order_of_execution = []

        # Get the root (input nodes) of the graph. We will assume all are root nodes
        # and then remove those that have edges to them.
        for node in graph.nodes:
            if self.verbose:
                print("\n  Init node: %s" % node.id)
            en = EvaluableNode(node, self.verbose)
            self.enodes[node.id] = en
            self.root_nodes.append(node.id)

        for edge in graph.edges:
            if (
                edge.receiver in self.root_nodes
            ):  # It could have been already removed...
                self.root_nodes.remove(edge.receiver)

        self.ordered_edges = []
        evaluated_nodes = []
        for rn in self.root_nodes:
            evaluated_nodes.append(rn)

        edges_to_eval = [edge for edge in self.graph.edges]

        while len(edges_to_eval) > 0:
            edge = edges_to_eval.pop(0)
            if edge.sender not in evaluated_nodes:
                edges_to_eval.append(edge)  # Add back to end of list...
            else:
                self.ordered_edges.append(edge)
                evaluated_nodes.append(edge.receiver)

        if self.graph.conditions is not None:
            if self.graph.conditions.node_specific is None:
                conditions = {}
            else:
                conditions = {
                    self.graph.get_node(node): self.parse_condition(cond)
                    for node, cond in self.graph.conditions.node_specific.items()
                }

            termination_conds = {}
            if self.graph.conditions.termination is not None:
                for scale, cond in self.graph.conditions.termination.items():
                    cond = self.parse_condition(cond)

                    # check for TimeScale in form of enum or equivalent unambiguous strings
                    try:
                        scale = re.match(time_scale_str_regex, scale).groups()[1]
                    except (AttributeError, IndexError, TypeError):
                        pass

                    try:
                        termination_conds[graph_scheduler.TimeScale[scale]] = cond
                    except KeyError:
                        termination_conds[scale] = cond
        else:
            conditions = {}
            termination_conds = {}
        self.scheduler = graph_scheduler.Scheduler(
            graph=self.graph.dependency_dict,
            conditions=conditions,
            termination_conds=termination_conds,
        )

        # We also need to get the output nodes
        self.output_nodes = [
            n
            for n in self.graph.nodes
            if n.id not in (e.sender for e in self.graph.edges)
        ]

        # Lets also get the corresponding EvaluableNode for outputs
        self.output_enodes = [self.enodes[n.id] for n in self.output_nodes]

    def evaluate(
        self,
        time_increment: Union[int, float] = None,
        array_format: str = FORMAT_DEFAULT,
        initializer: Optional[Dict[str, Any]] = None,
    ):
        """
        Evaluates a :class:`~modeci_mdf.mdf.Graph`. This is the top-level interface to the execution engine.

        Args:
            time_increment: Time step for next execution
            array_format: A n-dimensional array
            initializer: sets the initial value of parameters of the node

        """
        # Any values that are set via the passed in initalizer, set their values. This lets us avoid creating
        # dummy input nodes with parameters for evaluating the graph
        for en_id, en in self.enodes.items():
            for inp_name, inp in en.evaluable_inputs.items():
                if initializer and inp_name in initializer:
                    inp.set_input_value(initializer[inp_name])

        print(
            "Evaluating graph: %s, root nodes: %s, with array format %s"
            % (self.graph.id, self.root_nodes, array_format)
        )
        str_conds_nb = "\n  ".join(
            [
                f"{node.id}: {cond}"
                for node, cond in self.scheduler.conditions.conditions.items()
            ]
        )
        str_conds_term = "\n  ".join(
            [
                f"{scale}: {cond}"
                for scale, cond in self.scheduler.termination_conds.items()
            ]
        )
        if self.verbose:
            print(" node-based conditions\n  %s" % str_conds_nb)
            print(" termination conditions\n  %s" % str_conds_term)

        incoming_edges = {n: set() for n in self.graph.nodes}
        for edge in self.graph.edges:
            incoming_edges[self.graph.get_node(edge.receiver)].add(edge)

        for ts in self.scheduler.run():
            if self.verbose:
                print(
                    "> Evaluating time step: %s"
                    % self.scheduler.get_clock(None).simple_time
                )
            for node in ts:
                self.order_of_execution.append(node.id)
                for edge in incoming_edges[node]:
                    self.evaluate_edge(
                        edge, time_increment=time_increment, array_format=array_format
                    )
                self.enodes[node.id].evaluate(
                    time_increment=time_increment, array_format=array_format
                )

        if self.verbose:
            print("> Order of execution of nodes: %s" % self.order_of_execution)
            print("\n Trial terminated")

    def evaluate_edge(
        self,
        edge: Edge,
        time_increment: Union[int, float] = None,
        array_format: str = FORMAT_DEFAULT,
    ):
        """Evaluates edges in graph

        Args:
            time_increment: Time step for next execution
            array_format: A n-dimensional array

        """
        pre_node = self.enodes[edge.sender]
        post_node = self.enodes[edge.receiver]
        value = pre_node.evaluable_outputs[edge.sender_port].curr_value
        weight = (
            1
            if not edge.parameters or not "weight" in edge.parameters
            else edge.parameters["weight"]
        )

        if self.verbose:
            print(
                "  Edge %s connects %s to %s, passing %s with weight %s"
                % (
                    edge.id,
                    pre_node.node.id,
                    post_node.node.id,
                    _val_info(value),
                    _val_info(weight),
                )
            )
        if (type(weight) == int or type(weight) == float) and weight == 1:
            input_value = value
        else:
            input_value = weight * value
        post_node.evaluable_inputs[edge.receiver_port].set_input_value(input_value)

    def parse_condition(
        self, condition: Union[Condition, Dict]
    ) -> graph_scheduler.Condition:
        """Convert the condition in a specific format

        Args:
            condition: Specify the condition under which a Component should be allowed to execute

        Returns:
            Condition in specific format

        """

        def get_custom_parameter_getter(eobj):
            # try to pick a default based on expected shape of the
            # evaluable object before it has ever been executed
            for d in [
                lambda: np.zeros(eobj.input_port.shape),
                lambda: np.zeros(eobj.output_port.shape),
                lambda: eobj.parameter.default_initial_value,
            ]:
                try:
                    default = d()
                except AttributeError:
                    pass
                else:
                    break
            else:
                default = 0

            def getter(dependency, parameter):
                res = eobj.curr_value
                if res is None:
                    return default
                else:
                    return res

            return getter

        def update_condition_arguments(args, condition_type):
            # mdf format prefers the key name for all conditions that use it or
            # its value as an __init__ argument
            combined_condition_arguments = {"dependencies": "dependency"}
            sig = inspect.signature(condition_type)

            for preferred, actual in combined_condition_arguments.items():
                if (
                    preferred in condition.kwargs
                    and preferred not in sig.parameters
                    and actual in sig.parameters
                ):
                    args[actual] = args[preferred]
                    del args[preferred]

            if "custom_parameter_getter" in sig.parameters:
                try:
                    dependency = args["dependency"]
                    parameter = args["parameter"]
                except KeyError as e:
                    raise ValueError(
                        "Threshold condition did not specify dependency or parameter"
                    ) from e

                # self is EvaluableGraph condition is running under,
                enode = self.enodes[dependency.id]
                evaluable_objects = [
                    enode.evaluable_inputs,
                    enode.evaluable_parameters,
                    enode.evaluable_functions,
                    enode.evaluable_outputs,
                ]
                valid_parameters = itertools.chain.from_iterable(
                    [obj for obj in evaluable_objects]
                )
                args["custom_parameter_validator"] = (
                    lambda dependency, parameter: parameter in valid_parameters
                )
                # assumes a unique parameter id among evaluable objects for
                # dependency, or will resolve in favor of first with that id
                for obj in evaluable_objects:
                    try:
                        obj = obj[parameter]
                    except KeyError:
                        pass
                    else:
                        args["custom_parameter_getter"] = get_custom_parameter_getter(
                            obj
                        )
                        break
                else:
                    raise ValueError(
                        f"No {parameter} evaluable object for {dependency}, options: {list(valid_parameters)}"
                    )

            return args

        # if specified as dict
        try:
            args = condition["kwargs"]
        except (IndexError, TypeError, KeyError):
            args = {}

        try:
            condition = Condition(condition["type"], **args)
        except (IndexError, TypeError, KeyError):
            pass

        cond_type = condition.type
        cond_args = copy.copy(condition.kwargs)

        try:
            typ = getattr(graph_scheduler.condition, cond_type)
        except AttributeError as e:
            raise ValueError("Unsupported condition type: %s" % cond_type) from e
        except TypeError as e:
            raise TypeError("Invalid condition dictionary: %s" % condition) from e

        for k, v in cond_args.items():
            new_v = self.graph.get_node(v)
            if new_v is not None:
                # arg is a node id
                cond_args[k] = new_v

            try:
                if isinstance(v, list):
                    # arg is a list all of conditions
                    new_v = [self.parse_condition(item) for item in v]
                else:
                    # arg is another condition
                    new_v = self.parse_condition(v)
            except (AttributeError, TypeError, ValueError):
                try:
                    # value may be a string representing a TimeScale
                    cond_args[k] = getattr(
                        graph_scheduler.TimeScale,
                        re.match(time_scale_str_regex, v).groups()[1],
                    )
                except (AttributeError, IndexError, TypeError):
                    pass
            else:
                cond_args[k] = new_v

        cond_args = update_condition_arguments(cond_args, typ)

        try:
            return typ(**cond_args)
        except TypeError as e:
            sig = inspect.signature(typ)

            try:
                var_positional_arg = [
                    name
                    for name, param in sig.parameters.items()
                    if param.kind is inspect.Parameter.VAR_POSITIONAL
                ][0]
            except IndexError:
                # other unhandled situation
                raise e
            else:
                try:
                    cond_args[var_positional_arg]
                except KeyError:
                    # error is due to missing required parameter,
                    # not named variable positional argument
                    raise TypeError(f"Condition '{typ.__name__}': {e}")
                else:
                    return typ(
                        *cond_args[var_positional_arg],
                        **{
                            k: v
                            for k, v in cond_args.items()
                            if k != var_positional_arg
                        },
                    )


def main(example_file: str, array_format: str = FORMAT_NUMPY, verbose: bool = False):
    """
    Main entry point for execution engine.

    Args:
        example_file: The MDF file to execute.
        array_format: The format of arrays to use. Allowed values: 'numpy' or 'tensorflow'.
        verbose: Whether to print output to standard out during execution.

    """

    from modeci_mdf.utils import load_mdf, print_summary

    mod_graph = load_mdf(example_file).graphs[0]

    if verbose:
        print("Loaded Graph:")
        print_summary(mod_graph)

        print("------------------")
    eg = EvaluableGraph(mod_graph, verbose)
    eg.evaluate(array_format=array_format)

    return eg


if __name__ == "__main__":

    example_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "examples/MDF/Simple.json"
    )
    verbose = True
    if len(sys.argv) >= 2:
        example_file = sys.argv[1]

    if "-v" in sys.argv:
        verbose = True
    else:
        verbose = False

    from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

    print("Executing MDF file %s with scheduler" % example_file)

    main(example_file, array_format=format, verbose=verbose)
