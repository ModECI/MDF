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
import inspect
import os
import re
import sys
import sympy
import numpy as np


import graph_scheduler

from modeci_mdf.functions.standard import mdf_functions, create_python_expression
from modeci_mdf.utils import is_number

from neuromllite.utils import evaluate as evaluate_params_nmllite
from neuromllite.utils import _params_info, _val_info
from neuromllite.utils import FORMAT_NUMPY

from collections import OrderedDict
from typing import Union, List, Dict, Optional, Any
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


FORMAT_DEFAULT = FORMAT_NUMPY

KNOWN_PARAMETERS = ["constant"]


def evaluate_expr(
    expr: Union[str, List[str], np.ndarray, "tf.tensor"] = None,
    func_params: Dict[str, Any] = None,
    array_format: str = FORMAT_DEFAULT,
    verbose: Optional[bool] = False,
) -> np.ndarray:

    """Evaluates an expression given in string format and a :code:`dict` of parameters.

    Args:
        expr: Expression or list of expressions to be evaluated
        func_params: A dict of parameters (e.g. :code:`{'weight': 2}`)
        array_format: It can be a n-dimensional array or a tensor
        verbose: If set to True provides in-depth information else verbose message is not displayed

    Returns:
        n-dimensional array

    """

    e = evaluate_params_nmllite(
        expr, func_params, array_format=array_format, verbose=verbose
    )
    if type(e) == str and e not in KNOWN_PARAMETERS:
        raise Exception(
            "Error! Could not evaluate expression [%s] with params %s, returned [%s] which is a %s"
            % (expr, _params_info(func_params), e, type(e))
        )
    return e


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

        if self.function.function:

            for f in mdf_functions:
                if self.function.function == f:
                    expr = create_python_expression(
                        mdf_functions[f]["expression_string"]
                    )

        else:
            expr = self.function.value
            # raise "Unknown function: {}. Known functions: {}".format(
            #    self.function.function,
            #    mdf_functions.keys,
            # )

        func_params = {}
        func_params.update(parameters)
        if self.verbose:
            print(
                "    Evaluating %s with %s, i.e. [%s]"
                % (self.function, _params_info(func_params), expr)
            )
        if self.function.args:

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

        # If this is an ONNX operation, evaluate it without neuromlite.

        if "onnx_ops." in expr:
            # Get the ONNX function
            onnx_function = getattr(onnx_ops, expr.split("(")[0].split(".")[-1])

            # ONNX functions expect input args or kwargs first, followed by parameters (called attributes in ONNX) as
            # kwargs. Lets construct this.
            kwargs_for_onnx = {}
            for kw, arg_expr in self.function.args.items():

                # If this arg is a list of args, we are dealing with a variadic argument. Expand these
                if type(arg_expr) == str and arg_expr[0] == "[" and arg_expr[-1] == "]":
                    # Use the Python interpreter to parse this into a List[str]
                    arg_expr_list = eval(arg_expr)
                    kwargs_for_onnx.update({a: func_params[a] for a in arg_expr_list})
                else:
                    kwargs_for_onnx[kw] = func_params[kw]

            # Now add anything in parameters that isn't already specified as an input argument
            for kw, arg in parameters.items():
                if kw not in self.function.args.values():
                    kwargs_for_onnx[kw] = arg

            self.curr_value = onnx_function(**kwargs_for_onnx)
        elif "actr." in expr:
            actr_function = getattr(actr_funcs, expr.split("(")[0].split(".")[-1])
            self.curr_value = actr_function(
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
    Evaluates the current value of a :class:`~modeci_mdf.mdf.Parameter` during MDF graph execution.

    Args:
        parameter: The parameter to evaluate during execution.
        verbose: Whether to print output of parameter calculations.
    """

    DEFAULT_INIT_VALUE = 0  # Temporary!

    def __init__(self, parameter: Parameter, verbose: bool = False):
        self.verbose = verbose
        self.parameter = parameter

        if self.parameter.default_initial_value is not None:
            if is_number(self.parameter.default_initial_value):

                self.curr_value = float(self.parameter.default_initial_value)

            else:

                self.curr_value = self.parameter.default_initial_value

        else:
            self.curr_value = None

    def get_current_value(
        self, parameters: Dict[str, Any], array_format: str = FORMAT_DEFAULT
    ) -> Any:
        """
        Get the current value of the parameter; evaluates the expression if needed.

        Args:
            parameters: a dictionary  of parameters and their values that may or may not be needed to evaluate this
                parameter.
            array_format: The array format to use (either :code:`'numpy'` or :code:`tensorflow'`).

        Returns:
            The evaluated value of the parameter.
        """

        # FIXME: Shouldn't this just call self.evaluate, seems like there is redundant code here?
        if self.curr_value is None:

            if self.parameter.value is not None:
                if self.parameter.is_stateful():

                    if self.parameter.default_initial_value is not None:
                        return self.parameter.default_initial_value
                    else:
                        return self.DEFAULT_INIT_VALUE
                else:
                    ips = {}
                    ips.update(parameters)
                    ips[self.parameter.id] = self.DEFAULT_INIT_VALUE
                    self.curr_value = evaluate_expr(
                        self.parameter.value,
                        ips,
                        verbose=False,
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
            parameters: a dictionary  of parameters and their values that may or may not be needed to evaluate this
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
                    self.parameter, _params_info(parameters)
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

            # If this is an ONNX operation, evaluate it without neuromlite.
            if "onnx_ops." in expr:
                # Get the ONNX function
                onnx_function = getattr(onnx_ops, expr.split("(")[0].split(".")[-1])

                # ONNX functions expect input args or kwargs first, followed by parameters (called attributes in ONNX) as
                # kwargs. Lets construct this.
                kwargs_for_onnx = {}
                for kw, arg_expr in self.parameter.args.items():

                    # If this arg is a list of args, we are dealing with a variadic argument. Expand these
                    if (
                        type(arg_expr) == str
                        and arg_expr[0] == "["
                        and arg_expr[-1] == "]"
                    ):
                        # Use the Python interpreter to parse this into a List[str]
                        arg_expr_list = eval(arg_expr)
                        kwargs_for_onnx.update(
                            {a: func_params[a] for a in arg_expr_list}
                        )
                    else:
                        kwargs_for_onnx[kw] = func_params[kw]

                # Now add anything in parameters that isn't already specified as an input argument
                for kw, arg in parameters.items():

                    if (
                        kw not in self.parameter.args.values()
                        and kw != self.parameter.id
                        and kw != "__builtins__"
                    ):
                        kwargs_for_onnx[kw] = arg
                if self.verbose:
                    print(
                        "%s is evaluating ONNX function %s with %s"
                        % (self.parameter.id, expr, kwargs_for_onnx)
                    )
                self.curr_value = onnx_function(**kwargs_for_onnx)

            elif "actr." in expr:
                actr_function = getattr(actr_funcs, expr.split("(")[0].split(".")[-1])
                self.curr_value = actr_function(
                    *[func_params[arg] for arg in self.parameter.args]
                )
            else:
                self.curr_value = evaluate_expr(
                    expr, func_params, verbose=self.verbose, array_format=array_format
                )
        else:
            if time_increment == None:

                self.curr_value = evaluate_expr(
                    self.parameter.default_initial_value,
                    parameters,
                    verbose=False,
                    array_format=array_format,
                )

            else:
                td = evaluate_expr(
                    self.parameter.time_derivative,
                    parameters,
                    verbose=False,
                    array_format=array_format,
                )

                self.curr_value += td * time_increment

        if self.verbose:
            print(
                "    Evaluated %s with %s \n       =\t%s"
                % (self.parameter, _params_info(parameters), _val_info(self.curr_value))
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
        self.curr_value = 0

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
                "    Evaluated %s with %s =\t%s"
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

        all_funcs = [f for f in node.functions]

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

                    # some non-expression/str types will crash in sympy.simplify
                    if not isinstance(arg_expr, (sympy.Expr, str)):
                        continue

                    # If we are dealing with a list of symbols, each must treated separately
                    if (
                        type(arg_expr) == str
                        and arg_expr[0] == "["
                        and arg_expr[-1] == "]"
                    ):
                        # Use the Python interpreter to parse this into a List[str]
                        arg_expr_list = eval(arg_expr)
                    else:
                        arg_expr_list = [arg_expr]

                    for e in arg_expr_list:
                        func_expr = sympy.simplify(e)
                        all_req_vars.extend([str(s) for s in func_expr.free_symbols])

            all_present = [v in all_known_vars for v in all_req_vars]

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
                if len(all_funcs) == 0:
                    raise Exception(
                        "Error! Could not evaluate function: %s with args %s using known vars %s"
                        % (f.id, f.args, all_known_vars)
                    )
                else:
                    all_funcs.append(f)
        all_params_to_check = [p for p in node.parameters]
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
                param_expr = sympy.simplify(p.value)
                all_req_vars.extend([str(s) for s in param_expr.free_symbols])

            if p.args is not None:
                for arg in p.args:
                    arg_expr = p.args[arg]

                    # If we are dealing with a list of symbols, each must treated separately
                    if (
                        type(arg_expr) == str
                        and arg_expr[0] == "["
                        and arg_expr[-1] == "]"
                    ):
                        # Use the Python interpreter to parse this into a List[str]
                        arg_expr_list = eval(arg_expr)
                    else:
                        arg_expr_list = [arg_expr]

                    for e in arg_expr_list:
                        param_expr = sympy.simplify(e)
                        all_req_vars.extend([str(s) for s in param_expr.free_symbols])

            all_known_vars_plus_this = all_known_vars + [p.id]
            all_present = [v in all_known_vars_plus_this for v in all_req_vars]

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
                if len(all_params_to_check) == 0:
                    raise Exception(
                        "Error! Could not evaluate parameter: %s with args %s using known vars %s"
                        % (p.id, p.args, all_known_vars_plus_this)
                    )
                else:
                    all_params_to_check.append(p)  # Add back to end of list...

        for op in node.output_ports:
            rop = EvaluableOutput(op, self.verbose)
            self.evaluable_outputs[op.id] = rop

    def initialize(self):
        pass

    def evaluate(
        self,
        time_increment: Union[int, float] = None,
        array_format: str = FORMAT_DEFAULT,
    ):

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

    def get_output(self, id: str) -> Union[int, np.ndarray]:
        """Get value at output port for given output port's id

        Args:
            id: Unique identifier of the output port

        Returns:
            value at the output port

        """
        for rop in self.evaluable_outputs:
            if rop == id:
                return self.evaluable_outputs[rop].curr_value


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
            conditions = {
                self.graph.get_node(node): self.parse_condition(cond)
                for node, cond in self.graph.conditions.node_specific.items()
            }

            termination_conds = {
                scale: self.parse_condition(cond)
                for scale, cond in self.graph.conditions.termination.items()
            }
        else:
            conditions = {}
            termination_conds = {}

        self.scheduler = graph_scheduler.Scheduler(
            graph=self.graph.dependency_dict,
            conditions=conditions,
            termination_conds=termination_conds,
        )

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
            "\nEvaluating graph: %s, root nodes: %s, with array format %s"
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
                for edge in incoming_edges[node]:
                    self.evaluate_edge(
                        edge, time_increment=time_increment, array_format=array_format
                    )
                self.enodes[node.id].evaluate(
                    time_increment=time_increment, array_format=array_format
                )

        if self.verbose:
            print("Trial terminated")

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
        input_value = value if weight == 1 else value * weight
        post_node.evaluable_inputs[edge.receiver_port].set_input_value(input_value)

    def parse_condition(self, condition: Condition) -> graph_scheduler.Condition:
        """Convert the condition in a specific format

        Args:
            condition: Specify the condition under which a Component should be allowed to execute

        Returns:
            Condition in specific format

        """
        try:
            cond_type = condition["type"]
        except TypeError:
            cond_type = condition.type

        try:
            cond_args = condition["args"]
        except TypeError:
            cond_args = condition.args

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
                        re.match(r"TimeScale\.(.*)", v).groups()[0],
                    )
                except (AttributeError, IndexError, TypeError):
                    pass
            else:
                cond_args[k] = new_v

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


from neuromllite.utils import FORMAT_NUMPY


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

    from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

    format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

    print("Executing MDF file %s with scheduler" % example_file)

    main(example_file, array_format=format, verbose=verbose)
