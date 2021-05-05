import sys
import sympy

from modeci_mdf.standard_functions import mdf_functions, create_python_expression

from neuromllite.utils import evaluate as evaluate_params_nmllite
from neuromllite.utils import _params_info, _val_info
from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

from collections import OrderedDict

FORMAT_DEFAULT = FORMAT_NUMPY

import modeci_mdf.onnx_functions as onnx_ops


def evaluate_expr(expr, func_params, array_format, verbose=False):

    e = evaluate_params_nmllite(
        expr, func_params, array_format=array_format, verbose=verbose
    )
    if type(e) == str:
        raise Exception(
            "Error! Could not evaluate expression [%s] with params %s, returned [%s] which is a %s"
            % (expr, _params_info(func_params), e, type(e))
        )
    return e


class EvaluableFunction:
    def __init__(self, function, verbose=False):
        self.verbose = verbose
        self.function = function

    def evaluate(self, parameters, array_format=FORMAT_DEFAULT):

        expr = None
        for f in mdf_functions:
            if self.function.function == f:
                expr = create_python_expression(mdf_functions[f]["expression_string"])
        if not expr:
            raise "Unknown function: {}. Known functions: {}".format(
                self.function.function,
                mdf_functions.keys,
            )

        func_params = {}
        func_params.update(parameters)
        if self.verbose:
            print(
                "    Evaluating %s with %s, i.e. [%s]"
                % (self.function, _params_info(func_params), expr)
            )
        for arg in self.function.args:
            func_params[arg] = evaluate_expr(
                self.function.args[arg],
                func_params,
                verbose=False,
                array_format=array_format,
            )
            if self.verbose:
                print(
                    "      Arg: {} became: {}".format(arg, _val_info(func_params[arg]))
                )

        # If this is an ONNX operation, evaluate it withouth neuromlite.
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


class EvaluableState:
    def __init__(self, state, verbose=False):
        self.verbose = verbose
        self.state = state
        self.curr_value = 0

    def evaluate(self, parameters, time_increment=None, array_format=FORMAT_DEFAULT):
        if self.verbose:
            print(
                "    Evaluating {} with {} ".format(
                    self.state, _params_info(parameters)
                )
            )

        if self.state.value:

            self.curr_value = evaluate_expr(
                self.state.value,
                parameters,
                verbose=False,
                array_format=array_format,
            )
        else:
            if time_increment == None:

                self.curr_value = evaluate_expr(
                    self.state.default_initial_value,
                    parameters,
                    verbose=False,
                    array_format=array_format,
                )
            else:
                td = evaluate_expr(
                    self.state.time_derivative,
                    parameters,
                    verbose=False,
                    array_format=array_format,
                )
                self.curr_value += td * time_increment

        if self.verbose:
            print(
                "    Evaluated %s with %s \n       =\t%s"
                % (self.state, _params_info(parameters), _val_info(self.curr_value))
            )
        return self.curr_value


class EvaluableOutput:
    def __init__(self, output_port, verbose=False):
        self.verbose = verbose
        self.output_port = output_port

    def evaluate(self, parameters, array_format=FORMAT_DEFAULT):
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
    def __init__(self, input_port, verbose=False):
        self.verbose = verbose
        self.input_port = input_port
        self.curr_value = 0

    def set_input_value(self, value):
        if self.verbose:
            print(f"    Input value in {self.input_port.id} set to {_val_info(value)}")
        self.curr_value = value

    def evaluate(self, parameters, array_format=FORMAT_DEFAULT):
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
    def __init__(self, node, verbose=False):
        self.verbose = verbose
        self.node = node
        self.evaluable_inputs = {}
        self.evaluable_functions = OrderedDict()
        self.evaluable_states = OrderedDict()
        self.evaluable_outputs = {}

        all_known_vars = []
        if node.parameters:
            for p in node.parameters:
                all_known_vars.append(p)

        for ip in node.input_ports:
            rip = EvaluableInput(ip, self.verbose)
            self.evaluable_inputs[ip.id] = rip
            all_known_vars.append(ip.id)

        for s in node.states:
            es = EvaluableState(s, self.verbose)
            self.evaluable_states[s.id] = es
            all_known_vars.append(s.id)

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
            for arg in f.args:
                arg_expr = f.args[arg]

                # If we are dealing with a list of symbols, each must treated separately
                if type(arg_expr) == str and arg_expr[0] == "[" and arg_expr[-1] == "]":
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
            else:
                if len(all_funcs) == 0:
                    raise Exception(
                        "Error! Could not evaluate function: %s with args %s using known vars %s"
                        % (f.id, f.args, all_known_vars)
                    )
                else:
                    all_funcs.append(f)  # Add back to end of list...

        for op in node.output_ports:
            rop = EvaluableOutput(op, self.verbose)
            self.evaluable_outputs[op.id] = rop

    def initialize(self):
        pass

    def evaluate(self, time_increment=None, array_format=FORMAT_DEFAULT):

        if self.verbose:
            print(
                "  Evaluating Node: %s with %s"
                % (self.node.id, _params_info(self.node.parameters))
            )
        curr_params = {}
        if self.node.parameters:
            curr_params.update(self.node.parameters)

        for eip in self.evaluable_inputs:
            i = self.evaluable_inputs[eip].evaluate(
                curr_params, array_format=array_format
            )
            curr_params[eip] = i

        # First set params to previous state values for use in funcs and states...
        for es in self.evaluable_states:
            curr_params[es] = self.evaluable_states[es].curr_value

        for ef in self.evaluable_functions:
            curr_params[ef] = self.evaluable_functions[ef].evaluate(
                curr_params, array_format=array_format
            )

        # Now evaluate and set params to new state values for use in output...
        for es in self.evaluable_states:
            curr_params[es] = self.evaluable_states[es].evaluate(
                curr_params, time_increment=time_increment, array_format=array_format
            )

        for eop in self.evaluable_outputs:
            self.evaluable_outputs[eop].evaluate(curr_params, array_format=array_format)

    def get_output(self, id):
        for rop in self.evaluable_outputs:
            if rop == id:
                return self.evaluable_outputs[rop].curr_value


class EvaluableGraph:
    def __init__(self, graph, verbose=False):
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

    def evaluate(
        self, time_increment=None, array_format=FORMAT_DEFAULT, initializer=None
    ):

        # Any values that are set via the passed in initalizer, set their values. This lets us avoid creating
        # dummy input nodes with parameters for evaluating the graph
        for en_id, en in self.enodes.items():
            for inp_name, inp in en.evaluable_inputs.items():
                if initializer and inp_name in initializer:
                    inp.set_input_value(initializer[inp_name])

        for rn in self.root_nodes:
            self.enodes[rn].evaluate(
                array_format=array_format, time_increment=time_increment
            )

        for edge in self.ordered_edges:
            self.evaluate_edge(edge, array_format=array_format)
            self.enodes[edge.receiver].evaluate(
                time_increment=time_increment, array_format=array_format
            )

    def evaluate_edge(self, edge, time_increment=None, array_format=FORMAT_DEFAULT):
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
        post_node.evaluable_inputs[edge.receiver_port].set_input_value(value * weight)


from neuromllite.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW


def main(example_file, array_format=FORMAT_NUMPY, verbose=False):

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

    example_file = "../../examples/Simple.json"
    verbose = True
    if len(sys.argv) == 2:
        example_file = sys.argv[1]
        verbose = True
        verbose = False

    main(example_file, verbose)
