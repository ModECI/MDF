"""
    Example of ModECI MDF - Showing usage of parameters and functions
"""

from modeci_mdf.mdf import *

from modeci_mdf.utils import simple_connect, print_summary

import abcd_python as abcd
import os


def main():
    mod = Model(id="ParametersFunctions")
    mod_graph = Graph(id="params_funcs_example")
    mod.graphs.append(mod_graph)

    node0 = Node(id="node0", metadata={"color": ".8 .8 .8"})

    node0.parameters.append(
        Parameter(
            id="param_fixed_int",
            value=1,
        )
    )
    node0.parameters.append(Parameter(id="param_fixed_float", value=2.0))
    node0.parameters.append(Parameter(id="param_array_list", value=[3, 4.0]))
    node0.parameters.append(
        Parameter(id="param_expression", value="param_fixed_int + param_fixed_float")
    )
    node0.parameters.append(Parameter(id="param_stateful", value="param_stateful + 1"))

    param_func1 = Parameter(
        id="param_function",
        function="linear",
        args={"variable0": 1, "slope": 2, "intercept": 3},
    )
    node0.parameters.append(param_func1)

    node0.parameters.append(
        Parameter(id="param_time_deriv", default_initial_value=0, time_derivative="1")
    )

    func_func1 = Function(
        id="function_inbuilt_with_args",
        function="linear",
        args={"variable0": 1, "slope": 2, "intercept": 3},
    )
    node0.functions.append(func_func1)

    func_func1 = Function(
        id="function_with_value_args",
        value="A + B + C",
        args={"A": 1, "B": 2, "C": 3},
    )
    node0.functions.append(func_func1)

    func_func2 = Function(
        id="function_with_deep_args",
        function="linear",
    )
    # node0.functions.append(func_func2)

    mod_graph.nodes.append(node0)

    print("------------------")
    print(mod.to_yaml())
    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    # print_summary(mod_graph)

    import sys

    if "-run" in sys.argv:
        verbose = True
        # verbose = False
        from modeci_mdf.execution_engine import EvaluableGraph
        from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY
        eg = EvaluableGraph(mod_graph, verbose=verbose)
        eg.evaluate(array_format=format)

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="%s" % mod_graph.id,
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
