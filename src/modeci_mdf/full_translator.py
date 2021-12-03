import json
import ntpath
from modeci_mdf.functions.standard import mdf_functions, create_python_expression
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.execution_engine import EvaluableGraph
import glom

expression_items = ["+", "*", "-", "/", "%", "(", ")"]


def convert_states_to_stateful_parameters(file_path: str = None, dt=5e-05):

    """Translates json file if with states to json file with stateful_parameters, otherwise unchanged
    Args:
            file_path: File in Json Format
    Returns:
            file in json format
    """
    f = open(file_path)
    data = json.load(f)

    filtered_list = [
        "parameters",
        "metadata",
        "functions",
        "states",
        "output_ports",
        "input_ports",
        "notes",
    ]
    all_nodes = []
    all_keys = []

    def keysExtractor(nested_dictionary):
        for k, v in nested_dictionary.items():

            if isinstance(v, dict) and k in "conditions":
                continue
            elif isinstance(v, dict):
                all_keys.append(k)
                if isinstance(v, dict) and k in "nodes":
                    break
                elif isinstance(v, dict):
                    keysExtractor(v)

    keysExtractor(data)
    path = ".".join(all_keys)

    def nodeExtractor(nested_dictionary: Dict[str, Any] = None):
        """Extracts all the node objects in the graph
        Args:
                nested_dictionary: input data
        Returns:
                Dictionary of node objects
        """
        for k, v in nested_dictionary.items():
            if isinstance(v, dict) and k in "nodes":
                all_nodes.append(v.keys())
            elif isinstance(v, dict):
                nodeExtractor(v)

    nodeExtractor(data)
    nodes_dict = dict.fromkeys(all_nodes[0])

    for key in list(nodes_dict.keys()):
        nodes_dict[key] = {}

    def parameterExtractor(nested_dictionary: Dict[str, Any] = None):
        """Extracts Parameters, states, functions, input and output ports at each node object
        Args:
                nested_dictionary: Input Data
        Returns:
                stores states, parameters, functions, input and output ports
        """
        for k, v in nested_dictionary.items():
            if isinstance(v, dict) and k in list(nodes_dict.keys()):
                for kk, vv in v.items():

                    if (isinstance(vv, dict) and kk in filtered_list) or (
                        isinstance(vv, str) and kk in filtered_list
                    ):

                        nodes_dict[k][kk] = vv
            if isinstance(v, dict):
                parameterExtractor(v)

    parameterExtractor(data)

    arg_dict = {}

    def get_arguments(d: Dict[str, Any] = None):
        """Extracts all parameters including stateful,dt for each node object
        Args:
                d: Node level dictionary with filtered keys
        Returns:
                all parameters for each node object
        """
        for key in d.keys():
            vi = []
            flag = 0
            if "parameters" in d[key].keys():
                vi += list(d[key]["parameters"].keys())
                for param in d[key]["parameters"].keys():
                    if "time_derivative" in d[key]["parameters"][param].keys():
                        flag = 1
                    if flag == 1 and "dt" not in vi:
                        vi.append("dt")

            if "states" in d[key].keys():
                vi += list(d[key]["states"].keys())
            # if 'states' in d[key].keys():
            # 	for state in d[key]['states'].keys():
            # 		if 'time_derivative' in d[key]['states'][state].keys():
            # 			flag = 1
            # 		if flag == 1 and 'dt' not in vi :
            # 			vi.append('dt')

            arg_dict[key] = vi

    get_arguments(nodes_dict)

    expression_dict = {}

    def get_expression(d: Dict[str, Any] = None):
        """get any expression (including time derivative) for each state or output port variable
        Args:
                d: Node level dictionary with filtered keys
        Returns:
                store expression for each state or output port variable
        """
        for key in d.keys():
            vi = []
            li = []
            temp_dic = {}
            if "parameters" in d[key].keys():
                for param in d[key]["parameters"].keys():

                    if "time_derivative" in d[key]["parameters"][param].keys():
                        li.append(param)
                        vi.append(d[key]["parameters"][param]["time_derivative"])
                    elif "value" in d[key]["parameters"][param].keys():

                        if isinstance(d[key]["parameters"][param]["value"], str):
                            if any(
                                x in d[key]["parameters"][param]["value"]
                                for x in expression_items
                            ):
                                li.append(param)
                                vi.append(d[key]["parameters"][param]["value"])
                            else:
                                li.append(param)
                                vi.append(None)
                        else:
                            li.append(param)
                            vi.append(None)
                    else:
                        li.append(param)
                        vi.append(None)

            if "output_ports" in d[key].keys():
                for output_port in d[key]["output_ports"].keys():
                    if isinstance(d[key]["output_ports"][output_port]["value"], str):
                        if any(
                            x in d[key]["output_ports"][output_port]["value"]
                            for x in expression_items
                        ):
                            li.append(output_port)
                            vi.append(d[key]["output_ports"][output_port]["value"])
                        else:
                            li.append(output_port)
                            vi.append(None)
            for i in range(len(vi)):
                temp_dic[li[i]] = vi[i]
            expression_dict[key] = temp_dic

    get_expression(nodes_dict)
    # print("expression_dict>>>", expression_dict)

    def createFunctions(d: Dict[str, Any] = None):
        """create functions for time_derivative expression for each state variable
        Args:
                d: Node level dictionary with filtered keys
        Returns:
                functions replacing time derivative for each state variable
        """
        for key in d.keys():

            if "functions" not in d[key].keys():
                d[key]["functions"] = {}

            if "parameters" in d[key].keys():
                parameterlist = []

                for idx, param in enumerate(list(d[key]["parameters"].keys())):
                    if "time_derivative" in d[key]["parameters"][param].keys():

                        d[key]["functions"][f"evaluated_{key}_{param}_next_value"] = {}
                        d[key]["functions"][f"evaluated_{key}_{param}_next_value"][
                            "value"
                        ] = {}

                        d[key]["functions"][f"evaluated_{key}_{param}_next_value"][
                            "value"
                        ] = (
                            str(param) + "+"
                            "(dt*" + str(expression_dict[key][param]) + ")"
                        )

                        # d[key]['functions']["evaluated_{}_{}_next_value".format(key, param)]['args']=  {}
                        # for pp in arg_dict[key]:
                        # 	d[key]['functions']["evaluated_{}_{}_next_value".format(key, param)]['args'][pp] = pp

                        parameterlist.append(param)
                    elif "value" in d[key]["parameters"][param].keys():
                        if isinstance(d[key]["parameters"][param]["value"], str):

                            if any(
                                x in d[key]["parameters"][param]["value"]
                                for x in expression_items
                            ):
                                d[key]["functions"][
                                    f"evaluated_{key}_{param}_next_value"
                                ] = {}
                                d[key]["functions"][
                                    f"evaluated_{key}_{param}_next_value"
                                ]["value"] = {}

                                d[key]["functions"][
                                    f"evaluated_{key}_{param}_next_value"
                                ]["value"] = expression_dict[key][param]
                                # d[key]['functions']["evaluated_{}_{}_next_value".format(key, param)]['args']=  {}
                                # for pp in arg_dict[key]:
                                # d[key]['functions']["evaluated_{}_{}_next_value".format(key, param)]['args'][pp] = pp
                                parameterlist.append(param)

                    elif "function" in d[key]["parameters"][param].keys():

                        d[key]["functions"][param] = {}
                        d[key]["functions"][param]["function"] = {}

                        d[key]["functions"][param]["function"][
                            d[key]["parameters"][param]["function"]
                        ] = {}
                        d[key]["functions"][param]["function"][
                            d[key]["parameters"][param]["function"]
                        ] = d[key]["parameters"][param]["args"]

                    if idx > 0:
                        for prev_param in parameterlist[:-1]:

                            # d[key]['functions']["evaluated_{}_{}_next_value".format(key, param)]['args'][prev_param] = "evaluated_{}_{}_next_value".format(key, prev_param)
                            d[key]["functions"][f"evaluated_{key}_{param}_next_value"][
                                "value"
                            ] = d[key]["functions"][
                                f"evaluated_{key}_{param}_next_value"
                            ][
                                "value"
                            ].replace(
                                prev_param, f"evaluated_{key}_{prev_param}_next_value"
                            )

            if "output_ports" in d[key].keys():

                for idx, output_port in enumerate(list(d[key]["output_ports"].keys())):
                    if isinstance(d[key]["output_ports"][output_port]["value"], str):
                        if any(
                            x in d[key]["output_ports"][output_port]["value"]
                            for x in expression_items
                        ):
                            d[key]["functions"][
                                f"evaluated_{key}_{output_port}_value"
                            ] = {}
                            d[key]["functions"][f"evaluated_{key}_{output_port}_value"][
                                "value"
                            ] = {}
                            d[key]["functions"][f"evaluated_{key}_{output_port}_value"][
                                "value"
                            ] = expression_dict[key][output_port]
                            # d[key]['functions']["evaluated_{}_{}_value".format(key, output_port)]['args']=  {}
                            # for param in arg_dict[key]:
                            # 	d[key]['functions']["evaluated_{}_{}_value".format(key, output_port)]['args'][param] = param

    createFunctions(nodes_dict)

    def changetoValue(d: Dict[str, Any] = None):
        """Converts states into stateful_parameters, adds dt to parameters
        Args:
                d: dictionary with states information at the Node
        Returns:
                dictionary with stateful_parameters information
        """
        for key in d.keys():
            if "parameters" in d[key].keys():
                for param in list(d[key]["parameters"].keys()):
                    if "time_derivative" in d[key]["parameters"][param].keys():
                        d[key]["parameters"]["dt"] = {}
                        d[key]["parameters"]["dt"]["value"] = dt
                        if (
                            "default_initial_value"
                            in d[key]["parameters"][param].keys()
                        ):
                            if (
                                d[key]["parameters"][param]["default_initial_value"]
                                in d[key]["parameters"].keys()
                            ):
                                d[key]["parameters"][param][
                                    "default_initial_value"
                                ] = d[key]["parameters"][
                                    d[key]["parameters"][param]["default_initial_value"]
                                ][
                                    "value"
                                ]

                        else:
                            d[key]["parameters"][param]["default_initial_value"] = 0

                        d[key]["parameters"][param].pop("time_derivative")
                        d[key]["parameters"][param][
                            "value"
                        ] = f"evaluated_{key}_{param}_next_value"
                        d[key]["parameters"]["time"] = {
                            "default_initial_value": 0,
                            "value": "evaluated_time_next_value",
                        }

                        d[key]["functions"]["evaluated_time_next_value"] = {}
                        d[key]["functions"]["evaluated_time_next_value"][
                            "function"
                        ] = {}

                        d[key]["functions"]["evaluated_time_next_value"]["function"][
                            "linear"
                        ] = {}

                        d[key]["functions"]["evaluated_time_next_value"]["function"][
                            "linear"
                        ] = {"variable0": "time", "slope": 1, "intercept": "dt"}

                    elif "value" in d[key]["parameters"][param].keys():
                        if isinstance(d[key]["parameters"][param]["value"], str):
                            if any(
                                x in d[key]["parameters"][param]["value"]
                                for x in expression_items
                            ):
                                if (
                                    "default_initial_value"
                                    in d[key]["parameters"][param].keys()
                                ):
                                    if (
                                        d[key]["parameters"][param][
                                            "default_initial_value"
                                        ]
                                        in d[key]["parameters"].keys()
                                    ):
                                        d[key]["parameters"][param][
                                            "default_initial_value"
                                        ] = d[key]["parameters"][
                                            d[key]["parameters"][param][
                                                "default_initial_value"
                                            ]
                                        ][
                                            "value"
                                        ]

                                else:
                                    d[key]["parameters"][param][
                                        "default_initial_value"
                                    ] = 0

                                d[key]["parameters"][param][
                                    "value"
                                ] = f"evaluated_{key}_{param}_next_value"

                    elif "function" in d[key]["parameters"][param].keys():
                        d[key]["parameters"].pop(param)

            if "output_ports" in d[key].keys():
                for output_port in list(d[key]["output_ports"].keys()):
                    if isinstance(d[key]["output_ports"][output_port]["value"], str):
                        if any(
                            x in d[key]["output_ports"][output_port]["value"]
                            for x in expression_items
                        ):

                            d[key]["output_ports"][output_port][
                                "value"
                            ] = f"evaluated_{key}_{output_port}_value"

        for key in d.keys():
            if "states" in d[key].keys():
                d[key]["stateful_parameters"] = d[key].pop("states")

    changetoValue(nodes_dict)
    glom.assign(data, path, nodes_dict)

    def repl(dr):
        """Replaces all names containing states into stateful_parameters
        Args:
                dr: full dictionary with states name
        Returns:
                dictionary with stateful_parameters name
        """
        dr = str(dr)
        dr = dr.replace("states", "stateful_parameters")
        dr = dr.replace("States", "Stateful_Parameters")
        dr = dr.replace("state_example", "stateful_parameters_example")
        return eval(dr)

    data = repl(data)
    return data
