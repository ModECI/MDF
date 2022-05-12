import json
import ntpath

from modeci_mdf.functions.standard import mdf_functions, create_python_expression
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from modeci_mdf.utils import load_mdf, print_summary
from modeci_mdf.mdf import *
from modeci_mdf.full_translator import *
from modeci_mdf.execution_engine import EvaluableGraph

import argparse
import sys


def main():

    dt = 0.01
    file_path = "States.json"
    data = convert_states_to_stateful_parameters("../" + file_path, dt)

    with open("Translated_" + file_path, "w") as fp:
        json.dump(data, fp, indent=4)

    if "-run" in sys.argv:

        verbose = False

        mod_graph = load_mdf("Translated_%s" % file_path).graphs[0]
        eg = EvaluableGraph(mod_graph, verbose)

        mod_graph_old = load_mdf("../" + file_path).graphs[0]
        eg_old = EvaluableGraph(mod_graph_old, verbose)

        duration = 2
        t = 0
        recorded = {}
        times = []
        s = []
        s_old = []

        while t <= duration:

            if t == 0:
                eg_old.evaluate()  # replace with initialize?
                eg.evaluate()  # replace with initialize?

            # levels.append(eg.enodes['sine_node'].evaluable_stateful_parameters['level'].curr_value)
            # t+=args.dt

            # print("time first>>>",type(t))
            t = float(eg.enodes["sine_node"].evaluable_parameters["time"].curr_value)
            print("======   Evaluating at t = %s  ======" % (t))

            # times.append(float(eg.enodes['sine_node'].evaluable_parameters['time'].curr_value))

            times.append(t)

            eg_old.evaluate(time_increment=dt)

            s_old.append(
                eg_old.enodes["sine_node"].evaluable_outputs["out_port"].curr_value
            )

            eg.evaluate()
            s.append(eg.enodes["sine_node"].evaluable_outputs["out_port"].curr_value)
            # t+=dt

        print(s_old[:10], s[:10], times[:10])

        import matplotlib.pyplot as plt

        plt.plot(times, s, label="translated", linewidth=2.5)
        plt.plot(times, s_old, label="original", linewidth=1)
        plt.legend()

        plt.show()
        plt.savefig("translated_levelratesineplot.jpg")


if __name__ == "__main__":
    main()
