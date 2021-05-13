"""Create the MDF files for the given example and run using the scheduler."""
import sys
from modeci_mdf.interfaces.actr import actr_to_mdf
from modeci_mdf.scheduler import EvaluableGraph
from modeci_mdf.utils import load_mdf


def main(file_name):
    actr_to_mdf(file_name)
    mdf_graph = load_mdf(file_name[:-5] + ".json").graphs[0]
    eg = EvaluableGraph(graph=mdf_graph, verbose=True)
    eg.evaluate(initializer={"goal_input": {}, "dm_input": {}})


if __name__ == "__main__":
    main(sys.argv[1])