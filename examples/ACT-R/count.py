"""Create the MDF files for the count example and run using the scheduler."""
import os
from modeci_mdf.interfaces.actr import actr_to_mdf
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import load_mdf


def main():
    """Takes count.lisp, converts to MDF, and runs using the scheduler."""
    file_name = os.path.dirname(os.path.realpath(__file__)) + "/count.lisp"
    print(file_name)
    mod = actr_to_mdf(file_name)
    mdf_graph = load_mdf(file_name[:-5] + ".json").graphs[0]
    eg = EvaluableGraph(graph=mdf_graph, verbose=False)
    term = False
    goal = {}
    retrieval = {}
    while not term:
        eg.evaluate(initializer={"goal_input": goal, "dm_input": retrieval})
        term = eg.enodes["check_termination"].evaluable_outputs["check_output"].curr_value
        goal = eg.enodes["fire_production"].evaluable_outputs["fire_prod_output_to_goal"].curr_value
        retrieval = eg.enodes["fire_production"].evaluable_outputs["fire_prod_output_to_retrieval"].curr_value
    print("Final Goal:")
    print(eg.enodes["goal_buffer"].evaluable_outputs["goal_output"].curr_value)


if __name__ == "__main__":
    main()
