from modeci_mdf.mdf import Model, Graph, Node, Parameter, OutputPort
from modeci_mdf.execution_engine import EvaluableGraph
import matplotlib.pyplot as plt
import sys, os


def main():
    # Creating the model and graph
    mod = Model(id="NewtonCoolingModel")
    mod_graph = Graph(id="cooling_process")
    mod.graphs.append(mod_graph)

    # Defining Nodes and Parameters
    cool_node = Node(id="cool_node")
    cool_node.parameters.append(Parameter(id="cooling_coeff", value=0.1))
    cool_node.parameters.append(Parameter(id="T_a", value=20))
    s1 = Parameter(id="T_curr", default_initial_value=90, time_derivative="dT_dt")
    cool_node.parameters.append(s1)
    s2 = Parameter(
        id="dT_dt", default_initial_value=0, value="-cooling_coeff*(T_curr - T_a)"
    )
    cool_node.parameters.append(s2)

    # Output Ports
    op1 = OutputPort(id="out_port", value="T_curr")
    cool_node.output_ports.append(op1)
    op2 = OutputPort(id="out_port2", value="dT_dt")
    cool_node.output_ports.append(op2)

    mod_graph.nodes.append(cool_node)
    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        # Running the Model
        verbose = False
        eg = EvaluableGraph(mod_graph, verbose=verbose)
        dt = 0.1
        duration = 100
        t = 0
        times = []
        s = []

        while t <= duration:
            times.append(t)
            if verbose:
                print(f"======   Evaluating at t = {t:.1f}  ======")
            if t == 0:
                eg.evaluate()
            else:
                eg.evaluate(time_increment=dt)
            s.append(eg.enodes["cool_node"].evaluable_outputs["out_port"].curr_value)
            t += dt

        # Plotting the results
        plt.plot(times, s)
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.title("Newton's Cooling Law Simulation")
        plt.savefig("newton_plot.png")
        plt.show()

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="newton",
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )


if __name__ == "__main__":
    main()
