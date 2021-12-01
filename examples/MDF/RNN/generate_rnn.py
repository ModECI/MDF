"""
    Example of ModECI MDF - Testing RNNs
"""

from modeci_mdf.mdf import *
import sys


def main():
    mod = Model(id="RNNs")
    mod_graph = Graph(id="rnn_example")
    mod.graphs.append(mod_graph)

    ## Counter node
    input_node = Node(id="input_node")

    t_param = Parameter(id="t", default_initial_value=0, time_derivative="1")
    input_node.parameters.append(t_param)

    p0 = Parameter(id="initial", value=-3)
    input_node.parameters.append(p0)
    p1 = Parameter(id="rate", value=3)
    input_node.parameters.append(p1)

    p2 = Parameter(id="level", value="initial + rate*t")
    input_node.parameters.append(p2)

    op1 = OutputPort(id="out_port", value=p2.id)
    input_node.output_ports.append(op1)
    op2 = OutputPort(id="t_out_port", value=t_param.id)
    input_node.output_ports.append(op2)

    mod_graph.nodes.append(input_node)

    ## RNN node...
    rnn_node = Node(id="rnn_node")
    ip1 = InputPort(id="input")
    rnn_node.input_ports.append(ip1)

    s1 = Parameter(id="r", function="tanh", args={"variable0": ip1.id, "scale": 1})
    rnn_node.parameters.append(s1)

    op1 = OutputPort(id="out_port", value="r")
    rnn_node.output_ports.append(op1)
    mod_graph.nodes.append(rnn_node)

    e1 = Edge(
        id="input_edge",
        parameters={"weight": 1},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=rnn_node.id,
        receiver_port=ip1.id,
    )

    mod_graph.edges.append(e1)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        # verbose = False
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.01

        duration = 2
        t_ext = 0
        recorded = {}
        times = []
        t = []
        i = []
        s = []
        while t_ext <= duration:
            times.append(t_ext)
            print("======   Evaluating at t = %s  ======" % (t_ext))
            if t == 0:
                eg.evaluate()  # replace with initialize?
            else:
                eg.evaluate(time_increment=dt)

            i.append(eg.enodes["input_node"].evaluable_outputs["out_port"].curr_value)
            t.append(eg.enodes["input_node"].evaluable_outputs["t_out_port"].curr_value)
            s.append(eg.enodes["rnn_node"].evaluable_outputs["out_port"].curr_value)
            t_ext += dt

        if "-nogui" not in sys.argv:
            import matplotlib.pyplot as plt

            plt.plot(times, t, label="time at input node")
            plt.plot(times, i, label="state of input node")
            plt.plot(times, s, label="RNN 0 state")
            plt.legend()
            plt.show()

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="rnn",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

    return mod_graph


if __name__ == "__main__":
    main()
