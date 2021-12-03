"""
    Example of ModECI MDF - Testing RNNs
"""

from modeci_mdf.mdf import *
import sys
import numpy as np

from utils import create_rnn_node


def main():
    mod = Model(id="RNNs")
    mod_graph = Graph(id="rnn_example")
    mod.graphs.append(mod_graph)

    ## input node
    input_node = Node(id="input_node")

    t_param = Parameter(id="t", default_initial_value=0, time_derivative="1")
    input_node.parameters.append(t_param)

    p0 = Parameter(id="amplitude", value=[1])
    input_node.parameters.append(p0)
    p1 = Parameter(id="period", value=[0.5])
    input_node.parameters.append(p1)

    p2 = Parameter(
        id="level",
        function="sin",
        args={"variable0": "2*3.14159*t/period", "scale": "amplitude"},
    )
    input_node.parameters.append(p2)

    op1 = OutputPort(id="out_port", value=p2.id)
    input_node.output_ports.append(op1)
    op2 = OutputPort(id="t_out_port", value=t_param.id)
    input_node.output_ports.append(op2)

    mod_graph.nodes.append(input_node)

    N = 5
    g = 1
    rnn_node = create_rnn_node("rnn_node", N, g)

    mod_graph.nodes.append(rnn_node)

    weight = np.zeros(N)
    weight[0] = 1

    e1 = Edge(
        id="input_edge",
        parameters={"weight": weight},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=rnn_node.id,
        receiver_port=rnn_node.input_ports[0].id,
    )

    mod_graph.edges.append(e1)

    readout_node = Node(id="readout_node")

    ipro = InputPort(id="input")
    readout_node.input_ports.append(ipro)

    wr = Parameter(id="wr", value=np.ones(N))
    readout_node.parameters.append(wr)

    zi = Parameter(id="zi", function="MatMul", args={"A": "input", "B": "wr"})
    readout_node.parameters.append(zi)

    opro = OutputPort(id="z", value=zi.id)
    readout_node.output_ports.append(opro)

    mod_graph.nodes.append(readout_node)

    e2 = Edge(
        id="readout_edge",
        parameters={"weight": 0.5},
        sender=rnn_node.id,
        sender_port=rnn_node.get_output_port("out_port_r").id,
        receiver=readout_node.id,
        receiver_port=readout_node.input_ports[0].id,
    )
    mod_graph.edges.append(e2)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:

        verbose = "-v" in sys.argv
        # verbose = False

        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.01

        duration = 5
        t_ext = 0
        recorded = {}
        times = []
        ts = []
        ins = []
        xs = []
        rs = []
        zs = []
        while t_ext <= duration:
            times.append(t_ext)
            print("======   Evaluating at t = %s  ======" % (t_ext))
            if t_ext == 0:
                eg.evaluate()  # replace with initialize?
            else:
                eg.evaluate(time_increment=dt)

            t = eg.enodes["input_node"].evaluable_outputs["t_out_port"].curr_value
            i = eg.enodes["input_node"].evaluable_outputs["out_port"].curr_value
            xx = eg.enodes["rnn_node"].evaluable_outputs["out_port_x"].curr_value
            r = eg.enodes["rnn_node"].evaluable_outputs["out_port_r"].curr_value
            z = eg.enodes["readout_node"].evaluable_outputs["z"].curr_value

            print(f"  - Values at {t}: i={i}; x={xx}; r={r}; z={z}")

            ins.append(i[0])
            ts.append(t)
            xs.append(xx)
            rs.append(r)
            zs.append(z)

            t_ext += dt

        if "-nogui" not in sys.argv:
            import matplotlib.pyplot as plt

            """
            print("i: %s" % ins)
            print("x: %s" % xs)
            print("r: %s" % rs)"""

            # plt.plot(times, ts, label="time at input node")
            plt.plot(times, ins, label="state of input node")
            plt.plot(times, xs, label="RNN x state", linewidth=0.5)
            plt.plot(times, rs, label="RNN r state")
            plt.plot(times, zs, label="z readout", linewidth=3)
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
