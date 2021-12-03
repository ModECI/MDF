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

    p0 = Parameter(id="amplitude", value=[1, 2])
    input_node.parameters.append(p0)
    p1 = Parameter(id="period", value=[0.5, 0.4])
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

    ## RNN node...
    rnn_node = Node(id="rnn_node")
    ipr1 = InputPort(id="ext_input")
    rnn_node.input_ports.append(ipr1)

    x = Parameter(
        id="x", default_initial_value=[1.0, 1.0], time_derivative="-x + %s" % ipr1.id
    )
    rnn_node.parameters.append(x)

    r = Parameter(id="r", function="tanh", args={"variable0": x.id, "scale": 1})
    # r = Parameter(id="r", value="x")
    rnn_node.parameters.append(r)

    op_x = OutputPort(id="out_port_x", value="x")
    rnn_node.output_ports.append(op_x)

    op_r = OutputPort(id="out_port_r", value="r")
    rnn_node.output_ports.append(op_r)

    mod_graph.nodes.append(rnn_node)

    e1 = Edge(
        id="input_edge",
        parameters={"weight": 1},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=rnn_node.id,
        receiver_port=ipr1.id,
    )

    mod_graph.edges.append(e1)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:
        verbose = True
        #
        verbose = False
        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.01

        duration = 2
        t_ext = 0
        recorded = {}
        times = []
        ts = []
        ins = []
        xs = []
        rs = []
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

            print(f"  - Values at {t}: i={i}; x={xx}; r={r}")

            ins.append(i[0])
            ts.append(t)
            xs.append(xx)
            rs.append(r)

            t_ext += dt

        if "-nogui" not in sys.argv:
            import matplotlib.pyplot as plt

            print("i: %s" % ins)
            print("x: %s" % xs)
            print("r: %s" % rs)

            # plt.plot(times, ts, label="time at input node")
            plt.plot(times, ins, label="state of input node")
            plt.plot(times, xs, label="RNN x state")
            plt.plot(times, rs, label="RNN r state")
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
