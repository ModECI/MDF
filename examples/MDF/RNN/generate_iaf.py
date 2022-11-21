"""
    Example of ModECI MDF - Testing integrate and fire neurons
"""

from modeci_mdf.mdf import *
import sys
import numpy
import random


def main():
    mod = Model(id="IAFs")

    net = "-net" in sys.argv
    if net:
        mod.id = "IAF_net"

    mod_graph = Graph(id="iaf_example")
    mod.graphs.append(mod_graph)

    ## Counter node
    input_node = Node(id="input_node")

    t_param = Parameter(id="time", default_initial_value=0, time_derivative="1")
    input_node.parameters.append(t_param)

    start = Parameter(id="start", value=20)
    input_node.parameters.append(start)

    dur = Parameter(id="duration", value=60)
    input_node.parameters.append(dur)

    amp = Parameter(id="amplitude", value=10)
    input_node.parameters.append(amp)

    level = Parameter(id="level", value=0)

    level.conditions.append(
        ParameterCondition(
            id="on", test="time > start and time < start + duration", value=amp.id
        )
    )
    level.conditions.append(
        ParameterCondition(
            id="off", test="time > start + duration", value="amplitude*0"
        )
    )

    input_node.parameters.append(level)

    op1 = OutputPort(id="out_port", value=level.id)
    input_node.output_ports.append(op1)

    # op2 = OutputPort(id="t_out_port", value=t_param.id)
    # input_node.output_ports.append(op2)

    mod_graph.nodes.append(input_node)

    ## IAF node...
    iaf_node = Node(id="iaf_node")
    ip1 = InputPort(id="input")
    iaf_node.input_ports.append(ip1)

    v0 = Parameter(id="v0", value=-60)

    iaf_node.parameters.append(v0)

    erev = Parameter(id="erev", value=-70)
    iaf_node.parameters.append(erev)
    tau = Parameter(id="tau", value=10.0)
    iaf_node.parameters.append(tau)
    thresh = Parameter(id="thresh", value=-20.0)
    iaf_node.parameters.append(thresh)

    # v_init = Parameter(id="v_init", value=-30)
    # iaf_node.parameters.append(v_init)

    pc1 = ParameterCondition(id="is_spiking", test="v >= thresh", value="1")
    pc2 = ParameterCondition(id="not_spiking", test="v < thresh", value="0")

    spiking = Parameter(
        id="spiking",
        default_initial_value="0",
    )
    spiking.conditions.append(pc1)
    spiking.conditions.append(pc2)
    iaf_node.parameters.append(spiking)

    pc = ParameterCondition(id="reset", test="v > thresh", value="erev")

    v = Parameter(
        id="v",
        default_initial_value="v0",
        time_derivative="-1 * (v-erev)/tau + input",
    )
    v.conditions.append(pc)

    iaf_node.parameters.append(v)

    op_v = OutputPort(id="out_port_v", value="v")
    iaf_node.output_ports.append(op_v)

    mod_graph.nodes.append(iaf_node)

    e1 = Edge(
        id="input_edge",
        parameters={"weight": 1},
        sender=input_node.id,
        sender_port=op1.id,
        receiver=iaf_node.id,
        receiver_port=ip1.id,
    )

    if net:
        num = 8
        random.seed(123)
        v0.value = numpy.array([random.random() * 20 - 70 for r in range(num)])
        erev.value = numpy.array([-70.0] * len(v0.value))
        thresh.value = numpy.array([-20.0] * len(v0.value))
        amp.value = numpy.array([random.random() * 20 for r in range(num)])
        # e1.parameters['weight'] = [1,.5]

    mod_graph.edges.append(e1)

    new_file = mod.to_json_file("%s.json" % mod.id)
    new_file = mod.to_yaml_file("%s.yaml" % mod.id)

    if "-run" in sys.argv:

        verbose = "-v" in sys.argv

        from modeci_mdf.utils import load_mdf, print_summary

        from modeci_mdf.execution_engine import EvaluableGraph

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.1

        duration = 100

        t_ext = 0.0
        recorded = {}
        times = []
        t = []
        i = []
        s = []
        sp = []
        while t_ext <= duration:
            times.append(t_ext)
            print("======   Evaluating at t = %s  ======" % (t_ext))
            if t_ext == 0:
                eg.evaluate()  # replace with initialize?
            else:
                eg.evaluate(time_increment=dt)

            i.append(eg.enodes["input_node"].evaluable_outputs["out_port"].curr_value)
            t.append(eg.enodes["input_node"].evaluable_parameters["time"].curr_value)
            s.append(eg.enodes["iaf_node"].evaluable_outputs["out_port_v"].curr_value)
            sp.append(eg.enodes["iaf_node"].evaluable_parameters["spiking"].curr_value)
            t_ext += dt

        import matplotlib.pyplot as plt

        plt.plot(times, t, label="time at input node")
        if type(i[0]) == numpy.ndarray and i[0].size > 1:
            for ii in range(len(i[0])):
                iii = []
                for ti in range(len(t)):
                    iii.append(i[ti][ii])
                plt.plot(times, iii, label="Input node %s state" % ii)
        else:
            plt.plot(times, i, label="state of input node")

        if type(s[0]) == numpy.ndarray and s[0].size > 1:
            for si in range(len(s[0])):
                ss = []
                for ti in range(len(t)):
                    ss.append(s[ti][si])
                plt.plot(times, ss, label="IaF %s state" % si)
        else:
            plt.plot(times, s, label="IaF 0 state")

        if type(sp[0]) == numpy.ndarray and sp[0].size > 1:
            for spi in range(len(sp[0])):
                sps = []
                for ti in range(len(t)):
                    sps.append(sp[ti][spi])
                plt.plot(times, sps, label="IaF %s spiking" % spi)
        else:
            plt.plot(times, sp, label="spiking")

        plt.legend()

        plt.savefig("IaF%s.run.png" % (".net" if net else ""), bbox_inches="tight")

        if "-nogui" not in sys.argv:
            plt.show()

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=3,
            filename_root="iaf",
            only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
        )

    if "-neuroml" in sys.argv:
        from modeci_mdf.interfaces.neuroml.exporter import mdf_to_neuroml

        net, sim = mdf_to_neuroml(
            mod_graph, save_to="%s.nmllite.json" % mod.id, run_duration_sec=100
        )

        from neuromllite.NetworkGenerator import generate_and_run

        generate_and_run(sim, simulator="jNeuroML")

    return mod_graph


if __name__ == "__main__":
    main()
