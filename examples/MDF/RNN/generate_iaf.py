"""
    Example of ModECI MDF - Testing integrate and fire neurons
"""

from modeci_mdf.mdf import *
import sys
import os
import numpy
import random

random.seed(1234)


def create_current_pulse_node(id, start=20, duration=60, amplitude=10):

    ## Current input node
    current_pulse_node = Node(id=id)

    t_param = Parameter(id="time", default_initial_value=0, time_derivative="1")
    current_pulse_node.parameters.append(t_param)

    start = Parameter(id="start", value=start)
    current_pulse_node.parameters.append(start)

    dur = Parameter(id="duration", value=duration)
    current_pulse_node.parameters.append(dur)

    amp = Parameter(id="amplitude", value=amplitude)
    current_pulse_node.parameters.append(amp)

    level = Parameter(id="level", value=0)

    level.conditions.append(
        ParameterCondition(id="on", test="time > start", value=amp.id)
    )
    level.conditions.append(
        ParameterCondition(
            id="off", test="time > start + duration", value="amplitude*0"
        )
    )

    current_pulse_node.parameters.append(level)

    op1 = OutputPort(id="current_output", value=level.id)
    current_pulse_node.output_ports.append(op1)

    return current_pulse_node


def create_iaf_syn_node(
    id, num_cells=1, syn_also=True, v0=-60, erev=-70, thresh=-20, tau=10.0, syn_tau=10
):

    if syn_also:
        ## Syn node...
        syn_node = Node("syn_%s" % id)

        syn_tau = Parameter(id="syn_tau", value=syn_tau)
        syn_node.parameters.append(syn_tau)

        ip_spike = InputPort(id="spike_input", shape="(%i,)" % num_cells)
        syn_node.input_ports.append(ip_spike)

        spike_weights = Parameter(id="spike_weights", value=numpy.identity(num_cells))
        syn_node.parameters.append(spike_weights)

        weighted_spike = Parameter(
            id="weighted_spike",
            function="MatMul",
            args={"A": spike_weights.id, "B": ip_spike.id},
        )
        syn_node.parameters.append(weighted_spike)

        pc = ParameterCondition(
            id="spike_detected",
            test="%s > 0" % ip_spike.id,
            value="%s" % (weighted_spike.id),
        )

        syn_i = Parameter(
            id="syn_i",
            default_initial_value="0",
            time_derivative="-1 * syn_i",
        )
        syn_i.conditions.append(pc)
        syn_node.parameters.append(syn_i)

        op_v = OutputPort(id="current_output", value="syn_i")
        syn_node.output_ports.append(op_v)

    ## IAF node...
    iaf_node = Node(id)

    ip_current = InputPort(id="current_input", shape="(%i,)" % num_cells)
    iaf_node.input_ports.append(ip_current)

    v0 = Parameter(
        id="v0",
        value=numpy.array([v0] * num_cells) if isinstance(v0, (int, float)) else v0,
    )
    iaf_node.parameters.append(v0)

    erev = Parameter(id="erev", value=numpy.array([erev] * num_cells))
    iaf_node.parameters.append(erev)

    tau = Parameter(id="tau", value=tau)
    iaf_node.parameters.append(tau)

    thresh = Parameter(id="thresh", value=numpy.array([thresh] * num_cells))
    iaf_node.parameters.append(thresh)

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
        time_derivative=f"-1 * (v-erev)/tau + {ip_current.id}",
    )
    v.conditions.append(pc)

    iaf_node.parameters.append(v)

    op_v = OutputPort(id="v_output", value="v")
    iaf_node.output_ports.append(op_v)

    op_spiking = OutputPort(id="spiking_output", value="spiking")
    iaf_node.output_ports.append(op_spiking)

    if syn_also:

        internal_edge = Edge(
            id="%s_internal_edge" % id,
            sender=syn_node.id,
            sender_port=syn_node.get_output_port("current_output").id,
            receiver=iaf_node.id,
            receiver_port=iaf_node.get_input_port("current_input").id,
        )

        return iaf_node, syn_node, internal_edge
    else:
        return iaf_node


def main():
    mod = Model(id="IAFs")

    net = "-net" in sys.argv
    net2 = "-net2" in sys.argv
    net3 = "-net3" in sys.argv

    if net:
        mod.id = "IAF_net"
    if net2:
        mod.id = "IAF_net2"
    if net3:
        mod.id = "IAF_net3"

    some_net = net or net2 or net3

    num_cells = 8 if net or net2 else (1 if net3 else 1)

    mod_graph = Graph(id="iaf_example")
    mod.graphs.append(mod_graph)

    start = 20
    duration = 60
    amplitude = 10

    if num_cells > 1:
        amplitude = numpy.array([random.random() * 20 for r in range(num_cells)])

    if net2:
        amplitude = 15
        start = numpy.arange(10, 10 * (num_cells + 1), 10)
        duration = numpy.ones(num_cells) * 5
        # t_param.default_initial_value = numpy.zeros(num_cells)
        # t_param.time_derivative = str([0]*num_cells)

    if net3:
        amplitude = 10
        start = 20
        duration = 10

    input_node = create_current_pulse_node(
        "current_input_node", start, duration, amplitude
    )

    mod_graph.nodes.append(input_node)

    if net3:
        input_node2 = create_current_pulse_node("current_input_node2", 60, 10, 3)
        mod_graph.nodes.append(input_node2)

    v0 = (
        numpy.array([random.random() * 20 - 70 for r in range(num_cells)])
        if net
        else -70
        if net2
        else -60
    )

    iaf_node1 = create_iaf_syn_node("pre", num_cells, syn_also=False, v0=v0)

    mod_graph.nodes.append(iaf_node1)

    e1 = Edge(
        id="input_edge",
        sender=input_node.id,
        sender_port=input_node.get_output_port("current_output").id,
        receiver=iaf_node1.id,
        receiver_port=iaf_node1.get_input_port("current_input").id,
    )

    mod_graph.edges.append(e1)
    mod_graph.nodes.append(iaf_node1)

    if net3:
        e2 = Edge(
            id="input_edge2",
            sender=input_node2.id,
            sender_port=input_node2.get_output_port("current_output").id,
            receiver=iaf_node1.id,
            receiver_port=iaf_node1.get_input_port("current_input").id,
        )

        mod_graph.edges.append(e2)

    iaf_node2, syn_node2, internal_edge2 = create_iaf_syn_node(
        "post", num_cells, syn_also=True, v0=v0
    )
    mod_graph.nodes.append(iaf_node2)
    mod_graph.nodes.append(syn_node2)
    mod_graph.edges.append(internal_edge2)

    if net2:
        iaf_node2.get_parameter("tau").value = 1

    weight = [40]
    if net:
        weight = numpy.ones([num_cells, num_cells]) * 40
    if net2:
        weight = numpy.identity(num_cells)
        for i in range(num_cells):
            weight[i, i] = i

    syn_node2.get_parameter("spike_weights").value = weight

    e2 = Edge(
        id="syn_edge",
        sender=iaf_node1.id,
        sender_port=iaf_node1.get_output_port("spiking_output").id,
        receiver=syn_node2.id,
        receiver_port=syn_node2.get_input_port("spike_input").id,
    )

    mod_graph.edges.append(e2)

    j_file = "%s.json" % mod.id
    new_file = mod.to_json_file(j_file)
    print("Saved to %s" % j_file)
    y_file = "%s.yaml" % mod.id
    new_file = mod.to_yaml_file(y_file)
    print("Saved to %s" % y_file)

    if "-run" in sys.argv:

        verbose = "-v" in sys.argv

        from modeci_mdf.utils import load_mdf, print_summary
        from modeci_mdf.execution_engine import EvaluableGraph

        from modelspec.utils import FORMAT_NUMPY, FORMAT_TENSORFLOW

        format = FORMAT_TENSORFLOW if "-tf" in sys.argv else FORMAT_NUMPY

        eg = EvaluableGraph(mod_graph, verbose)
        dt = 0.1

        duration = 100

        t_ext = 0.0
        recorded = {}
        times = []
        t = []
        i = []
        s1 = []
        sp1 = []
        s2 = []
        sp2 = []

        while t_ext <= duration:
            times.append(t_ext)
            print("======   Evaluating at t = %s  ======" % (t_ext))
            if t_ext == 0:
                eg.evaluate(array_format=format)  # replace with initialize?
            else:
                eg.evaluate(time_increment=dt, array_format=format)

            if verbose:
                print(
                    "  Out v: %s"
                    % eg.enodes["post"].evaluable_outputs["v_output"].curr_value
                )

            i.append(
                eg.enodes[input_node.id].evaluable_outputs["current_output"].curr_value
            )
            t.append(eg.enodes[input_node.id].evaluable_parameters["time"].curr_value)
            s1.append(eg.enodes["pre"].evaluable_outputs["v_output"].curr_value)
            sp1.append(eg.enodes["pre"].evaluable_parameters["spiking"].curr_value)
            s2.append(eg.enodes["post"].evaluable_outputs["v_output"].curr_value)
            sp2.append(eg.enodes["post"].evaluable_parameters["spiking"].curr_value)
            t_ext += dt

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(4, 1, figsize=(7, 7))

        # axis[0].plot(times, t, label="time at input node")

        markersize = 2 if num_cells < 20 else 0.5

        if type(i[0]) == numpy.ndarray and i[0].size > 1:
            for ii in range(len(i[0])):
                iii = []
                for ti in range(len(t)):
                    iii.append(i[ti][ii])
                axis[0].plot(
                    times, iii, label="Input node %s current" % ii, linewidth="0.5"
                )
        else:
            axis[0].plot(times, i, label="Input node current", color="k")

        if not some_net:
            axis[0].legend()

        if type(s1[0]) == numpy.ndarray and s1[0].size > 1:
            for si in range(len(s1[0])):
                ss = []
                for ti in range(len(t)):
                    ss.append(s1[ti][si])
                axis[1].plot(times, ss, label="IaF pre %s v" % si, linewidth="0.5")
        else:
            axis[1].plot(times, s1, label="IaF pre v", color="r")

        if not some_net:
            axis[1].legend()

        if type(s2[0]) == numpy.ndarray and s2[0].size > 1:
            for si in range(len(s2[0])):
                ss = []
                for ti in range(len(t)):
                    ss.append(s2[ti][si])
                axis[2].plot(times, ss, label="IaF post %s v" % si, linewidth="0.5")
        else:
            axis[2].plot(times, s2, label="IaF post v", color="b")

        if not some_net:
            axis[2].legend()

        if type(sp1[0]) == numpy.ndarray and sp1[0].size > 1:
            for spi1 in range(len(sp1[0])):
                sps1 = []
                for ti in range(len(t)):
                    sps1.append(sp1[ti][spi1])

                nz = [t * dt for t in numpy.nonzero(sps1)][0]
                axis[3].plot(
                    nz,
                    numpy.ones(len(nz)) * spi1,
                    marker=".",
                    color="r",
                    linewidth=0,
                    markersize=markersize,
                )

            for spi in range(len(sp2[0])):
                sps = []
                for ti in range(len(t)):
                    sps.append(sp2[ti][spi])

                nz = [t * dt for t in numpy.nonzero(sps)][0]
                axis[3].plot(
                    nz,
                    numpy.ones(len(nz)) * spi + num_cells,
                    marker=".",
                    color="b",
                    linewidth=0,
                    markersize=markersize,
                )
        else:
            nz1 = [t * dt for t in numpy.nonzero(sp1)][0]
            axis[3].plot(
                nz1,
                numpy.zeros(len(nz1)),
                marker=".",
                linewidth=0,
                label="pre",
                color="r",
            )
            nz2 = [t * dt for t in numpy.nonzero(sp2)][0]
            axis[3].plot(
                nz2,
                numpy.ones(len(nz2)),
                marker=".",
                linewidth=0,
                label="post",
                color="b",
            )
            plt.ylim([-1, 2])

            axis[3].legend()

        plt.xlim([0, duration])
        plt.xlabel("Time")

        plt.savefig(
            "IaF%s.run.png"
            % (
                ".net"
                if (net)
                else (".net2" if (net2) else (".net3" if (net3) else ""))
            ),
            bbox_inches="tight",
        )

        if "-nogui" not in sys.argv:
            plt.show()

    if "-graph" in sys.argv:
        mod.to_graph_image(
            engine="dot",
            output_format="png",
            view_on_render=False,
            level=2,
            filename_root="iaf%s"
            % (
                ".net"
                if (net)
                else (".net2" if (net2) else (".net3" if (net3) else ""))
            ),
            is_horizontal=True,
            only_warn_on_fail=(
                os.name == "nt"
            ),  # Makes sure test of this doesn't fail on Windows on GitHub Actions
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
