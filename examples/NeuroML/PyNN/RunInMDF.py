from modeci_mdf.utils import load_mdf, print_summary

import sys
import matplotlib.pyplot as plt


def execute(mdf_filename):
    mdf_model = load_mdf(mdf_filename)
    mod_graph = mdf_model.graphs[0]

    dt = 0.001
    duration = 1

    mdf_model.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=1,
        filename_root=mdf_filename.replace(".yaml", ".1").replace(".json", ".1"),
        only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
    )
    mdf_model.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=2,
        filename_root=mdf_filename.replace(".yaml", "").replace(".json", ""),
        only_warn_on_fail=True,  # Makes sure test of this doesn't fail on Windows on GitHub Actions
    )

    verbose = "-v" in sys.argv

    from modeci_mdf.execution_engine import EvaluableGraph

    eg = EvaluableGraph(mod_graph, verbose)
    # duration= 2

    t = 0
    times = []
    vv = {}
    uu = {}
    ii = {}

    outputs = {}
    for n in mod_graph.nodes:
        outputs[n.id] = {}
        for op in n.output_ports:
            outputs[n.id][op.id] = {}

    while t <= duration:
        times.append(t)
        print("======   Evaluating at t = %s  ======" % (t))
        if t == 0:
            eg.evaluate(array_format=format)  # replace with initialize?
        else:
            eg.evaluate(array_format=format, time_increment=dt)

        for n in mod_graph.nodes:
            for op in n.output_ports:
                ov = eg.enodes[n.id].evaluable_outputs[op.id].curr_value
                if type(ov) == int or type(ov) == float:
                    ov = [ov]
                if ov is None:
                    raise Exception(
                        "Error getting value of output port: %s on node: %s"
                        % (op, n.id)
                    )
                for i in range(len(ov)):
                    if not i in outputs[n.id][op.id]:
                        outputs[n.id][op.id][i] = []

                    outputs[n.id][op.id][i].append(ov[i])

        t += dt

    # print(times)
    # print(outputs)

    for n in mod_graph.nodes:
        for op in n.output_ports:
            vals = outputs[n.id][op.id]
            plt.figure()
            for i in vals:
                label = "%s_%s_%i" % (n.id, op.id, i)
                print(
                    " - Plotting %s, points in times: %i, val: %s"
                    % (label, len(times), len(vals[i]))
                )
                try:
                    plt.plot(times, vals[i], label=label)
                    plt.legend()
                except Exception as e:
                    print(e)

    # plt.savefig("Izh_run.png", bbox_inches="tight")

    if not "-nogui" in sys.argv:
        plt.show()


if __name__ == "__main__":

    if len(sys.argv) == 1 or sys.argv[1] == "-run":
        print("Please specify the MDF file to run")
    else:
        f = sys.argv[1]
        execute(f)
