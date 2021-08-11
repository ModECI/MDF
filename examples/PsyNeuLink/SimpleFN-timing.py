import psyneulink as pnl
import matplotlib.pyplot as plt

dt = 0.05
simtime = 100

time_step_size = dt

fhn = pnl.FitzHughNagumoIntegrator(
    initial_v=-1,
    initial_w=0,
    d_v=1,
    time_step_size=time_step_size,
)

print(f"Running simple model of FitzHugh Nagumo cell for {simtime}ms: {fhn}")

fn = pnl.IntegratorMechanism(name="fn", function=fhn)

comp = pnl.Composition(name="comp")

im = pnl.IntegratorMechanism(name="im")  # only used to demonstrate conditions
comp.add_linear_processing_pathway([fn, im])
comp.scheduler.add_condition_set(
    {
        fn: pnl.TimeInterval(repeat=0.05, unit="ms"),
        im: pnl.TimeInterval(start=80, repeat=1, unit="ms"),
    }
)

comp.termination_processing = {
    pnl.TimeScale.RUN: pnl.Never(),  # default, "Never" for early termination - ends when all trials finished
    pnl.TimeScale.TRIAL: pnl.TimeTermination(100, unit="ms"),
}

print("Running the SimpleFN model...")

comp.run(inputs={fn: 0}, log=True, scheduling_mode=pnl.SchedulingMode.EXACT_TIME)


print(
    "\n".join(
        [
            "{:~}: {}".format(
                comp.scheduler.execution_timestamps[comp.default_execution_id][
                    i
                ].absolute,
                {node.name for node in time_step},
            )
            for i, time_step in enumerate(
                comp.scheduler.execution_list[comp.default_execution_id]
            )
            if len(time_step) > 0
        ]
    )
)


print("Finished running the SimpleFN model")

for node in comp.nodes:
    print(f"=== {node} {node.name}: {node.parameters.value.get(comp)}")


def generate_time_array(node, context=comp.default_execution_id, param="value"):
    return [entry.time.pass_ for entry in getattr(node.parameters, param).log[context]]


def generate_value_array(node, index, context=comp.default_execution_id, param="value"):
    return [
        float(entry.value[index])
        for entry in getattr(node.parameters, param).log[context]
    ]


"""
for node in comp.nodes:
    print(f'>> {node}: {generate_time_array(node)}')

    for i in [0,1,2]:
        print(f'>> {node}: {generate_value_array(node,i)}')"""


fig, axes = plt.subplots()
for i in [0, 1]:
    plot_nodes = [node for node in comp.nodes if len(node.defaults.value) > i]

    x_values = {node: generate_time_array(node) for node in plot_nodes}
    y_values = {node: generate_value_array(node, i) for node in plot_nodes}

    fout = open("SimpleFN-timing_%i.dat" % i, "w")
    for index in range(len(x_values[fn])):
        # 1000 to convert ms to s
        fout.write(
            f"{0}\t{1}\n".format(
                x_values[fn][index] * time_step_size / 1000.0, y_values[fn][index]
            )
        )
    fout.close()

    for node in plot_nodes:
        axes.plot(
            [t * time_step_size / 1000.0 for t in x_values[node]],
            y_values[node],
            label=f"Value of {i} {node.name}, {node.function.__class__.__name__}",
        )

axes.set_xlabel("Time (s)")
axes.set_title("FitzHughNagumoIntegrator")
axes.legend()
plt.savefig("SimpleFN-timing.png")
# plt.show()
