import psyneulink as pnl

comp = pnl.Composition(name="ABCD")

A = pnl.TransferMechanism(function=pnl.Linear(slope=2.0, intercept=2.0), name="A")
B = pnl.TransferMechanism(function=pnl.Logistic, name="B")
C = pnl.TransferMechanism(function=pnl.Exponential, name="C")
D = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(rate=0.05), name="D")

for m in [A, B, C, D]:
    comp.add_node(m)

comp.add_linear_processing_pathway([A, B, D])
comp.add_linear_processing_pathway([A, C, D])

comp.run(inputs={A: 0}, log=True, num_trials=50)

print("Finished running model")

print(comp.results)
for node in comp.nodes:
    print(f"{node} {node.name}: {node.parameters.value.get(comp)}")

# comp.show_graph()

try:
    import matplotlib.pyplot as plt

    def generate_time_array(node, context="ABCD", param="value"):
        return [
            entry.time.trial for entry in getattr(node.parameters, param).log[context]
        ]

    def generate_value_array(node, context="ABCD", param="value"):
        return [
            float(entry.value) for entry in getattr(node.parameters, param).log[context]
        ]

    x_values = {node: generate_time_array(node) for node in comp.nodes}
    y_values = {node: generate_value_array(node) for node in comp.nodes}

    fig, axes = plt.subplots()

    for node in comp.nodes:
        axes.plot(
            x_values[node],
            y_values[node],
            label=f"Value of {node.name}, {node.function.__class__.__name__}",
        )

    axes.set_xlabel("Trial")
    axes.legend()
    plt.savefig("model_ABCD.png")
except ImportError:
    pass
