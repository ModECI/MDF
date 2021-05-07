import psyneulink as pnl

comp = pnl.Composition(name='ABCD')

A = pnl.TransferMechanism(function=pnl.Linear(slope=2.0, intercept=2.0), name='A')
B = pnl.TransferMechanism(function=pnl.Logistic, name='B')
C = pnl.TransferMechanism(function=pnl.Exponential, name='C')
D = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(rate=.05), name='D')

for m in [A, B, C, D]:
    comp.add_node(m)

comp.add_linear_processing_pathway([A, B, D])
comp.add_linear_processing_pathway([A, C, D])

comp.run(inputs={A: 0}, log=True, num_trials=50)

print('Finished running model')

print(comp.results)
for node in comp.nodes:
    print(f'{node} {node.name}: {node.parameters.value.get(comp)}')

base_fname = __file__.replace('.py', '')
with open(f'{base_fname}.json', 'w') as outfi:
    outfi.write(comp.json_summary)

with open(f'{base_fname}.converted.py', 'w') as outfi:
    outfi.write(pnl.generate_script_from_json(comp.json_summary))
    outfi.write('\ncomp.show_graph()')

comp.show_graph()

try:
    import matplotlib.pyplot as plt

    def generate_time_array(node, context='ABCD', param='value'):
        return [entry.time.trial for entry in getattr(node.parameters, param).log[context]]

    def generate_value_array(node, context='ABCD', param='value'):
        return [float(entry.value) for entry in getattr(node.parameters, param).log[context]]

    x_values = {node: generate_time_array(node) for node in comp.nodes}
    y_values = {node: generate_value_array(node) for node in comp.nodes}

    fig, axes = plt.subplots()

    for node in comp.nodes:
        axes.plot(
            x_values[node],
            y_values[node],
            label=f'Value of {node.name}, {node.function.__class__.__name__}'
        )

    axes.set_xlabel('Trial')
    axes.legend()
    plt.show()
except ImportError:
    pass
