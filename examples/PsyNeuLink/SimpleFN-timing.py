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

print('Running simple model of FitzHugh Nagumo cell for %sms: %s' % (simtime, fhn))

fn = pnl.IntegratorMechanism(name='fn', function=fhn)

comp = pnl.Composition(name='comp')
comp.add_linear_processing_pathway([fn])

# Left commented because TimeInterval is still to be implemented in PNL
# im = pnl.IntegratorMechanism(name='im')  # only used to demonstrate conditions
# comp.add_linear_processing_pathway([fn, im])
# comp.scheduler.add_condition_set({
#     fn: pnl.TimeInterval(interval=.05, unit='ms')
#     im: pnl.TimeInterval(start=80, interval=1, unit='ms')
# })

comp.termination_processing = {
    pnl.TimeScale.RUN: pnl.Never(),  # default, "Never" for early termination - ends when all trials finished
    pnl.TimeScale.TRIAL: pnl.AfterNCalls(fn, int(simtime / dt))  # replicates time condition
    # pnl.TimeScale.TRIAL: pnl.TimeInterval(end=100, unit='ms')
}

print('Running the SimpleFN model...')

comp.run(inputs={fn: 0}, log=True)


print('Finished running the SimpleFN model')


base_fname = __file__.replace('.py', '')
with open(f'{base_fname}.json', 'w') as outfi:
    outfi.write(comp.json_summary)

with open(f'{base_fname}.converted.py', 'w') as outfi:
    outfi.write(pnl.generate_script_from_json(comp.json_summary))
    outfi.write('\ncomp.show_graph()\n')

for node in comp.nodes:
    print(f'=== {node} {node.name}: {node.parameters.value.get(comp)}')


def generate_time_array(node, context='comp', param='value'):
    return [entry.time.pass_ for entry in getattr(node.parameters, param).log[context]]


def generate_value_array(node, index, context='comp', param='value'):
    return [float(entry.value[index]) for entry in getattr(node.parameters, param).log[context]]


'''
for node in comp.nodes:
    print(f'>> {node}: {generate_time_array(node)}')

    for i in [0,1,2]:
        print(f'>> {node}: {generate_value_array(node,i)}')'''


fig, axes = plt.subplots()
for i in [0, 1]:
    x_values = {node: generate_time_array(node) for node in comp.nodes}
    y_values = {node: generate_value_array(node, i) for node in comp.nodes}

    fout = open('SimpleFN-timing_%i.dat' % i, 'w')
    for index in range(len(x_values[node])):
        # 1000 to convert ms to s
        fout.write(
            f'{0}\t{1}\n'.format(
                x_values[node][index] * time_step_size / 1000.0,
                y_values[node][index]
            )
        )
    fout.close()

    for node in comp.nodes:
        axes.plot(
            [t * time_step_size / 1000.0 for t in x_values[node]],
            y_values[node],
            label=f'Value of {i} {node.name}, {node.function.__class__.__name__}'
        )

axes.set_xlabel('Time (s)')
axes.legend()
plt.show()
