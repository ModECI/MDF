import psyneulink as pnl
import sys

from utilities import TestIntegrator

dt = 0.1
simtime = 1

time_step_size=dt
num_trials=int(simtime/dt)

fn_i = pnl.FitzHughNagumoIntegrator(
    initial_v=-1,
    initial_w=0,
    d_v=1,
    time_step_size=time_step_size,
)
s_i = pnl.SimpleIntegrator(
    rate=1.1,
    offset=2.2,
)
t_i = TestIntegrator(
        rate=3.3,
        offset=4.4,
)

print('Running simple models for %sms: %s'%(simtime, fn_i))
print('Running simple models for %sms: %s'%(simtime, s_i))
print('Running simple models for %sms: %s'%(simtime, t_i))

fn_m = pnl.IntegratorMechanism(name='fitz', function=fn_i)
s_m = pnl.IntegratorMechanism(name='simple', function=s_i)
t_m = pnl.IntegratorMechanism(name='test', function=t_i)

comp = pnl.Composition(name='comp')
comp.add_node(fn_m)
comp.add_node(s_m)
comp.add_node(t_m)

print('Running the model...')

comp.run(inputs={fn_m:0}, log=True, num_trials=num_trials)

print('Finished running the model')


for node in comp.nodes:
    print(f'>=== {node}: {node.name}: {node.parameters.value.get(comp)}')

import matplotlib.pyplot as plt

def generate_time_array(node, context=comp.default_execution_id, param='value'):
    return [entry.time.trial for entry in getattr(node.parameters, param).log[context]]

def generate_value_array(node, index, context=comp.default_execution_id, param='value'):
    return [float(entry.value[index]) for entry in getattr(node.parameters, param).log[context]]

def get_size_node_values(node, context=comp.default_execution_id, param='value'):
    return len(getattr(node.parameters, param).log[context][0].value)

for node in comp.nodes:
    print(f'>> {node} time: {generate_time_array(node)}')

    for i in range(get_size_node_values(node)):
        print(f'>> {node} value of variable {i}: {generate_value_array(node,i)}')


fig, axes = plt.subplots()
for i in [0]:
    x_values = {node: generate_time_array(node) for node in comp.nodes}
    y_values = {node: generate_value_array(node, i) for node in comp.nodes}

    fout = open('FN_%i.dat'%i,'w')
    for index in range(len(x_values[node])):
        #                                            1000 to convert ms to s
        fout.write('%s\t%s\n'%(x_values[node][index]*time_step_size/1000.0, \
                               y_values[node][index]))
    fout.close()

    for node in comp.nodes:
        axes.plot(
            [t*time_step_size/1000.0 for t in x_values[node]],
            y_values[node],
            label=f'Value of {i} {node.name}, {node.function.__class__.__name__}'
        )

axes.set_xlabel('Time (s)')
axes.legend()
plt.show()
