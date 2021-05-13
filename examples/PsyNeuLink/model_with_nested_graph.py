import psyneulink as pnl

comp = pnl.Composition(name='comp')
inner_comp = pnl.Composition(name='Inner Composition')
A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
B = pnl.TransferMechanism(function=pnl.Logistic, name='B')
C = pnl.TransferMechanism(function=pnl.Exponential, name='C')

E = pnl.TransferMechanism(name='E', function=pnl.Linear(slope=2.0))
F = pnl.TransferMechanism(name='F', function=pnl.Linear(intercept=2.0))


for m in [E, F]:
    inner_comp.add_node(m)


for m in [A, B, C, inner_comp]:
    comp.add_node(m)

comp.add_projection(pnl.MappingProjection(), A, B)
comp.add_projection(pnl.MappingProjection(), A, C)
comp.add_projection(pnl.MappingProjection(), C, inner_comp)

inner_comp.add_projection(pnl.MappingProjection(), E, F)

comp.run(inputs={A: 1}, log=True)

print(comp.results)
for node in comp.nodes + inner_comp.nodes:
    print(f'{node.name}: {node.parameters.value.get(comp)}')
