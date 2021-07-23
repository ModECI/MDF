import psyneulink as pnl

comp = pnl.Composition(name='model_nested_comp_with_scheduler_2')
inner_comp = pnl.Composition(name='Inner Composition')
A = pnl.TransferMechanism(function=pnl.Linear(slope=5.0, intercept=2.0), name='A')
B = pnl.TransferMechanism(function=pnl.Logistic, name='B')
C = pnl.RecurrentTransferMechanism(name='C')
D = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator, name='D')

E = pnl.TransferMechanism(name='E')
F = pnl.TransferMechanism(name='F')


for m in [E, F]:
    inner_comp.add_node(m)


for m in [A, B, C, D, inner_comp]:
    comp.add_node(m)

comp.add_projection(pnl.MappingProjection(), A, B)
comp.add_projection(pnl.MappingProjection(), A, C)
comp.add_projection(pnl.MappingProjection(), B, D)
comp.add_projection(pnl.MappingProjection(), C, D)
comp.add_projection(pnl.MappingProjection(), C, inner_comp)

inner_comp.add_projection(pnl.MappingProjection(), E, F)

comp.scheduler.add_condition_set({
    A: pnl.EveryNPasses(1),
    B: pnl.EveryNCalls(A, 2),
    C: pnl.EveryNCalls(B, 2)
})

comp.termination_processing = {
    pnl.TimeScale.RUN: pnl.AfterNTrials(1),
    pnl.TimeScale.TRIAL: pnl.AfterNCalls(D, 4)
}
comp.show_graph(output_fmt='pdf',show_node_structure=True,show_projection_labels=True)
