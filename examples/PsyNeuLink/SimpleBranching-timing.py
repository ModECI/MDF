import psyneulink as pnl

comp = pnl.Composition(name='comp')
A = pnl.TransferMechanism(name='A')
B = pnl.TransferMechanism(name='B')
C = pnl.TransferMechanism(name='C')
D = pnl.TransferMechanism(name='D')

comp.add_linear_processing_pathway([A, B, C])
comp.add_linear_processing_pathway([A, B, D])

# TimeInterval is not yet implemented in PsyNeuLink
comp.scheduler.add_condition_set({
    A: pnl.TimeInterval(repeat=7, unit='ms'),
    B: pnl.All(
        pnl.TimeInterval(start=1, repeat=1, unit='ms'),
        pnl.Not(pnl.TimeInterval(start=6, repeat=7, unit='ms')),
        pnl.Not(pnl.TimeInterval(start=7, repeat=7, unit='ms'))
    ),
    C: pnl.TimeInterval(start=6, repeat=7, unit='ms'),
    D: pnl.TimeInterval(start=13, repeat=7, unit='ms')
})

comp.run(inputs={A: 1}, scheduling_mode=pnl.SchedulingMode.EXACT_TIME)

print('\n'.join([
    '{0:~}: {1}'.format(
        comp.scheduler.execution_timestamps[comp.default_execution_id][i].absolute,
        {node.name for node in time_step}
    )
    for i, time_step in enumerate(comp.scheduler.execution_list[comp.default_execution_id])
]))

#  0   1   2   3   4   5   6   7   8   9   10  11  12  13
#  A   B   B   B   B   B   C   A   B   B   B   B   B   CD
