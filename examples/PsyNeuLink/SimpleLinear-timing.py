import psyneulink as pnl

comp = pnl.Composition(name='comp')
A = pnl.TransferMechanism(name='A')
B = pnl.TransferMechanism(name='B')
C = pnl.TransferMechanism(name='C')

comp.add_linear_processing_pathway([A, B, C])

comp.scheduler.add_condition_set({
    A: pnl.TimeInterval(interval=7, unit='ms'),
    B: pnl.All(
        pnl.TimeInterval(start=1, interval=1, unit='ms'),
        pnl.Not(pnl.TimeInterval(start=6, interval=7, unit='ms')),
        pnl.Not(pnl.TimeInterval(start=7, interval=7, unit='ms'))
    ),
    C: pnl.TimeInterval(start=6, interval=7, unit='ms')
})

#  0   1   2   3   4   5   6   7   8   9   10  11  12  13
#  A   B   B   B   B   B   C   A   B   B   B   B   B   C
