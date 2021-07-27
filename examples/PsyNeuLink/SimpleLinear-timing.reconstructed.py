import psyneulink as pnl

comp = pnl.Composition(name="comp")

A = pnl.TransferMechanism(
    name="A",
    function=pnl.Linear(default_variable=[[0]]),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
B = pnl.TransferMechanism(
    name="B",
    function=pnl.Linear(default_variable=[[0]]),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
C = pnl.TransferMechanism(
    name="C",
    function=pnl.Linear(default_variable=[[0]]),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)

comp.add_node(A)
comp.add_node(B)
comp.add_node(C)

comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from A[RESULT] to B[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]]),
    ),
    sender=A,
    receiver=B,
)
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from B[RESULT] to C[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]]),
    ),
    sender=B,
    receiver=C,
)

comp.scheduler.add_condition(
    A,
    pnl.TimeInterval(
        end=None,
        end_inclusive=True,
        repeat="7 ms",
        start=None,
        start_inclusive=True,
        unit="ms",
    ),
)
comp.scheduler.add_condition(
    B,
    pnl.All(
        pnl.TimeInterval(
            end=None,
            end_inclusive=True,
            repeat="1 ms",
            start="1 ms",
            start_inclusive=True,
            unit="ms",
        ),
        pnl.Not(
            pnl.TimeInterval(
                end=None,
                end_inclusive=True,
                repeat="7 ms",
                start="6 ms",
                start_inclusive=True,
                unit="ms",
            )
        ),
        pnl.Not(
            pnl.TimeInterval(
                end=None,
                end_inclusive=True,
                repeat="7 ms",
                start="7 ms",
                start_inclusive=True,
                unit="ms",
            )
        ),
    ),
)
comp.scheduler.add_condition(
    C,
    pnl.TimeInterval(
        end=None,
        end_inclusive=True,
        repeat="7 ms",
        start="6 ms",
        start_inclusive=True,
        unit="ms",
    ),
)

comp.scheduler.termination_conds = {
    pnl.TimeScale.RUN: pnl.Never(),
    pnl.TimeScale.TRIAL: pnl.AllHaveRun(),
}
