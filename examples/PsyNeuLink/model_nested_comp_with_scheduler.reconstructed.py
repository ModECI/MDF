import psyneulink as pnl

comp = pnl.Composition(name="comp")

A = pnl.TransferMechanism(
    name="A",
    function=pnl.Linear(intercept=2.0, slope=5.0, default_variable=[[0]]),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
B = pnl.TransferMechanism(
    name="B",
    function=pnl.Logistic(default_variable=[[0]]),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
C = pnl.RecurrentTransferMechanism(
    name="C",
    function=pnl.Linear(default_variable=[[0]]),
    initial_value=[[0]],
    output_ports=["RESULTS"],
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
D = pnl.IntegratorMechanism(
    name="D", function=pnl.SimpleIntegrator(initializer=[[0]], default_variable=[[0]])
)

Inner_Composition = pnl.Composition(name="Inner Composition")

E = pnl.TransferMechanism(
    name="E",
    function=pnl.Linear(default_variable=[[0]]),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
F = pnl.TransferMechanism(
    name="F",
    function=pnl.Linear(default_variable=[[0]]),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)

Inner_Composition.add_node(E)
Inner_Composition.add_node(F)

Inner_Composition.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from E[RESULT] to F[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]]),
    ),
    sender=E,
    receiver=F,
)


Inner_Composition.scheduler.termination_conds = {
    pnl.TimeScale.RUN: pnl.Never(),
    pnl.TimeScale.TRIAL: pnl.AllHaveRun(),
}

comp.add_node(A)
comp.add_node(B)
comp.add_node(C)
comp.add_node(D)
comp.add_node(Inner_Composition)

comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from A[RESULT] to B[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]], default_variable=[2.0]),
    ),
    sender=A,
    receiver=B,
)
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from A[RESULT] to C[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]], default_variable=[2.0]),
    ),
    sender=A,
    receiver=C,
)
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from B[RESULT] to D[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]], default_variable=[0.5]),
    ),
    sender=B,
    receiver=D,
)
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from C[RESULT] to D[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]]),
    ),
    sender=C,
    receiver=D,
)
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from C[RESULT] to Inner Composition Input_CIM[INPUT_CIM_E_InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]]),
    ),
    sender=C,
    receiver=Inner_Composition,
)

comp.scheduler.add_condition(A, pnl.EveryNPasses(1, pnl.TimeScale.TRIAL))
comp.scheduler.add_condition(B, pnl.EveryNCalls(A, 2))
comp.scheduler.add_condition(C, pnl.EveryNCalls(B, 2))

comp.scheduler.termination_conds = {
    pnl.TimeScale.RUN: pnl.AfterNTrials(1, pnl.TimeScale.RUN),
    pnl.TimeScale.TRIAL: pnl.AfterNCalls(D, 4),
}
