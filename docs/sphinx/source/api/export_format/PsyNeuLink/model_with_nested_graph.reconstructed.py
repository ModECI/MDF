import psyneulink as pnl

comp = pnl.Composition(name="comp")

A = pnl.TransferMechanism(
    name="A",
    function=pnl.Linear(intercept=2.0, slope=5.0, default_variable=[[0]]),
    integrator_function=pnl.AdaptiveIntegrator(
        initializer=[[0]], rate=0.5, default_variable=[[0]]
    ),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
B = pnl.TransferMechanism(
    name="B",
    function=pnl.Logistic(default_variable=[[0]]),
    integrator_function=pnl.AdaptiveIntegrator(
        initializer=[[0]], rate=0.5, default_variable=[[0]]
    ),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
C = pnl.TransferMechanism(
    name="C",
    function=pnl.Exponential(default_variable=[[0]]),
    integrator_function=pnl.AdaptiveIntegrator(
        initializer=[[0]], rate=0.5, default_variable=[[0]]
    ),
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
Inner_Composition = pnl.Composition(name="Inner_Composition")

comp.add_node(A)
comp.add_node(B)
comp.add_node(C)
comp.add_node(Inner_Composition)

comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_A_RESULT__to_B_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[2.0], matrix=[[1.0]]),
    ),
    sender=A,
    receiver=B,
)
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_A_RESULT__to_C_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[2.0], matrix=[[1.0]]),
    ),
    sender=A,
    receiver=C,
)

comp.scheduler.add_condition(A, pnl.Always())
comp.scheduler.add_condition(B, pnl.EveryNCalls(dependency=A, n=1))
comp.scheduler.add_condition(C, pnl.EveryNCalls(dependency=A, n=1))
comp.scheduler.add_condition(Inner_Composition, pnl.EveryNCalls(dependency=C, n=1))

comp.scheduler.termination_conds = {
    pnl.TimeScale.ENVIRONMENT_SEQUENCE: pnl.Never(),
    pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AllHaveRun(),
}
