import psyneulink as pnl

ABCD = pnl.Composition(name="ABCD")

A = pnl.TransferMechanism(
    name="A",
    function=pnl.Linear(intercept=2.0, slope=2.0, default_variable=[[0]]),
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
D = pnl.IntegratorMechanism(
    name="D",
    function=pnl.SimpleIntegrator(initializer=[[0]], rate=0.05, default_variable=[[0]]),
)

ABCD.add_node(A)
ABCD.add_node(B)
ABCD.add_node(C)
ABCD.add_node(D)

ABCD.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_A_RESULT__to_B_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[2.0], matrix=[[1.0]]),
    ),
    sender=A,
    receiver=B,
)
ABCD.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_B_RESULT__to_D_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[0.5], matrix=[[1.0]]),
    ),
    sender=B,
    receiver=D,
)
ABCD.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_A_RESULT__to_C_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[2.0], matrix=[[1.0]]),
    ),
    sender=A,
    receiver=C,
)
ABCD.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_C_RESULT__to_D_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[1.0], matrix=[[1.0]]),
    ),
    sender=C,
    receiver=D,
)

ABCD.scheduler.add_condition(A, pnl.Always())
ABCD.scheduler.add_condition(C, pnl.EveryNCalls(dependency=A, n=1))
ABCD.scheduler.add_condition(B, pnl.EveryNCalls(dependency=A, n=1))
ABCD.scheduler.add_condition(
    D, pnl.All(pnl.EveryNCalls(dependency=C, n=1), pnl.EveryNCalls(dependency=B, n=1))
)

ABCD.scheduler.termination_conds = {
    pnl.TimeScale.ENVIRONMENT_SEQUENCE: pnl.Never(),
    pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AllHaveRun(),
}
