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
C = pnl.RecurrentTransferMechanism(
    name="C",
    combination_function=pnl.LinearCombination(default_variable=[[0]]),
    function=pnl.Linear(default_variable=[[0]]),
    integrator_function=pnl.AdaptiveIntegrator(
        initializer=[[0]], rate=0.5, default_variable=[[0]]
    ),
    output_ports=["RESULTS"],
    termination_measure=pnl.Distance(
        metric=pnl.MAX_ABS_DIFF, default_variable=[[[0]], [[0]]]
    ),
)
D = pnl.IntegratorMechanism(
    name="D", function=pnl.SimpleIntegrator(initializer=[[0]], default_variable=[[0]])
)
Inner_Composition = pnl.Composition(name="Inner_Composition")

comp.add_node(A)
comp.add_node(B)
comp.add_node(C)
comp.add_node(D)
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
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_B_RESULT__to_D_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[0.5], matrix=[[1.0]]),
    ),
    sender=B,
    receiver=D,
)
comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_C_RESULT__to_D_InputPort_0_",
        function=pnl.LinearMatrix(matrix=[[1.0]]),
    ),
    sender=C,
    receiver=D,
)

comp.scheduler.add_condition(
    A, pnl.EveryNPasses(n=1, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE)
)
comp.scheduler.add_condition(B, pnl.EveryNCalls(dependency=A, n=2))
comp.scheduler.add_condition(C, pnl.EveryNCalls(dependency=B, n=2))

comp.scheduler.termination_conds = {
    pnl.TimeScale.ENVIRONMENT_SEQUENCE: pnl.AfterNTrials(
        n=1, time_scale=pnl.TimeScale.ENVIRONMENT_SEQUENCE
    ),
    pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(
        dependency=D, n=4, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE
    ),
}
