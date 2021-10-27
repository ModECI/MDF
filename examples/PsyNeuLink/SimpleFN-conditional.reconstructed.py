import psyneulink as pnl

comp = pnl.Composition(name="comp")

fn = pnl.IntegratorMechanism(
    name="fn",
    function=pnl.FitzHughNagumoIntegrator(
        name="FitzHughNagumoIntegrator_Function_0",
        d_v=1,
        initial_v=-1,
        initializer=[[0]],
        default_variable=[[0]],
    ),
)
im = pnl.IntegratorMechanism(
    name="im",
    function=pnl.AdaptiveIntegrator(
        initializer=[[0]], rate=0.5, default_variable=[[0]]
    ),
)

comp.add_node(fn)
comp.add_node(im)

comp.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_fn_OutputPort_0__to_im_InputPort_0_",
        function=pnl.LinearMatrix(default_variable=[-1.0], matrix=[[1.0]]),
    ),
    sender=fn,
    receiver=im,
)

comp.scheduler.add_condition(fn, pnl.Always())
comp.scheduler.add_condition(
    im,
    pnl.All(
        pnl.EveryNCalls(dependency=fn, n=20.0),
        pnl.AfterNCalls(
            dependency=fn, n=1600.0, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE
        ),
    ),
)

comp.scheduler.termination_conds = {
    pnl.TimeScale.ENVIRONMENT_SEQUENCE: pnl.Never(),
    pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AfterNCalls(
        dependency=fn, n=2000, time_scale=pnl.TimeScale.ENVIRONMENT_STATE_UPDATE
    ),
}
