import psyneulink as pnl

comp = pnl.Composition(name="comp")

fn = pnl.IntegratorMechanism(
    name="fn",
    function=pnl.FitzHughNagumoIntegrator(
        name="FitzHughNagumoIntegrator Function-0",
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
        name="MappingProjection from fn[OutputPort-0] to im[InputPort-0]",
        function=pnl.LinearMatrix(matrix=[[1.0]], default_variable=[-1.0]),
    ),
    sender=fn,
    receiver=im,
)

comp.scheduler.add_condition(fn, pnl.Always())
comp.scheduler.add_condition(
    im, pnl.All(pnl.EveryNCalls(fn, 20.0), pnl.AfterNCalls(fn, 1600.0))
)

comp.scheduler.termination_conds = {
    pnl.TimeScale.RUN: pnl.Never(),
    pnl.TimeScale.TRIAL: pnl.AfterNCalls(fn, 2000),
}
