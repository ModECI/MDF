import psyneulink as pnl

Stroop_model = pnl.Composition(name="Stroop_model")

color_input = pnl.ProcessingMechanism(
    name="color_input",
    function=pnl.Linear(default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
color_hidden = pnl.ProcessingMechanism(
    name="color_hidden",
    function=pnl.Logistic(bias=-4, default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
OUTPUT = pnl.ProcessingMechanism(
    name="OUTPUT",
    function=pnl.Logistic(default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
word_input = pnl.ProcessingMechanism(
    name="word_input",
    function=pnl.Linear(default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
word_hidden = pnl.ProcessingMechanism(
    name="word_hidden",
    function=pnl.Logistic(bias=-4, default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
task_input = pnl.ProcessingMechanism(
    name="task_input",
    function=pnl.Linear(default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
TASK = pnl.LCAMechanism(
    name="TASK",
    function=pnl.Logistic(default_variable=[[0.0, 0.0]]),
    initial_value=[[0.5, 0.5]],
    output_ports=["RESULTS"],
    termination_comparison_op=">=",
    default_variable=[[0.0, 0.0]],
)
DECISION = pnl.DDM(
    name="DECISION",
    function=pnl.DriftDiffusionAnalytical(default_variable=[[0.0]]),
    input_ports=[
        {
            pnl.FUNCTION: pnl.Reduce(default_variable=[[0.0, 0.0]], weights=[1, -1]),
            pnl.NAME: pnl.ARRAY,
            pnl.VARIABLE: [[0.0, 0.0]],
        }
    ],
)

CONTROL = pnl.ControlMechanism(
    name="CONTROL",
    default_allocation=[0.5],
    function=pnl.DefaultAllocationFunction(default_variable=[[1.0]]),
    monitor_for_control=[],
    objective_mechanism=pnl.ObjectiveMechanism(
        name="Conflict Monitor",
        function=pnl.Energy(
            matrix=[[0, -2.5], [-2.5, 0]], default_variable=[[0.0, 0.0]]
        ),
        monitor=[OUTPUT],
        default_variable=[[0.0, 0.0]],
    ),
    control=[{pnl.MECHANISM: TASK, pnl.NAME: pnl.GAIN}],
)

Stroop_model.add_node(color_input)
Stroop_model.add_node(color_hidden)
Stroop_model.add_node(OUTPUT)
Stroop_model.add_node(word_input)
Stroop_model.add_node(word_hidden)
Stroop_model.add_node(task_input)
Stroop_model.add_node(TASK)
Stroop_model.add_node(DECISION)

Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from OUTPUT[OutputPort-0] to DECISION[ARRAY]",
        function=pnl.LinearMatrix(
            matrix=[[1.0, 0.0], [0.0, 1.0]], default_variable=[0.5, 0.5]
        ),
    ),
    sender=OUTPUT,
    receiver=DECISION,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from TASK[RESULT] to color_hidden[InputPort-0]",
        function=pnl.LinearMatrix(
            matrix=[[4.0, 4.0], [0.0, 0.0]], default_variable=[0.5, 0.5]
        ),
        matrix=[[4, 4], [0, 0]],
    ),
    sender=TASK,
    receiver=color_hidden,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from TASK[RESULT] to word_hidden[InputPort-0]",
        function=pnl.LinearMatrix(
            matrix=[[0.0, 0.0], [4.0, 4.0]], default_variable=[0.5, 0.5]
        ),
        matrix=[[0, 0], [4, 4]],
    ),
    sender=TASK,
    receiver=word_hidden,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from color_hidden[OutputPort-0] to OUTPUT[InputPort-0]",
        function=pnl.LinearMatrix(
            matrix=[[2.0, -2.0], [-2.0, 2.0]],
            default_variable=[0.017986209962091562, 0.017986209962091562],
        ),
        matrix=[[2, -2], [-2, 2]],
    ),
    sender=color_hidden,
    receiver=OUTPUT,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from color_input[OutputPort-0] to color_hidden[InputPort-0]",
        function=pnl.LinearMatrix(
            matrix=[[2.0, -2.0], [-2.0, 2.0]], default_variable=[0.0, 0.0]
        ),
        matrix=[[2, -2], [-2, 2]],
    ),
    sender=color_input,
    receiver=color_hidden,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from task_input[OutputPort-0] to TASK[InputPort-0]",
        function=pnl.LinearMatrix(
            matrix=[[1.0, 0.0], [0.0, 1.0]], default_variable=[0.0, 0.0]
        ),
    ),
    sender=task_input,
    receiver=TASK,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from word_hidden[OutputPort-0] to OUTPUT[InputPort-0]",
        function=pnl.LinearMatrix(
            matrix=[[3.0, -3.0], [-3.0, 3.0]],
            default_variable=[0.017986209962091562, 0.017986209962091562],
        ),
        matrix=[[3, -3], [-3, 3]],
    ),
    sender=word_hidden,
    receiver=OUTPUT,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection from word_input[OutputPort-0] to word_hidden[InputPort-0]",
        function=pnl.LinearMatrix(
            matrix=[[3.0, -3.0], [-3.0, 3.0]], default_variable=[0.0, 0.0]
        ),
        matrix=[[3, -3], [-3, 3]],
    ),
    sender=word_input,
    receiver=word_hidden,
)
Stroop_model.add_controller(CONTROL)

Stroop_model.scheduler.add_condition(DECISION, pnl.EveryNCalls(OUTPUT, 1))
Stroop_model.scheduler.add_condition(
    OUTPUT, pnl.All(pnl.EveryNCalls(color_hidden, 1), pnl.EveryNCalls(word_hidden, 1))
)
Stroop_model.scheduler.add_condition(color_hidden, pnl.EveryNCalls(TASK, 10))
Stroop_model.scheduler.add_condition(word_hidden, pnl.EveryNCalls(TASK, 10))

Stroop_model.scheduler.termination_conds = {
    pnl.TimeScale.RUN: pnl.Never(),
    pnl.TimeScale.TRIAL: pnl.AllHaveRun(),
}
