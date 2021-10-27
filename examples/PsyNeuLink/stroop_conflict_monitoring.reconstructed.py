import psyneulink as pnl

Stroop_model = pnl.Composition(name="Stroop_model")

color_input = pnl.ProcessingMechanism(
    name="color_input",
    function=pnl.Linear(default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
color_hidden = pnl.ProcessingMechanism(
    name="color_hidden",
    function=pnl.Logistic(bias=-4.0, default_variable=[[0.0, 0.0]]),
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
    function=pnl.Logistic(bias=-4.0, default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
task_input = pnl.ProcessingMechanism(
    name="task_input",
    function=pnl.Linear(default_variable=[[0.0, 0.0]]),
    default_variable=[[0.0, 0.0]],
)
TASK = pnl.LCAMechanism(
    name="TASK",
    combination_function=pnl.LinearCombination(default_variable=[[0.0, 0.0]]),
    function=pnl.Logistic(default_variable=[[0.0, 0.0]]),
    integrator_function=pnl.LeakyCompetingIntegrator(
        name="LeakyCompetingIntegrator_Function_0",
        initializer=[[0.5, 0.5]],
        rate=0.5,
        default_variable=[[0.0, 0.0]],
    ),
    output_ports=["RESULTS"],
    termination_comparison_op=">=",
    default_variable=[[0.0, 0.0]],
)
DECISION = pnl.DDM(
    name="DECISION",
    function=pnl.DriftDiffusionAnalytical(default_variable=[[0.0]]),
    input_ports=[
        {
            pnl.NAME: pnl.ARRAY,
            pnl.VARIABLE: [[0.0, 0.0]],
            pnl.FUNCTION: pnl.Reduce(default_variable=[[0.0, 0.0]], weights=[1, -1]),
        }
    ],
)
Conflict_Monitor = pnl.ObjectiveMechanism(
    name="Conflict_Monitor",
    function=pnl.Energy(matrix=[[0, -2.5], [-2.5, 0]], default_variable=[[0.0, 0.0]]),
    monitor=[OUTPUT],
    default_variable=[[0.0, 0.0]],
)

CONTROL = pnl.ControlMechanism(
    name="CONTROL",
    default_allocation=[0.5],
    function=pnl.DefaultAllocationFunction(default_variable=[[1.0]]),
    monitor_for_control=[],
    objective_mechanism=Conflict_Monitor,
    control=[{pnl.NAME: pnl.GAIN, pnl.MECHANISM: TASK}],
)

Stroop_model.add_node(color_input)
Stroop_model.add_node(color_hidden)
Stroop_model.add_node(OUTPUT)
Stroop_model.add_node(word_input)
Stroop_model.add_node(word_hidden)
Stroop_model.add_node(task_input)
Stroop_model.add_node(TASK)
Stroop_model.add_node(DECISION)
Stroop_model.add_node(Conflict_Monitor, pnl.NodeRole.CONTROLLER_OBJECTIVE)

Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_task_input_OutputPort_0__to_TASK_InputPort_0_",
        function=pnl.LinearMatrix(
            default_variable=[0.0, 0.0], matrix=[[1.0, 0.0], [0.0, 1.0]]
        ),
    ),
    sender=task_input,
    receiver=TASK,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_OUTPUT_OutputPort_0__to_DECISION_ARRAY_",
        function=pnl.LinearMatrix(
            default_variable=[0.5, 0.5], matrix=[[1.0, 0.0], [0.0, 1.0]]
        ),
    ),
    sender=OUTPUT,
    receiver=DECISION,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_OUTPUT_OutputPort_0__to_Conflict_Monitor_InputPort_0_",
        function=pnl.LinearMatrix(
            default_variable=[0.5, 0.5], matrix=[[1.0, 0.0], [0.0, 1.0]]
        ),
    ),
    sender=OUTPUT,
    receiver=Conflict_Monitor,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_color_input_OutputPort_0__to_color_hidden_InputPort_0",
        function=pnl.LinearMatrix(
            default_variable=[0.0, 0.0], matrix=[[2.0, -2.0], [-2.0, 2.0]]
        ),
    ),
    sender=color_input,
    receiver=color_hidden,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_color_hidden_OutputPort_0__to_OUTPUT_InputPort_0",
        function=pnl.LinearMatrix(
            default_variable=[0.017986209962091562, 0.017986209962091562],
            matrix=[[2.0, -2.0], [-2.0, 2.0]],
        ),
    ),
    sender=color_hidden,
    receiver=OUTPUT,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_word_input_OutputPort_0__to_word_hidden_InputPort_0",
        function=pnl.LinearMatrix(
            default_variable=[0.0, 0.0], matrix=[[3.0, -3.0], [-3.0, 3.0]]
        ),
    ),
    sender=word_input,
    receiver=word_hidden,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_word_hidden_OutputPort_0__to_OUTPUT_InputPort_0",
        function=pnl.LinearMatrix(
            default_variable=[0.017986209962091562, 0.017986209962091562],
            matrix=[[3.0, -3.0], [-3.0, 3.0]],
        ),
    ),
    sender=word_hidden,
    receiver=OUTPUT,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_TASK_RESULT__to_color_hidden_InputPort_0",
        function=pnl.LinearMatrix(
            default_variable=[0.5, 0.5], matrix=[[4.0, 4.0], [0.0, 0.0]]
        ),
    ),
    sender=TASK,
    receiver=color_hidden,
)
Stroop_model.add_projection(
    projection=pnl.MappingProjection(
        name="MappingProjection_from_TASK_RESULT__to_word_hidden_InputPort_0",
        function=pnl.LinearMatrix(
            default_variable=[0.5, 0.5], matrix=[[0.0, 0.0], [4.0, 4.0]]
        ),
    ),
    sender=TASK,
    receiver=word_hidden,
)
Stroop_model.add_controller(CONTROL)

Stroop_model.scheduler.add_condition(
    word_hidden, pnl.EveryNCalls(dependency=TASK, n=10)
)
Stroop_model.scheduler.add_condition(
    color_hidden, pnl.EveryNCalls(dependency=TASK, n=10)
)
Stroop_model.scheduler.add_condition(
    OUTPUT,
    pnl.All(
        pnl.EveryNCalls(dependency=color_hidden, n=1),
        pnl.EveryNCalls(dependency=word_hidden, n=1),
    ),
)
Stroop_model.scheduler.add_condition(DECISION, pnl.EveryNCalls(dependency=OUTPUT, n=1))

Stroop_model.scheduler.termination_conds = {
    pnl.TimeScale.ENVIRONMENT_SEQUENCE: pnl.Never(),
    pnl.TimeScale.ENVIRONMENT_STATE_UPDATE: pnl.AllHaveRun(),
}
