import psyneulink as pnl

comp = pnl.Composition(name='comp')
A = pnl.TransferMechanism(name='A')
B = pnl.TransferMechanism(name='B')
C = pnl.TransferMechanism(name='C')

comp.add_linear_processing_pathway([A, B, C])

comp.scheduler.add_condition_set({
    A: pnl.AtNCalls(A, 0),
    B: pnl.Always(),
    C: pnl.EveryNCalls(B, 5),
})

comp.run(inputs={A: 1})

# comp.show_graph(output_fmt='pdf',show_node_structure=True,show_projection_labels=True)

# A, B, B, B, B, B, C
print([
    {node.name for node in time_step}
    for time_step in comp.scheduler.execution_list[comp.default_execution_id]
])
