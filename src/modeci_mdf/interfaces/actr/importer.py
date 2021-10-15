"""
    Code for importing ACT-R models into MDF.
"""
from modeci_mdf.mdf import *


def build_model() -> Model:
    """Builds the base model of the ACT-R production system without any chunks,
    goals, or productions specified.

    Returns:
        An MDF model object representing the core ACT-R production system.
    """
    mod = Model(id="ACT-R Base")
    mod_graph = Graph(id="actr_base")
    mod.graphs.append(mod_graph)

    # Declarative memory
    dm_node = Node(id="declarative_memory")
    dm_node.parameters.append(Parameter(id="chunks", value=[]))
    dm_node.parameters.append(Parameter(id="chunk_types", value=[]))
    dm_ip = InputPort(id="dm_input")
    dm_node.input_ports.append(dm_ip)
    retrieval_f = Parameter(
        id="retrieve_chunk",
        function="retrieve_chunk",
        args={"pattern": dm_ip.id, "dm_chunks": "chunks", "types": "chunk_types"},
    )
    dm_node.parameters.append(retrieval_f)
    dm_op = OutputPort(id="dm_output", value=retrieval_f.id)
    dm_node.output_ports.append(dm_op)
    mod_graph.nodes.append(dm_node)

    # Retrieval buffer
    retrieval_node = Node(id="retrieval_buffer")
    retrieval_ip = InputPort(id="retrieval_input")
    retrieval_node.input_ports.append(retrieval_ip)
    retrieval_op = OutputPort(id="retrieval_output", value=retrieval_ip.id)
    retrieval_node.output_ports.append(retrieval_op)
    mod_graph.nodes.append(retrieval_node)

    # Goal buffer with state
    goal_node = Node(id="goal_buffer")
    goal_node.parameters.append(Parameter(id="first_goal", value={}))
    goal_ip = InputPort(id="goal_input")
    goal_node.input_ports.append(goal_ip)
    goal_f = Parameter(
        id="change_goal",
        function="change_goal",
        args={"pattern": goal_ip.id, "curr_goal": "goal_state"},
    )
    goal_node.parameters.append(goal_f)
    goal_state = Parameter(
        id="goal_state",
        default_initial_value="first_goal",
        value=f"first_goal if {goal_ip.id} == {{}} else {goal_f.id}",
    )
    goal_node.parameters.append(goal_state)
    goal_op = OutputPort(id="goal_output", value=goal_state.id)
    goal_node.output_ports.append(goal_op)
    mod_graph.nodes.append(goal_node)

    # Procedural memory
    pm_node = Node(id="procedural_memory")
    pm_node.parameters.append(Parameter(id="productions", value=[]))
    pm_op = OutputPort(id="pm_output", value="productions")
    pm_node.output_ports.append(pm_op)
    mod_graph.nodes.append(pm_node)

    # Pattern matching
    pattern_node = Node(id="pattern_matching")
    pattern_ip1 = InputPort(id="pattern_input_from_pm")
    pattern_ip2 = InputPort(id="pattern_input_from_goal")
    pattern_ip3 = InputPort(id="pattern_input_from_retrieval")
    pattern_node.input_ports.extend([pattern_ip1, pattern_ip2, pattern_ip3])
    pattern_f = Parameter(
        id="pattern_matching_function",
        function="pattern_matching_function",
        args={
            "productions": pattern_ip1.id,
            "goal": pattern_ip2.id,
            "retrieval": pattern_ip3.id,
        },
    )
    pattern_node.parameters.append(pattern_f)
    pattern_op = OutputPort(id="pattern_output", value=pattern_f.id)
    pattern_node.output_ports.append(pattern_op)
    mod_graph.nodes.append(pattern_node)

    # Conflict resolution
    conflict_node = Node(id="conflict_resolution")
    conflict_ip = InputPort(id="conflict_input")
    conflict_node.input_ports.append(conflict_ip)
    conflict_f = Parameter(
        id="conflict_resolution_function",
        function="conflict_resolution_function",
        args={"productions": conflict_ip.id},
    )
    conflict_node.parameters.append(conflict_f)
    conflict_op1 = OutputPort(id="conflict_output_to_fire_prod", value=conflict_f.id)
    conflict_op2 = OutputPort(id="conflict_output_to_check", value=conflict_f.id)
    conflict_node.output_ports.extend([conflict_op1, conflict_op2])
    mod_graph.nodes.append(conflict_node)

    # Node for firing productions
    fire_prod_node = Node(id="fire_production")
    fire_prod_ip = InputPort(id="fire_prod_input")
    fire_prod_node.input_ports.append(fire_prod_ip)
    fire_prod_f1 = Parameter(
        id="update_goal", function="update_goal", args={"production": fire_prod_ip.id}
    )
    fire_prod_f2 = Parameter(
        id="update_retrieval",
        function="update_retrieval",
        args={"production": fire_prod_ip.id},
    )
    fire_prod_node.parameters.extend([fire_prod_f1, fire_prod_f2])
    fire_prod_op1 = OutputPort(id="fire_prod_output_to_goal", value=fire_prod_f1.id)
    fire_prod_op2 = OutputPort(
        id="fire_prod_output_to_retrieval", value=fire_prod_f2.id
    )
    fire_prod_node.output_ports.extend([fire_prod_op1, fire_prod_op2])
    mod_graph.nodes.append(fire_prod_node)

    # Node to check termination
    check_node = Node(id="check_termination")
    check_ip = InputPort(id="check_input")
    check_node.input_ports.append(check_ip)
    check_f = Parameter(
        id="check_termination",
        function="check_termination",
        args={"production": check_ip.id},
    )
    check_node.parameters.append(check_f)
    check_op = OutputPort(id="check_output", value=check_f.id)
    check_node.output_ports.append(check_op)
    mod_graph.nodes.append(check_node)

    # Edges
    dm_to_retrieval = Edge(
        id="dm_to_pattern_edge",
        sender=dm_node.id,
        sender_port=dm_op.id,
        receiver=retrieval_node.id,
        receiver_port=retrieval_ip.id,
    )
    mod_graph.edges.append(dm_to_retrieval)

    retrieval_to_pattern = Edge(
        id="retrieval_to_pattern_edge",
        sender=retrieval_node.id,
        sender_port=retrieval_op.id,
        receiver=pattern_node.id,
        receiver_port=pattern_ip3.id,
    )
    mod_graph.edges.append(retrieval_to_pattern)

    goal_to_pattern = Edge(
        id="goal_to_pattern_edge",
        sender=goal_node.id,
        sender_port=goal_op.id,
        receiver=pattern_node.id,
        receiver_port=pattern_ip2.id,
    )
    mod_graph.edges.append(goal_to_pattern)

    pm_to_pattern = Edge(
        id="pm_to_pattern_edge",
        sender=pm_node.id,
        sender_port=pm_op.id,
        receiver=pattern_node.id,
        receiver_port=pattern_ip1.id,
    )
    mod_graph.edges.append(pm_to_pattern)

    pattern_to_conflict = Edge(
        id="pattern_to_conflict_edge",
        sender=pattern_node.id,
        sender_port=pattern_op.id,
        receiver=conflict_node.id,
        receiver_port=conflict_ip.id,
    )
    mod_graph.edges.append(pattern_to_conflict)

    conflict_to_fire_prod = Edge(
        id="conflict_to_fire_prod_edge",
        sender=conflict_node.id,
        sender_port=conflict_op1.id,
        receiver=fire_prod_node.id,
        receiver_port=fire_prod_ip.id,
    )
    mod_graph.edges.append(conflict_to_fire_prod)

    conflict_to_check = Edge(
        id="conflict_to_check_edge",
        sender=conflict_node.id,
        sender_port=conflict_op1.id,
        receiver=check_node.id,
        receiver_port=check_ip.id,
    )
    mod_graph.edges.append(conflict_to_check)

    # Conditions
    cond_dm = Condition(type="Always")
    cond_retrieval = Condition(type="JustRan", dependencies=dm_node.id)
    cond_goal = Condition(type="Always")
    cond_pm = Condition(type="Always")
    cond_pattern = Condition(
        type="And",
        dependencies=[
            Condition(type="EveryNCalls", dependencies=retrieval_node.id, n=1),
            Condition(type="EveryNCalls", dependencies=goal_node.id, n=1),
            Condition(type="EveryNCalls", dependencies=dm_node.id, n=1),
        ],
    )
    cond_conflict = Condition(type="JustRan", dependencies=pattern_node.id)
    cond_fire_prod = Condition(type="JustRan", dependencies=conflict_node.id)
    cond_check = Condition(type="JustRan", dependencies=conflict_node.id)
    cond_term = Condition(type="JustRan", dependencies=[check_node.id])
    mod_graph.conditions = ConditionSet(
        node_specific={
            dm_node.id: cond_dm,
            retrieval_node.id: cond_retrieval,
            goal_node.id: cond_goal,
            pm_node.id: cond_pm,
            pattern_node.id: cond_pattern,
            conflict_node.id: cond_conflict,
            fire_prod_node.id: cond_fire_prod,
            check_node.id: cond_check,
        },
        termination={"check_term_true": cond_term},
    )

    return mod


def actr_to_mdf(file_name: str):
    """Parses an ACT-R .lisp model file and outputs MDF .json and .yaml files.

    Args:
        file_name: The name of the ACT-R model file ending in .lisp.
    """
    with open(file_name) as actr_file:
        mod = build_model()
        add_dm = False
        add_lhs = False
        add_rhs = False
        chunk_types = {}
        dm = []
        pm = []
        goal = None
        curr_prod = None
        curr_pattern = None
        for l in actr_file:
            line = l.strip()
            # Add the chunk types
            if line.startswith("(chunk-type"):
                tokens = line.replace("(", "").replace(")", "").split(" ")
                chunk_types[tokens[1]] = tokens[2:]
            # Add the chunks to declarative memory
            elif line.startswith("(add-dm"):
                add_dm = True
            elif add_dm:
                if line == "":
                    add_dm = False
                else:
                    chunk = {}
                    tokens = line.replace("(", "").replace(")", "").split(" ")
                    chunk["name"] = tokens[0]
                    for i in range(1, len(tokens), 2):
                        chunk[tokens[i]] = tokens[i + 1]
                    dm.append(chunk)
            # Add the goal to the goal buffer
            elif line.startswith("(goal-focus"):
                goal_name = line[12:-1]
                for chunk in dm:
                    if chunk["name"] == goal_name:
                        goal = chunk
                        break
                for k in chunk_types[goal["ISA"]]:
                    if k not in goal.keys():
                        goal[k] = "nil"
            # Add the productions to procedural memory
            elif line.startswith(("(P", "(p")):
                curr_prod = {"name": line[3:], "lhs": [], "rhs": []}
                pm.append(curr_prod)
                add_lhs = True
            # Left hand side of production
            elif add_lhs:
                if line == "==>":
                    add_lhs = False
                    add_rhs = True
                elif line.endswith(">"):
                    curr_pattern = {"buffer": line[1:-1]}
                    curr_prod["lhs"].append(curr_pattern)
                else:
                    tokens = [t for t in line.split(" ") if t != ""]
                    if tokens[0] == "-":
                        curr_pattern[tokens[1]] = "-" + tokens[2]
                    else:
                        curr_pattern[tokens[0]] = tokens[1]
            # Right hand side of production
            elif add_rhs:
                if line == "" or line == ")" or "!output!" in line:
                    add_rhs = False
                elif line.endswith(">"):
                    curr_pattern = {"buffer": line[1:-1]}
                    curr_prod["rhs"].append(curr_pattern)
                else:
                    tokens = [t for t in line.split(" ") if t != ""]
                    if len(tokens) == 3:
                        curr_pattern[tokens[0] + tokens[1]] = tokens[2]
                    else:
                        curr_pattern[tokens[0]] = tokens[1]

        # Update production patterns
        for prod in pm:
            lhs = []
            for pattern in prod["lhs"]:
                isa = pattern["ISA"]
                final_pattern = {"buffer": pattern["buffer"], "ISA": isa}
                for k in chunk_types[isa]:
                    if k in pattern.keys():
                        final_pattern[k] = pattern[k]
                    else:
                        final_pattern[k] = "=" + k
                lhs.append(final_pattern)
            prod["lhs"] = lhs

        # Generate MDF files
        mod.id = (
            file_name[file_name.rindex("/") + 1 : -5]
            if "/" in file_name
            else file_name[:-5]
        )
        mod.graphs[0].id = mod.id + "_graph"
        mod.graphs[0].get_node("declarative_memory").get_parameter("chunks").value = dm
        mod.graphs[0].get_node("declarative_memory").get_parameter(
            "chunk_types"
        ).value = chunk_types
        mod.graphs[0].get_node("procedural_memory").get_parameter(
            "productions"
        ).value = pm
        mod.graphs[0].get_node("goal_buffer").get_parameter("first_goal").value = goal
        mod.to_json_file("%s.json" % file_name[:-5])
        mod.to_yaml_file("%s.yaml" % file_name[:-5])

        return mod
