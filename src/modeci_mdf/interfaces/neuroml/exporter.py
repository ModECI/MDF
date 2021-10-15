"""
Simple export of MDF to NeuroML(2/lite) and LEMS.

Work in progress...

"""

import sys
import os
import neuromllite
import lems.api as lems

from modeci_mdf.functions.standard import mdf_functions, substitute_args


def mdf_to_neuroml(graph, save_to=None, format=None, run_duration_sec=2):

    print("Converting graph: %s to NeuroML" % (graph.id))

    net = neuromllite.Network(id=graph.id)
    net.notes = "NeuroMLlite export of {} graph: {}".format(
        format if format else "MDF",
        graph.id,
    )

    model = lems.Model()
    lems_definitions = "%s_lems_definitions.xml" % graph.id

    for node in graph.nodes:
        print("    Node: %s" % node.id)

        node_comp_type = "%s__definition" % node.id
        node_comp = "%s__instance" % node.id

        # Create the ComponentType which defines behaviour of the general class
        ct = lems.ComponentType(node_comp_type, extends="baseCellMembPotDL")
        ct.add(lems.Attachments("only_input_port", "basePointCurrentDL"))
        ct.dynamics.add(
            lems.DerivedVariable(name="V", dimension="none", value="0", exposure="V")
        )
        model.add(ct)

        # Define the Component - an instance of the ComponentType
        comp = lems.Component(node_comp, node_comp_type)
        model.add(comp)

        cell = neuromllite.Cell(id=node_comp, lems_source_file=lems_definitions)
        net.cells.append(cell)

        pop = neuromllite.Population(
            id=node.id,
            size=1,
            component=cell.id,
            properties={"color": "0.2 0.2 0.2", "radius": 3},
        )
        net.populations.append(pop)

        if len(node.input_ports) > 1:
            raise Exception("Currently only max 1 input port supported in NeuroML...")

        for ip in node.input_ports:
            ct.add(lems.Exposure(ip.id, "none"))
            ct.dynamics.add(
                lems.DerivedVariable(
                    name=ip.id,
                    dimension="none",
                    select="only_input_port[*]/I",
                    reduce="add",
                    exposure=ip.id,
                )
            )

        on_start = None

        for p in node.parameters:
            print("Converting %s" % p)
            if p.value is not None:
                try:
                    v_num = float(p.value)
                    ct.add(lems.Parameter(p.id, "none"))
                    comp.parameters[p.id] = v_num
                    print(comp.parameters[p.id])
                except Exception as e:

                    ct.add(lems.Exposure(p.id, "none"))
                    dv = lems.DerivedVariable(
                        name=p.id,
                        dimension="none",
                        value="%s" % (p.value),
                        exposure=p.id,
                    )
                    ct.dynamics.add(dv)

            elif p.function is not None:
                ct.add(lems.Exposure(p.id, "none"))
                func_info = mdf_functions[p.function]
                expr = func_info["expression_string"]
                expr2 = substitute_args(expr, p.args)
                for arg in p.args:
                    expr += ";{}={}".format(arg, p.args[arg])
                dv = lems.DerivedVariable(
                    name=p.id, dimension="none", value="%s" % (expr2), exposure=p.id
                )
                ct.dynamics.add(dv)
            else:
                ct.add(lems.Exposure(p.id, "none"))
                ct.dynamics.add(
                    lems.StateVariable(name=p.id, dimension="none", exposure=p.id)
                )
                if p.default_initial_value:
                    if on_start is None:
                        on_start = lems.OnStart()
                        ct.dynamics.add(on_start)
                    sa = lems.StateAssignment(
                        variable=p.id, value=p.default_initial_value
                    )
                    on_start.actions.append(sa)

                if p.time_derivative:
                    td = lems.TimeDerivative(variable=p.id, value=p.time_derivative)
                    ct.dynamics.add(td)

        if len(node.output_ports) > 1:
            raise Exception("Currently only max 1 output port supported in NeuroML...")

        for op in node.output_ports:
            ct.add(lems.Exposure(op.id, "none"))
            ct.dynamics.add(
                lems.DerivedVariable(
                    name=op.id, dimension="none", value=op.value, exposure=op.id
                )
            )
            only_output_port = "only_output_port"
            ct.add(lems.Exposure(only_output_port, "none"))
            ct.dynamics.add(
                lems.DerivedVariable(
                    name=only_output_port,
                    dimension="none",
                    value=op.id,
                    exposure=only_output_port,
                )
            )

    if len(graph.edges) > 0:

        model.add(
            lems.Include(os.path.join(os.path.dirname(__file__), "syn_definitions.xml"))
        )
        rsDL = neuromllite.Synapse(id="rsDL", lems_source_file=lems_definitions)
        net.synapses.append(rsDL)
        # syn_id = 'silentSyn'
        # silentSynDL = neuromllite.Synapse(id=syn_id, lems_source_file=lems_definitions)

    for edge in graph.edges:
        print(f"    Edge: {edge.id} connects {edge.sender} to {edge.receiver}")

        ssyn_id = "silentSyn_proj_%s" % edge.id
        ssyn_id = "silentSyn_proj_%s" % edge.id
        # ssyn_id = 'silentSynX'
        silentDLin = neuromllite.Synapse(id=ssyn_id, lems_source_file=lems_definitions)

        model.add(lems.Component(ssyn_id, "silentRateSynapseDL"))

        net.synapses.append(silentDLin)

        net.projections.append(
            neuromllite.Projection(
                id="proj_%s" % edge.id,
                presynaptic=edge.sender,
                postsynaptic=edge.receiver,
                synapse=rsDL.id,
                pre_synapse=silentDLin.id,
                type="continuousProjection",
                weight=1,
                random_connectivity=neuromllite.RandomConnectivity(probability=1),
            )
        )

    # Much more todo...
    model.export_to_file(lems_definitions)

    print("Nml net: %s" % net)
    if save_to:
        new_file = net.to_json_file(save_to)
        print("Saved NML to: %s" % save_to)

    ################################################################################
    ###   Build Simulation object & save as JSON

    simtime = 1000 * run_duration_sec
    dt = 0.1
    sim = neuromllite.Simulation(
        id="Sim%s" % net.id,
        network=new_file,
        duration=simtime,
        dt=dt,
        seed=123,
        recordVariables={"OUTPUT": {"all": "*"}},
    )

    recordVariables = {}
    for node in graph.nodes:
        for ip in node.input_ports:
            if not ip.id in recordVariables:
                recordVariables[ip.id] = {}
            recordVariables[ip.id][node.id] = 0

        for p in node.parameters:
            if p.is_stateful():
                if not p.id in recordVariables:
                    recordVariables[p.id] = {}
                recordVariables[p.id][node.id] = 0

        for op in node.output_ports:
            if not op.id in recordVariables:
                recordVariables[op.id] = {}
            recordVariables[op.id][node.id] = 0

    sim.recordVariables = recordVariables
    if save_to:
        sf = sim.to_json_file()

        print("Saved Simulation to: %s" % sf)

    return net, sim


if __name__ == "__main__":

    from modeci_mdf.utils import load_mdf, print_summary

    example = "../../../../examples/MDF/Simple.json"
    verbose = True
    run = False
    if "-run" in sys.argv:
        run = True
        sys.argv.remove("-run")

    if len(sys.argv) >= 2:
        example = sys.argv[1]
        verbose = False

    model = load_mdf(example)
    mod_graph = model.graphs[0]

    print("Loaded Graph:")
    print_summary(mod_graph)

    print("------------------")
    nmllite_file = example.replace(".json", ".nmllite.json")
    # nmllite_file = example.split('/')[-1].replace('.json','.nmllite.json')
    net, sim = mdf_to_neuroml(
        mod_graph, save_to=nmllite_file, format=model.format, run_duration_sec=2
    )

    if run:
        sf = "%s.json" % sim.id
        print("Running model: %s" % sf)
        from neuromllite.NetworkGenerator import generate_and_run

        simulator = "jNeuroML"
        traces, events = generate_and_run(
            sim,
            simulator,
            network=net,
            return_results=True,
            base_dir=None,
            target_dir=None,
            num_processors=1,
        )
        for t in traces:
            if t != "t":  # the time array
                print("    {} = {}".format(t, traces[t][-1]))
