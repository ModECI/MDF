'''
Simple export of MDF to NeuroML(2/lite) & LEMS... 

Work in progress...

'''

import sys
import neuromllite

import lems.api as lems

from modeci_mdf.standardfunctions import mdf_functions, substitute_args

def mdf_to_neuroml(graph, save_to=None, format=None):
    
    print('Converting graph: %s to NeuroML'%(graph.id))
 
    net = neuromllite.Network(id=graph.id)
    net.notes = 'NeuroMLlite export of %s graph: %s'%(format if format else 'MDF',graph.id)
    
    model = lems.Model()
    lems_definitions = '%s_lems_definitions.xml'%graph.id
    
    for node in graph.nodes:
        print('    Node: %s'%node.id)

        node_comp_type = '%s__definition'%node.id
        node_comp = '%s__instance'%node.id
        ct = lems.ComponentType(node_comp_type)
        
        ct.add(lems.Attachments("synapses","basePointCurrentDL"))
        
        
        model.add(ct)
        comp = lems.Component(node_comp, node_comp_type)
        model.add(comp)
        
        cell = neuromllite.Cell(id=node_comp, lems_source_file=lems_definitions)
        
        net.cells.append(cell)
        
        pop = neuromllite.Population(id=node.id, 
                    size='1', 
                    component=cell.id, 
                    properties={'color':'0.2 0.2 0.2', 'radius':3})
        net.populations.append(pop)
        
        for p in node.parameters:
            ct.add(lems.Parameter(p, 'none'))
            comp.parameters[p]=node.parameters[p]
        
        for ip in node.input_ports:
            ct.add(lems.Exposure(ip.id, 'none'))
            ct.dynamics.add(lems.DerivedVariable(name=ip.id,dimension='none', value='0.11', exposure=ip.id)) 
            
        for f in node.functions:
            ct.add(lems.Exposure(f.id, 'none'))
            func_info = mdf_functions[f.function]
            expr = func_info['expression_string']
            expr2 = substitute_args(expr, f.args)
            for arg in f.args:
                expr+=';%s=%s'%(arg,f.args[arg])
            ct.dynamics.add(lems.DerivedVariable(name=f.id,dimension='none', value='%s'%(expr2), exposure=f.id)) 
            
        for op in node.output_ports:
            ct.add(lems.Exposure(op.id, 'none'))
            ct.dynamics.add(lems.DerivedVariable(name=op.id,dimension='none', value=op.value, exposure=op.id)) 
        
    if len(graph.edges)>0:

        rsDL = neuromllite.Synapse(id='rsDL', lems_source_file=lems_definitions)
        
        model.add(lems.Component(rsDL.id, 'silentSynapse'))
        net.synapses.append(rsDL)
        
    for edge in graph.edges:
        print('    Edge: %s connects %s to %s'%(edge.id,edge.sender,edge.receiver))
        '''
        ssyn_id = 'silentSyn_proj_%s'%edge.id
        silentDLin = neuromllite.Synapse(id=ssyn_id, 
                             lems_source_file=lems_definitions)
                             
        model.add(lems.Component(ssyn_id, 'silentSynapse'))
        
        net.synapses.append(silentDLin)
        net.projections.append(neuromllite.Projection(id='proj_%s'%edge.id,
                                           presynaptic=edge.sender, 
                                           postsynaptic=edge.receiver,
                                           synapse=rsDL.id,
                                           pre_synapse=silentDLin.id,
                                           type='continuousProjection',
                                           weight=1,
                                           random_connectivity=neuromllite.RandomConnectivity(probability=1)))'''
        
    # Much more todo...   
    
    
    model.export_to_file(lems_definitions)

    print('Nml net: %s'%net)
    if save_to:
        new_file = net.to_json_file(save_to)

        print('Saved NML to: %s'%save_to)

        ################################################################################
        ###   Build Simulation object & save as JSON

        simtime=1000
        dt=0.1
        sim = neuromllite.Simulation(id='Sim%s'%net.id,
                         network=new_file,
                         duration=simtime,
                         dt=dt,
                         seed= 123,
                         recordVariables={'OUTPUT':{'all':'*'}})
                         
        recordVariables = {}
        for node in graph.nodes:
            for ip in node.input_ports:
                if not ip.id in recordVariables:
                    recordVariables[ip.id] = {}
                recordVariables[ip.id][node.id] = 0
                
            for f in node.functions:
                if not f.id in recordVariables:
                    recordVariables[f.id] = {}
                recordVariables[f.id][node.id] = 0
            for op in node.output_ports:
                if not op.id in recordVariables:
                    recordVariables[op.id] = {}
                recordVariables[op.id][node.id] = 0
                
                
        sim.recordVariables = recordVariables
        sim.to_json_file()

    
    return net
    
if __name__ == "__main__":
    
    from modeci_mdf.utils import load_mdf_json, print_summary
    
    example = '../../examples/Simple.json'
    verbose = True
    if len(sys.argv)==2:
        example = sys.argv[1]
        verbose = False
        
    model = load_mdf_json(example)
    mod_graph = model.graphs[0]
    
    print('Loaded Graph:')
    print_summary(mod_graph)
    
    print('------------------')
    
    mdf_to_neuroml(mod_graph, save_to=example.replace('.json','.nmllite.json'), format=model.format)
        