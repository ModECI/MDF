'''
Simple export of MDF to NeuroML(2/lite) & LEMS... 

Work in progress...

'''

import sys
import neuromllite

import lems.api as lems

def mdf_to_neuroml(graph, save_to=None):
    
    print('Converting graph: %s to NeuroML'%(graph.id))
 
    net = neuromllite.Network(id=graph.id)
    net.notes = 'NeuroMLlite export of MDF graph: %s'%graph.id
    
    model = lems.Model()
    lems_definitions = '%s_lems_definitions.xml'%graph.id
    
    for node in graph.nodes:
        print('    Node: %s'%node.id)

        node_comp_type = '%s__definition'%node.id
        node_comp = '%s__instance'%node.id
        ct = lems.ComponentType(node_comp_type)
        
        ct.add(lems.Exposure('OUTPUT', 'none'))
        ct.dynamics.add(lems.StateVariable('OUTPUT','none', 'OUTPUT')) 

        model.add(ct)
        model.add(lems.Component(node_comp, node_comp_type))
        
        cell = neuromllite.Cell(id=node_comp, lems_source_file=lems_definitions)
        
        net.cells.append(cell)
        
        pop = neuromllite.Population(id=node.id, 
                    size='1', 
                    component=cell.id, 
                    properties={'color':'0.2 0.2 0.2', 'radius':3})
        net.populations.append(pop)
        
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

        sim.to_json_file()

    
    return net
    
if __name__ == "__main__":
    
    from modeci_mdf.utils import load_mdf_json, print_summary
    
    example = '../../examples/Simple.json'
    verbose = True
    if len(sys.argv)==2:
        example = sys.argv[1]
        verbose = False
        
    mod_graph = load_mdf_json(example).graphs[0]
    
    print('Loaded Graph:')
    print_summary(mod_graph)
    
    print('------------------')
    
    mdf_to_neuroml(mod_graph, save_to=example.replace('.json','.nmllite.json'))
        