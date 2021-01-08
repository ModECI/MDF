import collections

'''
    Example of ModECI MDF - Work in progress!!!
'''

from modeci_mdf.MDF import *

from modeci_mdf.utils import create_node, simple_connect, print_summary
    
if __name__ == "__main__":

    mod = Model(id='ABCD')
    mod_graph = ModelGraph(id='abcd_example')
    mod.graphs.append(mod_graph)

    input_node  = Node(id='input0', parameters={'input_level':0.0})
    op1 = OutputPort(id='out_port')
    op1.value = 'input_level'
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)
    
    print(input_node)
    print(input_node.output_ports)

    a = create_node('A', mod_graph)
    
    e1 = simple_connect(input_node, a, mod_graph)
    
    b = create_node('B', mod_graph)
    simple_connect(a, b, mod_graph)
    
    c = create_node('C', mod_graph)
    simple_connect(b, c, mod_graph)
    
    d = create_node('D', mod_graph)
    simple_connect(c, d, mod_graph)

    print(mod)

    print('------------------')
    #print(mod.to_json())
    new_file = mod.to_json_file('%s.json'%mod.id)
    
    print_summary(mod_graph)

