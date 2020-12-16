import collections

'''
    Example of ModECI MDF - Work in progress!!!
'''

from MDF import *

def create_node(node_id, graph):
    
    a = Node(id=node_id)
    graph.nodes.append(a)

    a.parameters = {'logistic_gain':3, 'slope':0.5}
    ip1 = InputPort(id='input_port1', shape='(1,)')
    a.input_ports.append(ip1)

    f1 = Function(id='logistic_1', function='logistic', args={'variable1':ip1.id,'gain':'logistic_gain'})
    a.functions.append(f1)
    f2 = Function(id='linear_1', function='linear', args={'variable1':f1.id,'slope':'slope'})
    a.functions.append(f2)
    a.output_ports.append(OutputPort(id='output_1', value='linear_1'))
    
    return a

def simple_connect(pre_node, post_node, graph):
    
    e1 = Edge(id="input_edge",
              sender=pre_node.id,
              sender_port=pre_node.output_ports[0].id,
              receiver=post_node.id,
              receiver_port=post_node.input_ports[0].id)
              
    graph.edges.append(e1)
    return e1
    

if __name__ == "__main__":

    mod = Model(id='ABCD')
    mod_graph = ModelGraph(id='abcd_example')
    mod.graphs.append(mod_graph)

    input_node  = Node(id='input0', parameters={'input_level':0.0})
    op1 = OutputPort(id='out_port')
    op1.value = 'input_level'
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)

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
    print(mod.to_json())
    new_file = mod.to_json_file('%s.json'%mod.id)

