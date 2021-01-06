
'''
    Example of ModECI MDF - Work in progress!!!
'''

from modeci_mdf.MDF import *

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
    
def print_summary(graph):
    print('Graph %s with %i nodes and %s edges\n'%(graph.id, len(graph.nodes), len(graph.edges)))
    for node in graph.nodes:
        print('%s'%node)
    for edge in graph.edges:
        print('%s'%edge)
        
        

    
def load_mdf_json(filename):
    """
    Load an MDF JSON file
    """
    
    from neuromllite.utils import load_json, _parse_element
    
    data = load_json(filename)
        
    print("Loaded graph from %s"%filename)
    
    model = Model()
    model = _parse_element(data, model)
    
    return model