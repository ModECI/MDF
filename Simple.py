import collections

'''
    Example of ModECI MDF - Work in progress!!!
'''

from MDF import *


if __name__ == "__main__":

    mod = Model(id='Simple')
    mod_graph = ModelGraph(id='simple_example')
    mod.graphs.append(mod_graph)

    input_node  = Node(id='input_node', parameters={'input_level':0.0})
    op1 = OutputPort(id='out_port')
    op1.value = 'input_level'
    input_node.output_ports.append(op1)
    mod_graph.nodes.append(input_node)


    processing_node = Node(id='processing_node')
    mod_graph.nodes.append(processing_node)

    processing_node.parameters = {'logistic_gain':3, 'slope':0.5}
    ip1 = InputPort(id='input_port1', shape='(1,)')
    processing_node.input_ports.append(ip1)

    f1 = Function(id='logistic_1', function='logistic', args={'variable1':ip1.id,'gain':'logistic_gain'})
    processing_node.functions.append(f1)
    f2 = Function(id='linear_1', function='linear', args={'variable1':f1.id,'slope':'slope'})
    processing_node.functions.append(f2)
    processing_node.output_ports.append(OutputPort(id='output_1', value='linear_1'))

    e1 = Edge(id="input_edge",
              sender=input_node.id,
              sender_port=op1.id,
              receiver=processing_node.id,
              receiver_port=ip1.id)


    mod_graph.edges.append(e1)


    print(mod)

    print('------------------')
    print(mod.to_json())
    new_file = mod.to_json_file('%s.json'%mod.id)

