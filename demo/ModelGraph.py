import collections


from neuromllite.BaseTypes import Base
from neuromllite.BaseTypes import BaseWithId


class EvaluableExpression(str):
    
    def __init__(self,expr):
        self.expr = expr
        
    
      
class Model(BaseWithId):

    def __init__(self, **kwargs):
        
        self.allowed_children = collections.OrderedDict([
                                   ('graphs',('The definition of top level entry ...', ModelGraph))])
                                 
                        
        super(Model, self).__init__(**kwargs)
      
class ModelGraph(BaseWithId):

    def __init__(self, **kwargs):
        
        self.allowed_children = collections.OrderedDict([
                                   ('nodes',('The definition of node ...',Node)),
                                   ('edges',('The definition of edge...',Edge))])
                                 
        self.allowed_fields = collections.OrderedDict([('parameters',('Dict of global parameters for the network',dict))])
                        
        super(ModelGraph, self).__init__(**kwargs)
        
        
class Node(BaseWithId):

    def __init__(self, **kwargs):
        
        self.allowed_children = collections.OrderedDict([('input_ports',('Dict of ...',InputPort)),
             ('functions',('Dict of functions for the node',Function)),
             ('output_ports',('Dict of ...',OutputPort))])
        
        self.allowed_fields = collections.OrderedDict([('type',('Type...',str)),
                               ('parameters',('Dict of parameters for the node',dict))])
                      
        super(Node, self).__init__(**kwargs)
        
        
class Function(BaseWithId):

    def __init__(self, **kwargs):
        
        self.allowed_fields = collections.OrderedDict([('function',('...',str)),
                               ('args',('Dict of args...',dict))])
                      
        super(Function, self).__init__(**kwargs)
        
        
class InputPort(BaseWithId):

    def __init__(self, **kwargs):
        
        self.allowed_fields = collections.OrderedDict([('shape',('...',str))])
                      
        super(InputPort, self).__init__(**kwargs)
        
        
class OutputPort(BaseWithId):

    def __init__(self, **kwargs):
        
        self.allowed_fields = collections.OrderedDict([('value',('...',str))])
                      
        super(OutputPort, self).__init__(**kwargs)
        
        
class Edge(BaseWithId):

    def __init__(self, **kwargs):
        
        self.allowed_fields = collections.OrderedDict([
                ('sender',('...',str)),
                ('receiver',('...',str)),
                ('sender_port',('...',str)),
                ('receiver_port',('...',str))])
                      
        super(Edge, self).__init__(**kwargs)
          
  
if __name__ == "__main__":
    
    mod_graph0 = ModelGraph(id='Test', parameters={'speed':4})
    
    node  = Node(id='N0', parameters={'rate':5})
    
    mod_graph0.nodes.append(node)
        
    print(mod_graph0)
    print('------------------')
    print(mod_graph0.to_json())
    print('==================')
    
    mod = Model(id='MyModel')
    mod_graph = ModelGraph(id='rl_ddm_model')
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
    new_file = mod.to_json_file('Model.json')