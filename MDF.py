import collections

'''
    Defines the structure of ModECI MDF - Work in progress!!!
'''


# Currently based on elements of NeuroMLlite: https://github.com/NeuroML/NeuroMLlite/tree/master/neuromllite
#  Try: pip install neuromllite
from neuromllite.BaseTypes import Base
from neuromllite.BaseTypes import BaseWithId



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

