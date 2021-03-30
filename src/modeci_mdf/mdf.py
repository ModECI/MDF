import collections

'''
    Defines the structure of ModECI MDF - Work in progress!!!
'''


# Currently based on elements of NeuroMLlite: https://github.com/NeuroML/NeuroMLlite/tree/master/neuromllite
#  Try: pip install neuromllite
from neuromllite.BaseTypes import Base
from neuromllite.BaseTypes import BaseWithId



class Model(BaseWithId):

    _definition = 'The top level Model containing a number of _Graph_s'

    def __init__(self, **kwargs):

        self.allowed_children = collections.OrderedDict([
                                   ('graphs',('The list of _Graph_s in this Model', Graph))])

        self.allowed_fields = collections.OrderedDict([('format',('Information on the version of MDF used in this file',str)),
                                  ('generating_application',('Information on what application generated/saved this file',str))])

        super().__init__(**kwargs)

    def _include_metadata(self):

        from modeci_mdf import MODECI_MDF_VERSION
        from modeci_mdf import __version__
        self.format = 'ModECI MDF v%s' % MODECI_MDF_VERSION
        self.generating_application = 'Python modeci-mdf v%s' % __version__


    # Overrides BaseWithId.to_json_file
    def to_json_file(self, filename, include_metadata=True):

        if include_metadata: self._include_metadata()

        new_file = super().to_json_file(filename)

    # Overrides BaseWithId.to_yaml_file
    def to_yaml_file(self, filename, include_metadata=True):

        if include_metadata: self._include_metadata()

        new_file = super().to_yaml_file(filename)


class Graph(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_children = collections.OrderedDict([
                                   ('nodes',('The _Node_s present in the Graph',Node)),
                                   ('edges',('The _Edge_s between _Node_s in the Graph',Edge))])

        self.allowed_fields = collections.OrderedDict([('parameters',('Dict of global parameters for the Graph',dict))])

        super().__init__(**kwargs)


    def get_node(self, id):
        for node in self.nodes:
            if id == node.id:
                return node


class Node(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_children = collections.OrderedDict([('input_ports',('The _InputPort_s into the Node',InputPort)),
             ('functions',('The _Function_s for the Node',Function)),
             ('output_ports',('The _OutputPort_s into the Node',OutputPort))])

        self.allowed_fields = collections.OrderedDict([('type',('Type...',str)),
                               ('parameters',('Dict of parameters for the Node',dict))])

        super().__init__(**kwargs)


class Function(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([('function',('Which of the in-build MDF functions (linear etc.) this uses',str)),
                               ('args',('Dictionary of arguments for the Function',dict))])

        super().__init__(**kwargs)


class InputPort(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([('shape',('The shape of the variable (limited support so far...)',str))])

        super().__init__(**kwargs)


class OutputPort(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([('value',('The value of the OutputPort in terms of the _InputPort_ and _Function_ values',str))])

        super().__init__(**kwargs)


class Edge(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([
                ('sender',('The _Node_ which is the source of the Edge',str)),
                ('receiver',('The _Node_ which is the target of the Edge',str)),
                ('sender_port',('The _OutputPort_ on the sender _Node_',str)),
                ('receiver_port',('The _InputPort_ on the sender _Node_',str))])

        super().__init__(**kwargs)


if __name__ == "__main__":

    mod_graph0 = Graph(id='Test', parameters={'speed':4})

    node  = Node(id='N0', parameters={'rate':5})

    mod_graph0.nodes.append(node)

    print(mod_graph0)
    print('------------------')
    print(mod_graph0.to_json())
    print('==================')
