import collections
import yaml
import json

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

        self.allowed_fields = collections.OrderedDict([('format',('Information on verson of MDF',str)),
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

    def to_yaml(self, include_metadata=True):

        if include_metadata: self._include_metadata()

        return yaml.dump(json.loads(self.to_json()), default_flow_style=False, sort_keys=False)


class ModelGraph(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_children = collections.OrderedDict([
                                   ('nodes',('The definition of node ...',Node)),
                                   ('edges',('The definition of edge...',Edge))])

        self.allowed_fields = collections.OrderedDict([('parameters',('Dict of global parameters for the network',dict))])

        super().__init__(**kwargs)


    def get_node(self, id):
        for node in self.nodes:
            if id == node.id:
                return node


class Node(BaseWithId):

    def __init__(self, **kwargs):

        # It seems empty dictionaries cause JSON dump errors. This looks like a bug in neuromllite.  This is a work
        # around that just removes them.
        if 'parameters' in kwargs and not kwargs['parameters']:
            del kwargs['parameters']

        self.allowed_children = collections.OrderedDict([('input_ports',('Dict of ...',InputPort)),
             ('functions',('Dict of functions for the node',Function)),
             ('output_ports',('Dict of ...',OutputPort))])

        self.allowed_fields = collections.OrderedDict([('type',('Type...',str)),
                               ('parameters',('Dict of parameters for the node',dict))])

        super().__init__(**kwargs)


class Function(BaseWithId):

    def __init__(self, **kwargs):

        # It seems empty dictionaries cause JSON dump errors. This looks like a bug in neuromllite.  This is a work
        # around that just removes them.
        if 'args' in kwargs and not kwargs['args']:
            del kwargs['args']

        self.allowed_fields = collections.OrderedDict([('function',('...',str)),
                               ('args',('Dict of args...',dict))])

        super().__init__(**kwargs)


class InputPort(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([
            ('shape',('...',str)),
            ('type', ('...', str))
        ])

        super().__init__(**kwargs)


class OutputPort(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([('value',('...',str))])

        super().__init__(**kwargs)


class Edge(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([
                ('sender',('...',str)),
                ('receiver',('...',str)),
                ('sender_port',('...',str)),
                ('receiver_port',('...',str))])

        super().__init__(**kwargs)


class Condition(BaseWithId):

    def __init__(self, **kwargs):

        self.allowed_fields = collections.OrderedDict([
            ('type', ('Type...', str)),
            ('args', ('Dict of args...', dict))
        ])

        super().__init__(**kwargs)


if __name__ == "__main__":

    mod_graph0 = ModelGraph(id='Test', parameters={'speed':4})

    node  = Node(id='N0', parameters={'rate':5})

    mod_graph0.nodes.append(node)

    print(mod_graph0)
    print('------------------')
    print(mod_graph0.to_json())
    print('==================')
