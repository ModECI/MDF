""" """ 
Example of ModECI MDF- A simple 3 Node graph satisfying the EveryNCalls Condition

"""

import graph_scheduler
from modeci_mdf.mdf import (
    Condition,
    ConditionSet,
    Parameter,
    Graph,
    InputPort,
    Model,
    Node,
    OutputPort,
)
from modeci_mdf.utils import print_summary, simple_connect

def main():
    mod = Model(id="everyncalls_condition")
    mod_graph = Graph(id="everyncalls_example")
    mod.graphs.append(mod_graph)

    def create_simple_node(graph, id_, sender=None):
        n = Node(id=id_)
        graph.nodes.append(n)

        ip1 = InputPort(id="input_port1", shape="(1,)")
        n.input_ports.append(ip1)

        n.output_ports.append(OutputPort(id="output_1", value=ip1.id))

        if sender is not None:
            simple_connect(sender, n, graph)

        return n  
    a = create_simple_node(mod_graph, "A", sender=None)
    a.parameters.append(Parameter(id="param_A", value="param_A + 1",default_initial_value=0)) """