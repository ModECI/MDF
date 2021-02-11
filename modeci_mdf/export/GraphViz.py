'''
Simple export of MDF to GraphViz for generating graphics

Work in progress...

'''

import sys
import neuromllite

from graphviz import Digraph

from modeci_mdf.standardfunctions import mdf_functions, substitute_args

engines = {'d':'dot',
           'c':'circo',
           'n':'neato',
           't':'twopi',
           'f':'fdp',
           's':'sfdp',
           'p':'patchwork'}

def mdf_to_graphviz(mdf_graph, engine='dot', output_format=None, view_on_render=False):

    DEFAULT_POP_SHAPE = 'ellipse'
    DEFAULT_ARROW_SHAPE = 'empty'

    print('Converting MDF graph: %s to graphviz'%(mdf_graph.id))

    graph = Digraph(mdf_graph.id, filename='%s.gv'%mdf_graph.id, engine=engine, format=output_format)

    for node in mdf_graph.nodes:
        print('    Node: %s'%node.id)
        graph.attr('node', color='#444444', style='', fontcolor = '#444444')
        graph.node(node.id, label=node.id)

        for p in node.parameters:
            pass

        for ip in node.input_ports:
            pass

        for f in node.functions:
            pass

        for op in node.output_ports:
            pass



    for edge in mdf_graph.edges:
        print('    Edge: %s connects %s to %s'%(edge.id,edge.sender,edge.receiver))

        label = '%s'%edge.id
        graph.edge(edge.sender, edge.receiver, arrowhead=DEFAULT_ARROW_SHAPE, label=label)

    if view_on_render:
        graph.view()
    else:
        graph.render()


if __name__ == "__main__":

    from modeci_mdf.utils import load_mdf_json, print_summary

    example = '../../examples/Simple.json'
    verbose = True

    if len(sys.argv)>=2:
        example = sys.argv[1]
        verbose = False

    model = load_mdf_json(example)
    mod_graph = model.graphs[0]

    print('Loaded Graph:')
    print_summary(mod_graph)

    print('------------------')
    #nmllite_file = example.replace('.json','.nmllite.json')
    mdf_to_graphviz(mod_graph, engine=engines['c'],view_on_render=True)
