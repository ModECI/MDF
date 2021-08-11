from modeci_mdf.mdf import *
from modeci_mdf.mdf import Model
from modeci_mdf.utils import load_mdf
from modeci_mdf.scheduler import EvaluableGraph
import json
from modeci_mdf.interfaces.graphviz.importer import *

m_graph=load_mdf('C:/Users/mraunak/PycharmProjects/MDF2/examples/MDF/abc_conditions.json').graphs[0]

l=[]
# print(m_graph.conditions['node_specific'][''])

# for node in m_graph.nodes:
# # #     print(node.id)
# # #     print(m_graph.conditions['node_specific'][node.id]['type'])
# # #     print(m_graph.conditions['node_specific'][node.id]['args'])
#     for p in m_graph.conditions['node_specific'][node.id]['args']:
#         nn = m_graph.conditions['node_specific'][node.id]['args'][p]
#         print(p,nn)




mdf_to_graphviz(
m_graph, engine=engines["d"], view_on_render=True, level=3,
)



