from modeci_mdf.mdf import Model, Graph, Node, OutputPort, Parameter,Function, Condition,ConditionSet


#
#
#
#
#
#
# g=Node(id=1,parameters={})
# import inspect
# k=inspect.Parameter(kind=VAR_KEYWORD,g)
#
# # print(g.id)
# # print(g.allowed_fields)
# # for i in g.allowed_fields:
# #     print(g.i)
#
# # for k in g.allowed_fields:
# #     print(k)
# #     print(g.allowed_fields[k][1])
#     # if (self.k)!=None and type(self.k) != self.allowed_fields[k][1]:
#     #     raise TypeError(f'{k} must be {(self.allowed_fields[k][1])}')

# g=InputPort(id=id_to_port(inp))
# print(g)
from modeci_mdf.utils import load_mdf
c1=Condition(type='Always')
# print(c1)
print(c1.is_absolute)

# c=load_mdf('C:/Users/mraunak/PycharmProjects/MDF2/examples/PsyNeuLink/model_nested_comp_with_scheduler.json').graphs[0]
# g1=SubGraph(id=1,conditions={'A':2},parameters={'B':7})
# print(SubGraph.__mro__)
# print(g1)
#
# c=Graph()
# print(c.allowed_children['subgraphs'][1])

# c=load_mdf('C:/Users/mraunak/PycharmProjects/MDF2/examples/PsyNeuLink/SimpleLinear-conditional.json').graphs[0]

c=load_mdf('C:/Users/mraunak/PycharmProjects/MDF2/examples/PsyNeuLink/SimpleBranching-timing.json').graphs[0]