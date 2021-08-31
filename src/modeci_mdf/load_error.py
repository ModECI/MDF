from modeci_mdf.mdf import Model, Graph, Node, OutputPort, Parameter,Function, Condition,ConditionSet
#
#
#
# from neuromllite.BaseTypes import Base
# from neuromllite.BaseTypes import BaseWithId
from neuromllite import EvaluableExpression
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

p=Parameter(id=1,default_initial_value=1,value=1)
print(p.__dict__)
print(type(p.value))
print(type(EvaluableExpression(1)))
