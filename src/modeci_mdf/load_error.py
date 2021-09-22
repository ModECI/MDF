# from graph_scheduler.condition import Condition
from modeci_mdf.mdf import (
    Model,
    Graph,
    Node,
    OutputPort,
    Function,
    Condition,
    ConditionSet,
    Parameter,
    Edge,
    InputPort,
)

# print(Condition)
from modeci_mdf.utils import load_mdf

c1 = Condition(type="Always")
# print(c1)
print(c1.is_absolute)

c = load_mdf(
    "C:/Users/mraunak/PycharmProjects/MDF/examples/MDF/abc_conditions.json"
).graphs[0]
for node, cond in c.conditions.node_specific.items():
    print(node)
    print(cond)
