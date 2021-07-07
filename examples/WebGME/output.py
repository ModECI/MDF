from neuromllite.utils import _parse_element
from modeci_mdf.mdf import Model
from modeci_mdf.execution_engine import EvaluableGraph
import json

data = json.loads(
    """{"Simple":{"format":"ModECI MDF v0.1","generating_application":"Python modeci-mdf v0.1.2","notes":"","graphs":{"simple_example":{"notes":"","nodes":{"processing_node":{"parameters":{"slope":0.5,"logistic_gain":3,"intercept":0},"input_ports":{"input_port1":{"shape":"(1,)"}},"functions":{"logistic_1":{"function":"logistic","args":{"gain":"logistic_gain","bias":0,"variable0":"linear_1","offset":0}},"linear_1":{"function":"linear","args":{"variable0":"input_port1","intercept":"intercept","slope":"slope"}}},"output_ports":{"output_1":{"value":"logistic_1"}}},"input_node":{"parameters":{"input_level":0.5},"output_ports":{"out_port":{"value":"input_level"}}}},"edges":{"input_edge":{"sender":"input_node","receiver":"processing_node","sender_port":"out_port","receiver_port":"input_port1"}}}}}}"""
)
model = Model()
model = _parse_element(data, model)
print("----")
print(model)
graph = model.graphs[0]
egraph = EvaluableGraph(graph, False)
result = graph  # DeepForge requires the concept of interest to be available as "result"
