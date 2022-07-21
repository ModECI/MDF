from neuromllite.utils import _parse_element
from modeci_mdf.mdf import Model
from modeci_mdf.execution_engine import EvaluableGraph
import json


data = """{
        "Simple": {
            "format": "ModECI MDF v0.2",
            "generating_application": "Python modeci-mdf v0.2.1",
            "graphs": {
                "simple_example": {
                    "nodes": {
                        "input_node": {
                            "parameters": {
                                "input_level": {
                                    "value": 0.5
                                }
                            },
                            "output_ports": {
                                "out_port": {
                                    "value": "input_level"
                                }
                            }
                        },
                        "processing_node": {
                            "input_ports": {
                                "input_port1": {}
                            },
                            "parameters": {
                                "lin_slope": {
                                    "value": 0.5
                                },
                                "lin_intercept": {
                                    "value": 0
                                },
                                "log_gain": {
                                    "value": 3
                                },
                                "linear_1": {
                                    "function": "linear",
                                    "args": {
                                        "variable0": "input_port1",
                                        "slope": "lin_slope",
                                        "intercept": "lin_intercept"
                                    }
                                },
                                "logistic_1": {
                                    "function": "logistic",
                                    "args": {
                                        "variable0": "linear_1",
                                        "gain": "log_gain",
                                        "bias": 0,
                                        "offset": 0
                                    }
                                }
                            },
                            "output_ports": {
                                "output_1": {
                                    "value": "logistic_1"
                                }
                            }
                        }
                    },
                    "edges": {
                        "input_edge": {
                            "parameters": {
                                "weight": 0.55
                            },
                            "sender": "input_node",
                            "receiver": "processing_node",
                            "sender_port": "out_port",
                            "receiver_port": "input_port1"
                        }
                    }
                }
            }
        }
    }"""

model = Model.from_json(data)
print("----")
print(model)
graph = model.graphs[0]
egraph = EvaluableGraph(graph, False)

###### TODO: update WebGME import to handle post https://github.com/ModECI/MDF/issues/101 changes!
result = graph  # DeepForge requires the concept of interest to be available as "result"
