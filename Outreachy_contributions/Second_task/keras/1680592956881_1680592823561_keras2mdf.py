import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Input

from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.mdf import *
from modeci_mdf.utils import simple_connect

import numpy as np






def get_weights_and_activation(layers, model):

    params = {}
    activations = []
    for layer in layers:
        n = {}
        lyr = model.get_layer(layer)
        wgts, bias = lyr.weights
        n["weights"], n["bias"] = wgts.numpy(), bias.numpy()
        params[layer] = n
        activations.append(str(lyr.activation).split()[1])
    return params, activations


def init_model_with_graph(model_id, graph_id):
    mod = Model(id=model_id)
    mod_graph = Graph(id=graph_id)
    mod.graphs.append(mod_graph)
    return mod, mod_graph


def create_input_node(node_id, value):
    input_node = Node(id=node_id)
    input_node.parameters.append(
        Parameter(id=f"{node_id}_in", value=np.array(value).tolist())
    )
    input_node.output_ports.append(
        OutputPort(id=f"{node_id}_out", value=f"{node_id}_in")
    )
    return input_node


def create_dense_node(node_id, weights, bias):
    node = Node(id=node_id)
    node.input_ports.append(InputPort(id=f"{node_id}_in"))
    # Weights
    node.parameters.append(Parameter(id="wgts", value=weights))
    # bias
    node.parameters.append(Parameter(id="bias", value=bias))
    # Value Weights + bias
    node.parameters.append(
        Parameter(id="Output", value=f"({node_id}_in @ wgts) + bias")
    )

    node.output_ports.append(Parameter(id=f"{node_id}_out", value="Output"))
    return node


def create_activation_node(node_id, activation_name):
    activation = Node(id=node_id)
    activation.input_ports.append(InputPort(id=f"{node_id}_in"))

    # Functionality of relu
    if activation_name == "relu":
        # Value of relu function
        relu_ = f"({node_id}_in * ({node_id}_in > 0))"
        activation.parameters.append(Parameter(id="Output", value=relu_))

    # Functionality of sigmoid
    elif activation_name == "sigmoid":
        # args for exponential function
        args = {"variable0": "pos_in", "scale": 1, "rate": 1, "bias": 0, "offset": 0}

        # this will make x => x
        activation.parameters.append(Parameter(id="pos_in", value=f"{node_id}_in"))
        # value of e^x
        activation.functions.append(
            Function(id="exp", function="exponential", args=args)
        )
        # value of sigmoid
        activation.functions.append(Function(id="output", value="1 / (1 + exp)"))

    elif activation_name == "softmax":
        # args for exponential function
        args = {
            "variable0": f"{node_id}_in",
            "scale": 1,
            "rate": 1,
            "bias": 0,
            "offset": 0,
        }

        # exponential of each value
        activation.functions.append(
            Function(id="exp", function="exponential", args=args)
        )
        # sum of all exponentials
        activation.functions.append(Function(id="exp_sum", value="sum(exp)"))
        # normalizing results
        activation.functions.append(Function(id="Output", value="exp / exp_sum"))

    activation.output_ports.append(OutputPort(id=f"{node_id}_out", value="Output"))
    return activation