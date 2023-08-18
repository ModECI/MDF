from inspect import Parameter
from pyclbr import Function
import random

from typing import Union, Dict, Any, Tuple, List, Callable
from xml.dom import Node

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
        if lyr.weights != []:
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
    input_node.parameters.append(Parameter(id=f"{node_id}_in", value=value))
    input_node.output_ports.append(
        OutputPort(id=f"{node_id}_out", value=f"{node_id}_in")
    )
    return input_node


def create_flatten_node(node_id):
    flatten_node = Node(id=node_id)
    flatten_node.input_ports.append(InputPort(id=f"{node_id}_in"))
    args = {"input": f"{node_id}_in"}
    # application of the onnx::flatten function to the input
    flatten_node.functions.append(
        Function(id="onnx_Flatten", function="onnx::Flatten", args=args)
    )
    flatten_node.output_ports.append(
        OutputPort(id=f"{node_id}_out", value="onnx_Flatten")
    )
    return flatten_node


def create_dense_node(node_id, weights, bias, activation_name):
    node = Node(id=node_id)
    node.input_ports.append(InputPort(id=f"{node_id}_in"))
    # Weights
    node.parameters.append(Parameter(id="wgts", value=weights))
    # bias
    node.parameters.append(Parameter(id="bias", value=bias))
    # Value Weights + bias
    node.parameters.append(
        Parameter(id="linear", value=f"({node_id}_in @ wgts) + bias")
    )

    if activation_name == "linear":
        node.parameters.append(Parameter(id="Output", value="linear"))

    else:
        add_activation(node, activation_name, "linear")

    node.output_ports.append(OutputPort(id=f"{node_id}_out", value="Output"))
    return node


def add_activation(node, activation_name, str_input):
    """This function does not return anything. It is used to add an activation function implementation to a dense node"""

    # Functionality of relu
    if activation_name == "relu":
        # Value of relu expression
        relu_ = str_input + "*" + "(" + str_input + ">" + "0" + ")"
        node.parameters.append(Parameter(id="Output", value=relu_))

    # Functionality of sigmoid
    elif activation_name == "sigmoid":
        # args for exponential function
        args = {"variable0": "pos_in", "scale": 1, "rate": 1, "bias": 0, "offset": 0}

        # this will make x => x
        node.parameters.append(Parameter(id="pos_in", value=str_input))
        # value of e^x
        node.functions.append(Function(id="exp", function="exponential", args=args))
        # value of sigmoid
        node.functions.append(Function(id="Output", value="1 / (1 + exp)"))

    elif activation_name == "softmax":
        # args for softmax
        args = {"input": str_input}

        # softmax function implementation
        node.functions.append(
            Function(id="softmax", function="onnx::Softmax", args=args)
        )
        node.functions.append(Function(id="Output", value="softmax"))


def keras_to_mdf(
    model: Union[Callable, tf.keras.Model, tf.Module],
    args: Union[None, np.ndarray, tf.Tensor] = None,
    # trace: bool = False,
) -> Union[Model, Graph]:
    r"""
    Convert a Keras model to an MDF model.
    Args:
        model: The model to translate into MDF.
        args: The input arguments for this model.

    Returns:
        The translated MDF model
    """

    print("About to work!!")
    # create mdf model and graph
    mdf_model, mdf_model_graph = init_model_with_graph(
        f"{model.name}".capitalize(), f"{model.name}_Graph".capitalize()
    )

    # create the input node
    input_node = create_input_node("Input", args)
    mdf_model_graph.nodes.append(input_node)

    # create other nodes needed for mdf graph using the type of layers from the keras model
    # get layers in the keras model
    layers = []
    layers_types = []
    for layer in model.layers:
        layers.append(layer.name)
        layers_types.append(type(layer))

    # get the parameters and activation in each dense layer
    params, activations = get_weights_and_activation(layers, model)

    for layer, layer_type in zip(layers, layers_types):
        if layer_type == Flatten:
            flatten_node = create_flatten_node(f"{layer.capitalize()}")
            mdf_model_graph.nodes.append(flatten_node)

        elif layer_type == Dense:
            weights = params[f"{layer}"]["weights"]
            bias = params[f"{layer}"]["bias"]
            activation_name = str(model.get_layer(layer).activation).split()[1]

            dense_node = create_dense_node(
                f"{layer.capitalize()}", weights, bias, activation_name
            )
            mdf_model_graph.nodes.append(dense_node)

    for i in range(len(mdf_model_graph.nodes) - 1):
        e1 = simple_connect(
            mdf_model_graph.nodes[i], mdf_model_graph.nodes[i + 1], mdf_model_graph
        )

    return mdf_model, params
