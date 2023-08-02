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


def create_input_node(node_id, value):  # , reshape=False):
    # if reshape == True:
    #     value = np.array(value).flatten()
    # else:
    #     value = np.array(value)
    input_node = Node(id=node_id)
    input_node.parameters.append(Parameter(id=f"{node_id}_in", value=value))
    input_node.output_ports.append(
        OutputPort(id=f"{node_id}_out", value=f"{node_id}_in")
    )
    return input_node


def create_flatten_node(node_id):
    flatten_node = Node(id=node_id)
    flatten_node.input_ports.append(InputPort(id=f"{node_id}_in"))

    # args for onnx::reshape function
    args = {"data": f"{node_id}_in", "shape": np.array([-1], dtype=np.int64)}

    # application of the onnx::reshape function to the input
    flatten_node.functions.append(
        Function(id="onnx_Reshape", function="onnx::Reshape", args=args)
    )
    flatten_node.output_ports.append(
        OutputPort(id=f"{node_id}_out", value="onnx_Reshape")
    )
    return flatten_node


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

    node.output_ports.append(OutputPort(id=f"{node_id}_out", value="Output"))
    return node


def create_activation_node(node_id, activation_name):
    activation = Node(id=node_id)
    activation.input_ports.append(InputPort(id=f"{node_id}_in"))

    # Functionality of relu
    if activation_name == "relu":
        # Value of relu function
        relu_ = f"({node_id}_in * ({node_id}_in > 0 ))"
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
        activation.functions.append(Function(id="Output", value="1 / (1 + exp)"))

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

    # create mdf model and graph
    mdf_model, mdf_model_graph = init_model_with_graph(
        f"{model.name}".capitalize(), f"{model.name}_Graph".capitalize()
    )

    # create the input node
    input_node = create_input_node("Input_0", args)
    mdf_model_graph.nodes.append(input_node)

    # create other nodes needed for mdf graph using the type of layers from the keras model
    # get layers in the keras model
    layers = []
    for layer in model.layers:
        layers.append(layer.name)

    # get the parameters and activation in each dense layer
    params, activations = get_weights_and_activation(layers, model)

    node_count = 1
    for layer in layers:
        if layer == "flatten":
            # input_node = create_input_node(f"{layer.capitalize()}_{node_count}", args, reshape=True)
            flatten_node = create_flatten_node(f"{layer.capitalize()}_{node_count}")
            mdf_model_graph.nodes.append(flatten_node)
            node_count += 1

        elif "dense" in layer:
            weights = params[f"{layer}"]["weights"]
            bias = params[f"{layer}"]["bias"]
            dense_node = create_dense_node(
                f"{layer[:5].capitalize()}_{node_count}", weights, bias
            )
            mdf_model_graph.nodes.append(dense_node)
            node_count += 1

            activation = str(model.get_layer(layer).activation).split()[1]
            if activation != "linear":
                activation_node = create_activation_node(
                    f"{activation.capitalize()}_{node_count}", activation
                )
                mdf_model_graph.nodes.append(activation_node)
                node_count += 1

    for i in range(len(mdf_model_graph.nodes) - 1):
        e1 = simple_connect(
            mdf_model_graph.nodes[i], mdf_model_graph.nodes[i + 1], mdf_model_graph
        )

    return mdf_model, params
