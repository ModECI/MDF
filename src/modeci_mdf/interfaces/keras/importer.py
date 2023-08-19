from inspect import Parameter
from pyclbr import Function
import random

from typing import Union, Dict, Any, Tuple, List, Callable
from xml.dom import Node

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.mdf import *
from modeci_mdf.utils import simple_connect

import numpy as np


def get_weights_and_activation(layers, model):

    params = {}
    activations = {}
    for layer in layers:
        n = {}
        lyr = model.get_layer(layer)
        if type(lyr) == Dense:
            wgts, bias = lyr.weights
            n["weights"], n["bias"] = wgts.numpy(), bias.numpy()
            params[layer] = n
            activations[layer] = str(lyr.activation).split()[1]

        elif type(lyr) == Conv1D or type(lyr) == Conv2D or type(lyr) == Conv3D:
            kernel, bias = lyr.weights
            n["weights"], n["bias"] = kernel.numpy(), bias.numpy()
            n["padding"], n["strides"] = lyr.padding, lyr.strides
            n["dilation_rate"], n["groups"] = lyr.dilation_rate, lyr.groups
            params[layer] = n
            n["activation"] = str(lyr.activation).split()[1]

        elif type(lyr) == BatchNormalization:
            gamma, beta, moving_mean, moving_variance = lyr.weights
            n["gamma"], n["beta"] = gamma.numpy(), beta.numpy()
            n["moving_mean"], n["moving_variance"] = (
                moving_mean.numpy(),
                moving_variance.numpy(),
            )
            n["epsilon"] = lyr.epsilon
            params[layer] = n

        elif type(lyr) == Dropout:
            n["rate"] = lyr.rate
            params[layer] = n
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
        Function(id="Output", function="onnx::Flatten", args=args)
    )
    flatten_node.output_ports.append(OutputPort(id=f"{node_id}_out", value="Output"))
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


def create_conv_node(
    node_id,
    kernel,
    bias,
    activation_name,
    dilations,
    groups,
    padding,
    strides,
    conv_type,
):
    node = Node(id=node_id)
    node.input_ports.append(InputPort(id=f"{node_id}_in"))

    # args for onnx::batchnormalization
    args = {
        "X": "transposed_input",
        "W": "transposed_kernel",
        "B": bias,
        "dilations": dilations,
        "group": groups,
        "strides": strides,
    }

    if conv_type == "2d":
        node.functions.append(
            Function(
                id="transposed_input",
                function="onnx:Transpose",
                args={"data": f"{node_id}_in", "perm": [0, 3, 1, 2]},
            )
        )
        node.functions.append(
            Function(
                id="transposed_kernel",
                function="onnx:Transpose",
                args={"data": kernel, "perm": [3, 2, 0, 1]},
            )
        )

    elif conv_type == "3d":
        node.functions.append(
            Function(
                id="transposed_input",
                function="onnx:Transpose",
                args={"data": f"{node_id}_in", "perm": [0, 4, 1, 2, 3]},
            )
        )
        node.functions.append(
            Function(
                id="transposed_kernel",
                function="onnx:Transpose",
                args={"data": kernel, "perm": [4, 3, 0, 1, 2]},
            )
        )

    if padding == "same":
        args["autopad"] = "SAME_UPPER"
    else:
        args["autopad"] = "VALID"

    # application of the onnx::conv function
    node.functions.append(Function(id="onnx_conv", function="onnx::Conv", args=args))

    if activation_name == "linear":
        node.functions.append(Function(id="Output", value="onnx_conv"))

    else:
        add_activation(node, activation_name, "onnx_conv")

    if conv_type == "2d":
        node.functions.append(
            Function(
                id="transposed_output",
                function="onnx:Transpose",
                args={"data": "Output", "perm": [0, 2, 3, 1]},
            )
        )
    elif conv_type == "3d":
        node.functions.append(
            Function(
                id="transposed_output",
                function="onnx:Transpose",
                args={"data": f"{node_id}_in", "perm": [0, 2, 3, 4, 1]},
            )
        )

    node.output_ports.append(OutputPort(id=f"{node_id}_out", value="transposed_output"))
    return node


def create_batch_normalization_node(
    node_id, gamma, beta, moving_mean, moving_variance, epsilon
):
    node = Node(id=node_id)
    node.input_ports.append(InputPort(id=f"{node_id}_in"))

    # args for onnx::batchnormalization
    args = {
        "X": f"{node_id}_in",
        "scale": gamma,
        "B": beta,
        "input_mean": moving_mean,
        "input_var": moving_variance,
        "epsilon": epsilon,
    }

    # application of the onnx::batchnormalization function
    node.functions.append(
        Function(id="Output", function="onnx::Batchnormalization", args=args)
    )
    node.output_ports.append(OutputPort(id=f"{node_id}_out", value="Output"))
    return node


def create_dropout_node(node_id, rate):
    node = Node(id=node_id)
    node.input_ports.append(InputPort(id=f"{node_id}_in"))

    # args for onnx::dropout function
    args = {"data": f"{node_id}_in", "ratio": rate}

    # application of onnx:dropout function
    node.functions.append(Function(id="Output", function="onnx::Dropout", args=args))
    node.output_ports.append(OutputPort(id=f"{node_id}_out", value="Output"))
    return node


def create_global_average_pool(node_id):
    node = Node(id=node_id)
    node.input_ports.append(InputPort(id=f"{node_id}_in"))

    # application of onnx::globalaveragepool
    node.functions.append(
        Function(
            id="Output", function="onnx::GlobalAveragePool", args={"X": f"{node_id}_in"}
        )
    )
    node.output_ports.append(OutputPort(id=f"{node_id}_out", value="Output"))


def add_activation(node, activation_name, str_input):
    """This function does not return anything.
    It is used to add an activation function to a dense or convolution node"""

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
            activation_name = activations[f"{layer}"]

            dense_node = create_dense_node(
                f"{layer.capitalize()}", weights, bias, activation_name
            )
            mdf_model_graph.nodes.append(dense_node)

    for i in range(len(mdf_model_graph.nodes) - 1):
        e1 = simple_connect(
            mdf_model_graph.nodes[i], mdf_model_graph.nodes[i + 1], mdf_model_graph
        )

    return mdf_model, params
