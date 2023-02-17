#!/usr/bin/env python

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten


from modeci_mdf.mdf import *
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import simple_connect
import graph_scheduler
import random


# In[2]:


new_model = tf.keras.models.load_model("kr_N_model.h5")
for i in new_model.layers:
    print(i.name)


# In[3]:


def get_weights_and_activation(layers, model):

    params = {}
    activations = []
    for layer in layers_to_extract:
        d = {}
        l = model.get_layer(layer)
        w, b = l.weights
        d["weights"], d["bias"] = w.numpy(), b.numpy()
        params[layer] = d
        activations.append(str(l.activation).split()[1])
    return params, activations


# In[4]:


# selective layers which will be used in MDF Model
layers_to_extract = ["dense", "dense_1", "dense_2"]

# Calling the function to get weights and activation of layer
params, activations = get_weights_and_activation(layers_to_extract, new_model)


# In[5]:


# print(params)


# In[6]:


def init_model_with_graph(model_id, graph_id):
    mod = Model(id=model_id)
    mod_graph = Graph(id=graph_id)
    mod.graphs.append(mod_graph)
    return mod, mod_graph


# In[7]:


def create_input_node(node_id, value):
    input_node = Node(id=node_id)
    input_node.parameters.append(
        Parameter(id=f"{node_id}_in", value=np.array(x).tolist())
    )
    input_node.output_ports.append(
        OutputPort(id=f"{node_id}_out", value=f"{node_id}_in")
    )
    return input_node


# In[ ]:


# In[8]:


def create_dense_node(node_id, weights, bias):
    node = Node(id=node_id)
    node.input_ports.append(InputPort(id=f"{node_id}_in"))
    # W
    node.parameters.append(Parameter(id="w", value=weights))
    # b
    node.parameters.append(Parameter(id="b", value=bias))
    # XW + b
    node.parameters.append(Parameter(id="op", value=f"({node_id}_in @ w) + b"))

    node.output_ports.append(Parameter(id=f"{node_id}_out", value="op"))
    return node


# In[9]:


def create_activation_node(node_id, activation_name):
    activation = Node(id=node_id)
    activation.input_ports.append(InputPort(id=f"{node_id}_in"))

    # Functionality of relu
    if activation_name == "relu":
        # Value of relu function
        relu_fn = f"({node_id}_in > 0 ) * {node_id}_in"
        activation.parameters.append(Parameter(id="op", value=relu_fn))

    # Functionality of sigmoid
    elif activation_name == "sigmoid":
        # args for exponential function
        args = {"variable0": "neg_in", "scale": 1, "rate": 1, "bias": 0, "offset": 0}

        # this will make x => -x
        activation.parameters.append(Parameter(id="neg_in", value=f"-{node_id}_in"))
        # value of e^-x
        activation.functions.append(
            Function(id="exp", function="exponential", args=args)
        )
        # value of sigmoid
        activation.functions.append(Function(id="op", value="1 / (1 + exp)"))

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
        activation.functions.append(Function(id="op", value="exp / exp_sum"))

    activation.output_ports.append(OutputPort(id=f"{node_id}_out", value="op"))
    return activation


# In[ ]:


# In[10]:
