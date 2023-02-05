import matplotlib.pyplot as plt
import numpy as np

from modeci_mdf.mdf import *
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import simple_connect
import graph_scheduler
import random

data_index = 15
(x, y), model = load_model(data_index)


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


# Extract layers
layers_to_extract = ["dense", "dense_1", "dense_2"]

# Get weights and activation of layers
params, activations = get_weights_and_activation(layers_to_extract, model)


def init_model_with_graph(model_id, graph_id):

    mod = Model(id=model_id)
    mod_graph = Graph(id=graph_id)
    mod.graphs.append(mod_graph)
    return mod, mod_graph


def create_input_node(node_id, value):
    input_node = Node(id=node_id)
    input_node.parameters.append(
        Parameter(id=f"{node_id}_in", value=np.array(x).tolist())
    )
    input_node.output_ports.append(
        OutputPort(id=f"{node_id}_out", value=f"{node_id}_in")
    )
    return input_node


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


# creating initial MDf model
mod, mod_graph = init_model_with_graph("keras-model", "main")

# appending input node to model
mod_graph.nodes.append(create_input_node("x", np.array(x[0]).tolist()))

# looping on layers we selected to extract
for i, layer in enumerate(layers_to_extract):
    # current last layer in MDF model
    prev = mod_graph.nodes[-1]
    # Dense layer
    dense = create_dense_node(layer, params[layer]["weights"], params[layer]["bias"])
    mod_graph.nodes.append(dense)
    # Activation layer
    activation_id = layer + "_" + activations[i]
    activation = create_activation_node(activation_id, activations[i])
    mod_graph.nodes.append(activation)
    # Edges connecting Nodes together
    simple_connect(prev, dense, mod_graph)
    simple_connect(dense, activation, mod_graph)

# Saving Model to JSON file
mod.to_json_file("keras-model.json")

# Evaluating Model graph
eg = EvaluableGraph(mod_graph, verbose=False)
eg.evaluate()

# storing final calculation of graph
pred = eg.enodes["dense_2_sigmoid"].evaluable_outputs["dense_2_sigmoid_out"].curr_value

for i in range(10):
    p = str(float(pred[i]))
    y_ = str(y[i][0])
    count = 0
    for a, b in zip(p, y_):
        if a == b:
            count += 1
        else:
            print(p, y_)
            print("Values are Same upto {} precision points".format(count - 2))
            break


"""
# To generate a YAML file from the model and Save the model architecture to a YAML file
model_yaml = kr_model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

# Save the model weights to a HDF5 file
kr_model.save_weights("model.h5")

# To generate a Json file from the model and Save the model architecture to a json file
model_json = kr_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights to a HDF5 file
kr_model.save_weights("model.h5")


The model.to_json() function converts the Keras model into a JSON string, which is the Model description format.
You can then save this string to a file, and later load it to recreate the model architecture.


with open("model.json", "w") as json_file:
    json_file.write(model_json)


# use the model.save_weights to save the weights of the model in h5 format.

model.save_weights("model.h5")

# load the model again
from keras.models import model_from_json

# load json and create model
json_file = open("model.json")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
"""
