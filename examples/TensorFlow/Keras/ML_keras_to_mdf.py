from keras.models import Sequential
from keras.layers import Dense

# Define the Keras model
model = Sequential()
model.add(Dense(32, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Convert to Model description format
model_json = model.to_json()

"""
The model.to_json() function converts the Keras model into a JSON string, which is the Model description format.
You can then save this string to a file, and later load it to recreate the model architecture.
"""

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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten


from modeci_mdf.mdf import *
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import simple_connect
import graph_scheduler
import random

# Create a sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

def get_weights_and_activation(layers, model):
    '''
    This function accepts the layers and return their weights and bias and activate them.
    Args:
        layers : List of strings representing name of layers in keras
        model (tf.keras.Model): model from which we want to extract layer (weights and biases) and activation function
    Return:
        params: dictionary representing weights and bias of the layers
    '''
    params = {}
    activations = []
    for layer in layers_to_extract:
        d = {}
        l = model.get_layer(layer)
        w, b = l.weights
        d['weights'], d['bias'] = w.numpy(), b.numpy()
        params[layer] = d
        activations.append(str(l.activation).split()[1])
    return params, activations


"""
