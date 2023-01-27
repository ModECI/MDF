""" This network has 3 layers: an input layer with 784 neurons (input_dim=784),
a hidden layer with 32 neurons, and an output layer with 10 neurons.
The activation function used in the hidden layers is 'relu', and in the output layer is 'softmax' .
The 'Adam' optimizer is used to update the weights during training and
the loss function used is 'categorical_crossentropy'. You can also use the method model.summary()
 to display the network output, it will show you the layers and the number of neurons or parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(32, input_dim=784, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()
