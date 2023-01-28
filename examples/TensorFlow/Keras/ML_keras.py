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

<<<<<<< HEAD
# Add layers to the model
model.add(Dense(32, input_dim=784, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()
=======

def download_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )
    return (x_train, y_train), (x_test, y_test)


def process_dataset(train, test):
    (x_train, y_train), (x_test, y_test) = train, test
    x_train_processed = x_scaler.fit_transform(x_train)
    y_train_processed = y_scaler.fit_transform(np.expand_dims(y_train, axis=-1))
    x_test_processed = x_scaler.transform(x_test)
    y_test_processed = y_scaler.transform(np.expand_dims(y_test, axis=-1))

    return (x_train_processed, y_train_processed), (x_test_processed, y_test_processed)


def define_and_train_model(x, y, epochs=100):
    model = keras.Sequential(
        [
            kl.Dense(64, activation="relu", input_shape=(13,)),
            kl.Dropout(rate=0.1),
            kl.Dense(64, activation="relu"),
            kl.Dropout(rate=0.1),
            kl.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    history = model.fit(x, y, epochs=epochs)
    return model


def evaluate_model(x, y, model):
    result = model.evaluate(x, y)
    return result


def load_model(index):
    train, test = download_dataset()
    (x_train, y_train), (x_test, y_test) = process_dataset(train, test)
    if "model" in os.listdir(os.curdir) and "model.h5" in os.listdir(
        os.path.join(os.curdir, "model")
    ):
        model = keras.models.load_model("model/model.h5")
    else:
        model = define_and_train_model(x_train, y_train)
        result = evaluate_model(x_test, y_test, model)
        print(result)
    pair = get_pair(x_test, index, model)
    return pair, model


def get_pair(x_ds, index, model):
    test_preds = model.predict(x_ds)
    x = x_ds[index : index + 10].tolist()
    y = test_preds[index : index + 10].tolist()
    return (x, y)
>>>>>>> 1d3e17da01141bc1fa9517e442456261e2d27af7
