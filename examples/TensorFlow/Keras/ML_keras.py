import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kl
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import numpy as np


x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()


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
