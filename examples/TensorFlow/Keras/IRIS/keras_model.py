import sys
import tensorflow as tf

# tf.__version__

from keras.layers import Dense
from keras.utils import plot_model
from keras.models import Sequential

# from keras_visualizer import visualizer
from keras import layers

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Controlling randomness, to facilitate testing/reproducibility
# based on https://keras.io/examples/keras_recipes/reproducibility_recipes/
from keras.utils import set_random_seed

set_random_seed(1234)
tf.config.experimental.enable_op_determinism()

print("Loading data")
iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Build Model
print("Building the Model")
model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Dense(32, activation=tf.nn.relu, input_dim=4, name="first_layer")
)  # hidden layers
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, name="second_layer"))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, name="third_layer"))
model.add(
    tf.keras.layers.Dense(3, activation=tf.nn.softmax, name="fourth_layer")
)  # output layer

# Compile Model
print("Compiling the model")
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# train the model
print("Training the Model")
model.fit(X_train, y_train, epochs=3)


# check if the model actually generalize the data.
print("Accuracy of our model is:")
val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss, val_acc)


# Print summary
model.summary()

# plot the model
print("Plotting the model")
plot_model(
    model, to_file="model_on_iris_plot.png", show_shapes=True, show_layer_names=True
)


# Saving model in h5 for Ploting Nuetron visual
model.save("keras_model_on_iris.keras")


# predict example for index 0
print("Predict value at index 0:")
predictions = model.predict([X_test])
print("The predicted number at index 0 is", np.argmax(predictions[0]))
