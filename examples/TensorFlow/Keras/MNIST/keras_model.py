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

# Controlling randomness, to facilitate testing/reproducibility
# based on https://keras.io/examples/keras_recipes/reproducibility_recipes/
from keras.utils import set_random_seed

set_random_seed(1234)
tf.config.experimental.enable_op_determinism()

print("Loading data")
mnist = tf.keras.datasets.mnist  # 28*28 images of handwritten digits (0-9)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data Normalizations ( scalling numbers between 0 and 1)
print("Normalizing data")
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build Model
print("Building the Model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  # input layer

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer

# Compile Model
print("Compiling the model")
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# train the model
print("Training the Model")
model.fit(x_train, y_train, epochs=3)

# loadthe model above
# new_model = tf.keras.models.load_model("num_reader.model")


# check if the model actually generalize the data.
print("Accuracy of our model is:")
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Print summary
model.summary()

# plot the model
print("Plotting the model")
plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)


# Saving model in h5 for Ploting Nuetron visual
model.save("kr_N_model.keras")


# predict example for index 0
print("Predict value at index 0:")
predictions = model.predict([x_test])
print("The predicted number at index 0 is", np.argmax(predictions[0]))

# print the actual value at that index
print("The actually number at index zero is (see the image below):")
plt.imshow(x_test[0])

if not "-nogui" in sys.argv:
    plt.show()
