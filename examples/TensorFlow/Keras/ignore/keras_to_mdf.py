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
