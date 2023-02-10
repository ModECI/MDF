import tensorflow as tf

from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.models import Sequential

#from keras_visualizer import visualizer
from keras import layers


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
