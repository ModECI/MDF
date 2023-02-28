import numpy as np
from tensorflow.keras.models import load_model
from modeci_mdf.utils import simple_connect

# import mdf
from modeci_mdf.mdf import *

# for executing the graph
from modeci_mdf.execution_engine import EvaluableGraph

from modeci_mdf.utils import simple_connect

# from Keras_2_mdf import * #contains helper functions for this model
from keras2mdf import *

# import the necessary package to use Conditions in MDF
import graph_scheduler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten


new_model = tf.keras.models.load_model("kr_N_model.h5")
for i in new_model.layers:
    print(i.name)


# selective layers which will be used in MDF Model
layers_to_extract = ["dense", "dense_1", "dense_2"]

# Call the function to get weights and activation of layer
params, activations = get_weights_and_activation(layers_to_extract, new_model)


# View Weights of the Model
# new_model.get_weights()


# weights = new_model.layers[1].get_weights()[1]
# bias = new_model.layers[0].get_weights()


mod, mod_graph = init_model_with_graph("keras_to_MDF", "Keras_to_MDF_graph")

input_node = create_input_node("input_node", [1.0])
mod_graph.nodes.append(input_node)
print(mod_graph.to_yaml())


node = create_dense_node("dense_node", "weights", "bias")
mod_graph.nodes.append(node)
print(mod_graph.to_yaml())


activation = create_activation_node("activation_node", "activation_name")
mod_graph.nodes.append(activation)
print(mod_graph.to_yaml())


#  Save the model to file
mod.to_json_file("keras_to_MDF.json")
mod.to_yaml_file("keras_to_MDF.yaml")


""""mod.to_graph_image(
        engine="dot",
        output_format="png",
        view_on_render=False,
        level=3,
        filename_root="keras_to_MDF",
        is_horizontal=True
    )

from IPython.display import Image
Image(filename="Keras_to_MDF_Example.png")
"""
