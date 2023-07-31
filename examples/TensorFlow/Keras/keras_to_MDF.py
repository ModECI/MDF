import numpy as np
import tensorflow as tf

from modeci_mdf.interfaces.keras import keras_to_mdf
from modelspec.utils import _val_info
from modeci_mdf.execution_engine import EvaluableGraph

# load the keras model
model = tf.keras.models.load_model("kr_N_model.h5")

# get one of the test images from the mnnist test dataset
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)
one_x_test = x_test[0, :, :]

# reshape the dimension of the test image to 2-D image
one_x_test = one_x_test.reshape(1, 28, 28)

# get the output of predicting with the keras model
output = model.predict(one_x_test)

# Convert the Keras model to MDF
mdf_model, params_dict = keras_to_mdf(model=model, args=one_x_test)

# Get mdf graph
mdf_graph = mdf_model.graphs[0]

# visualize mdf graph-image
mdf_model.to_graph_image(
    engine="dot",
    output_format="png",
    view_on_render=False,
    level=1,
    filename_root="keras_to_MDF",
    is_horizontal=True,
    solid_color=True,
)
from IPython.display import Image

Image(filename="Keras_to_MDF.png")

# Evaluate the model via the MDF scheduler
eg = EvaluableGraph(graph=mdf_graph, verbose=False)
eg.evaluate()
output_mdf = eg.output_enodes[0].get_output()
print("Evaluated the graph in Keras, output: %s" % (_val_info(output_mdf)))

# Assert that the results are the same for Keras and MDF
assert np.allclose(
    output,
    output_mdf,
)
print("Passed all comparison tests!")
