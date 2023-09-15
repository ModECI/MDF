# Keras to MDF example: IRIS dataset

**For more details on Keras and the current state of the Keras->MDF mapping see the [MNIST example](../MNIST).**

This model uses the [IRIS dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) in the [trained Keras model](keras_model.py) and the MDF equivalent.

### Summarize Model

Below is the summary image of the trained Keras model. We can clearly see the output shape and number of weights in each layer:

![summary](summary.png)


### Keras Model

Visualization of the model from Keras:

<p align="center"><img src="model_on_iris_plot.png"/></p>
<br>

### MDF Model

Graphviz is used to generate visualization for the MDF graph. Below is the visualization of the MDF graph after converting the keras model to MDF.

![keras_to_MDF](keras_to_MDF.1.png)

More detailed graphical representation of the MDF:

<p align="center"><img src="keras_to_MDF.png" width="400"/></p>

##### Netron
Below is the visualization of this model using netron

![keras-model-to-netron](layers_netron.png)
