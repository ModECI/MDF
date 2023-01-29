# Keras to/from MDF Converter

## Keras
[Keras](https://keras.io/) is a high-level, deep learning API developed by Google for implementing neural networks. It is written in Python and is used to make the implementation of neural networks easy. It also supports multiple backend neural network computation.

## Prerequisites
- Keras installed
- A simple feedforward model trained in Keras

## Steps
1. First, you will need to install the keras.js library by running .

```Python
pip install keras.js
```

2. Next, you will need to save your trained Keras model to a file by running model.save("model.h5").

3. Now, you can convert the saved model to a model description format by running the following command:

```Python
from keras.js import model_to_json
json_string = model_to_json(model)

```
This will convert the model to a JSON format.


4. To convert to YAML format use this command

```Python
from keras.js import model_to_yaml
yaml_string = model_to_yaml(model)
```


5. Save the json_string or yaml_string to a file

That's it! Your Keras simple feedforward model is now in a MDF format and can be loaded into other applications or environments.

Note: This guide is for simple feedforward model, for other complex architecture model, you may need to use other library or different approach.
