# Keras to/from MDF Converter


## Keras

[Keras](https://keras.io/) is a high-level API for building and training deep learning models, built on top of TensorFlow, which provides a low-level and flexible platform for numerical computation and data flow programming. We are going to build a simple machine learning Model using Keras and MNIST data.

## Project objective
The objective of the above model is to classify handwritten digits in the MNIST dataset.
The model takes as input an image of a handwritten digit and outputs a label corresponding to the digit (0-9). The model uses a neural network architecture with a series of dense layers to process the input image and produce a prediction.

The objective is to train the model to accurately predict the correct label for each image in the MNIST dataset. This is achieved by optimizing the model's parameters (weights and biases) using a supervised learning approach, where the model is trained on a labeled training set and evaluated on a separate test set.

The ultimate goal is to produce a model that generalizes well to new, unseen images, and can accurately classify digits with high accuracy. The model is trained using a categorical cross-entropy loss function and the accuracy metric is used to evaluate the performance of the model on the test set.

### MNIST Data
The MNIST (Modified National Institute of Standards and Technology) dataset is a widely used dataset for machine learning and computer vision. It consists of a large collection of grayscale images of handwritten digits (0-9), along with their corresponding labels (the digit in each image).Each image in the MNIST dataset is 28x28 pixels in size, and is represented as a 2-dimensional array of pixel intensities. The reason why we use MNISt to build this model its because of the following:

###### Availability:
The MNIST dataset is widely available and easy to download you can also load it online using the Commands below
```Python

```
###### Simple and well-defined:
The MNIST dataset consists of simple grayscale images of handwritten digits, and the task of classifying the digit in each image is well-defined and straightforward.
The grayscale images look like the one below

![](https://raw.githubusercontent.com/ModECI/MDF/test_keras/examples/TensorFlow/Keras/3.jpeg)

######  Low dimensional:
The MNIST images are 28x28 pixels in size, which is a relatively low dimensional input space compared to more complex image datasets. This makes it easier to train machine learning models and to visualize the results.

######  Clean and preprocessed:
The MNIST dataset has already been preprocessed and cleaned, which saves time and reduces the amount of preprocessing required.


## Prerequisites

##### Get Keras installed
Open your command prompt and enter the commands below
```Python
    pip install tensorflow
```
Then enter

```Python
    pip install keras
```

## Steps
1. First, you will need to install the keras.js library by running .

```Python
pip install keras
```

2. Next, you will need to save your trained Keras model to a file by running model.save("model.h5").

3. Now, you can convert the saved model to a model description format by running the following command:

```Python
from keras import model_to_json
json_string = model_to_json(model)

```
   This will convert the model to a JSON format.


4. To convert to YAML format use this command

```Python
from keras import model_to_yaml
yaml_string = model_to_yaml(model)
```


5. Save the json_string or yaml_string to a file

That's it! Your Keras simple feedforward model is now in a MDF format and can be loaded into other applications or environments.

Note: This guide is for simple feedforward model, for other complex architecture model, you may need to use other library or different approach.
