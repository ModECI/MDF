# Benchmarking Script README
This README provides an overview of the benchmarking code and how it is used for evaluating PyTorch and MDF (Model Description Format) models.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Available Models](#available-models)
- [Benchmarking Process](#benchmarking-process)
- [Result Output](#result-output)

## Introduction
The benchmarking script is intended to evaluate the performance of different PyTorch and MDF converted models. Its functionality cuts across benchmarking both PyTorch and MDF models, evaluating and providing information regarding model prediction time, prediction count, and node density.

## Getting Started
To use the benchmarking script, it is necessary to specify the script which contains the model definition and the model input data shape. A simple example is:

    **To define the model and get it ready for benchmarking, we have to define the model type in this form**


    def get_pytorch_model():
        model = model_definition_goes_here
        return model


    **To define the model data and get it ready for benchmarking, we have to define the model input data in this form**


    def get_example_input():
        x = data_to_be_predicted_goes_here
        return model


    **Where to put the benchmarking script with minimal code changes**
    In the directory where the models are defined, create a sub-directory where the PyTorch defined scripts are located and move the the benchmarking script to this directory.

## Usage
The benchmarking script can be used to evaluate specific models or all available models in a particular directory. Here are some usage examples:

### **Benchmark a Specific Model**
To benchmark a specific model, run the following command:

```
python benchmark.py [model] count [integer]
```

- `[model]`: Replace this with the name of the model you want to benchmark. Ensure that a corresponding `.py` file exist in the directory, with the necessary functions:
    - `get_example_input()`
    - `get_pytorch_model()`.

- `[integer]`: Replace this with the number of iterations for benchmarking. The larger the count, the longer the benchmark will take to completely run. Keep the count within a suitable range of between 10 to 50 for a reasonable runtime.

```
# Example:
python benchmark.py squeezenet1_1 count 10
```

### **Benchmark All Available Models**
To benchmark all available models in the directory, run the following command:

```
python benchmark.py --all
```

This will iterate through all available models, benchmarking each one. Results will be saved to a JSON file named benchmark_results.json.

On successful execution of the code below, This will display all available and suitable models in the directory.

```
python benchmark.py
```

Also, on successful command execution, the code runs a demo metric evaluation of the convolution.py script displaying results for both PyTorch and MDF models.

## Available Models
The benchmarking script automatically identifies available and viable models in the specified directory. These models should have corresponding `.py` files containing the necessary functions for benchmarking: `get_example_input()` and `get_pytorch_model()`.

## Benchmarking Process
The benchmarking code follows these steps:
1. **Model Initialization**: The code loads the selected model and its associated resources, such as model definition and suitable data sets.
2. **Data Generation**: It generates datasets with randomized values for each iteration. These datasets are used for benchmarking both PyTorch and MDF.
3. **Benchmarking Engine**: The benchmarking engine is triggered, and it measures the time taken for predictions. It computes the total prediction time, the number of predictions made and node density.
4. **Results**: The code displays the benchmarking results for both PyTorch and MDF models, including prediction time, count, and node density.

## Result Output
Benchmarking results are displayed for both PyTorch and MDF models. The output includes the following information:
- Time taken for predictions in seconds.
- Number of predictions made.
- Node density for MDF models.
- MDF-to-PyTorch ratio for time taken in case both models are benchmarked.

The results are also saved to a JSON file for reference.

This benchmarking code is a useful tool for evaluating the performance of machine learning models in PyTorch and MDF frameworks. You can use it to assess the efficiency and effectiveness of different models for your specific tasks.
