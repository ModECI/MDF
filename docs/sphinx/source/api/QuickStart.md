# Quick Start Guide to MDF

This is a quick guide to the various parts of the ModECI Model Description Format (MDF) specification, API and examples.

## Specification of MDF language

The specification for the language, including the core types <a href="Specification.html#graph">Graph</a>, <a href="Specification.html#node">Node</a>, <a href="Specification.html#edge">Edge</a> etc. is available <a href="Specification.html">here</a>.

## Installation of Python API

There is a prototype implementation of an API (Application Programming Interface) in Python which can be used to build models in the MDF format, as well as save (serialize) the models in JSON, YAML and other formats. It also has an [Execution Engine](https://mdf.readthedocs.io/en/latest/api/_autosummary/modeci_mdf.execution_engine.html#module-modeci_mdf.execution_engine) which can be used to execute/evaluate the models.

Use **pip** to install the latest version of MDF (plus dependencies) from [PyPI](https://pypi.org/project/modeci-mdf):
```
pip install modeci_mdf
```

More details, and importantly, how to set up a [virtual environment](https://virtualenv.pypa.io/en/latest/) for the package, can be found [here](Installation).

## Examples of MDF

### Simple examples

Some basic examples of models in MDF format which illustrate how a model can be 1) created using the Python API, 2) saved to JSON and YAML, 3) exported to graphical form and 4) executed to evaluate all parameters, can be found [here](export_format/MDF/MDF).

### More complex examples

- An example of a simple **spiking neuronal network** can be found [here](https://github.com/ModECI/MDF/tree/main/examples/MDF/RNN#integrate-and-fire-iaf-neuron-model).

## Export/import formats

### Serialization formats

Whenever a model is exchanged between different environments...

While Python scripts can be used to generate MDF models (e.g. [this](https://github.com/ModECI/MDF/blob/main/examples/MDF/simple.py)) the models are saved in standardized format in text based [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.json) or [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.yaml) formats or in binary [BSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.bson) format.

### Current supported environments

#### PyTorch

More...

#### ONNX



#### NeuroML
#### PsyNeuLink

### Planned supported environments

#### BIDS

#### ACT-R




## Background to the ModECI Initiative

See [here](https://modeci.org/#aboutPage) for details about the Model Exchange and Convergence Initiative (ModECI).
