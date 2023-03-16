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

- An example of a simple **Spiking Neuronal Network (SNN)** can be found [here](https://github.com/ModECI/MDF/tree/main/examples/MDF/RNN#integrate-and-fire-iaf-neuron-model).
- Multiple examples of **Convolutional Neural Network (CNN)** models can be found in the [PyTorch to MDF](https://github.com/ModECI/MDF/tree/main/examples/PyTorch#pytorch-to-mdf) documentation.
- An example of a **Recurrent Neural Network (RNN)** in MDF can be found [here](https://github.com/ModECI/MDF/blob/main/examples/MDF/RNN/README.md#recurrent-neural-network-rnn-model).


## Export/import formats

### Serialization formats

Whenever a model is exchanged between different environments it will usually be a serialized form of the model which is exported/imported. Python scripts can be used to generate MDF models (e.g. [this](https://github.com/ModECI/MDF/blob/main/examples/MDF/simple.py)), but the models are saved in standardized format in either text based [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.json) or [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.yaml) formats or in binary [BSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.bson) format.

### Currently supported environments

#### PyTorch

Models can be created in [PyTorch](http://www.pytorch.org) and exported into MDF format, or MDF models can be converted to code which executes natively in PyTorch. See [here](export_format/PyTorch/PyTorch) for more details.

#### ONNX

[ONNX](https://onnx.ai) (Open Neural Network Exchange) is an important format for exchanging models between machine learning environments. It is used in the MDF function ontology, and models in ONNX format can be exported to MDF. See [here](export_format/ONNX/ONNX) for more details. Converting MDF->ONNX is best enabled currently by converting the model to PyTorch and from there to ONNX.

#### NeuroML

Examples of converting MDF to/from [NeuroML2/LEMS](https://docs.neuroml.org/Userdocs/NeuroMLv2.html) can be found [here](export_format/NeuroML/NeuroML).

#### PsyNeuLink

An outline of interactions between [PsyNeuLink](https://www.psyneulink.org) and MDF can be found [here](export_format/PsyNeuLink/PsyNeuLink).

### Planned environments to support

#### ACT-R

We have started some preliminary interactions between [ACT-R](http://act-r.psy.cmu.edu/software/) and MDF. See [here](export_format/ACT-R/ACT-R) for more details.

#### BIDS

The MDF format was first proposed following a meeting organised at Princeton in July 2019 by Russ Poldrack of the Center for Reproducible Neuroscience (CRN) at Stanford and the [Brain Imaging Data Standard (BIDS)](https://bids.neuroimaging.io/) initiative. While the prototype Python API and MDF specification have been developed independently of the BIDS initiative (which focusses on exchange of neuroimaging data), there is interest in that community to allow MDF to be used as a way to encode models of neuronal activity, which can be embedded in BIDS datasets. The [BIDS Extension Proposal Computational Models](https://docs.google.com/document/d/1NT1ERdL41oz3NibIFRyVQ2iR8xH-dKY-lRCB4eyVeRo/edit#heading=h.mqkmyp254xh6) is a potential avenue for this.


## Background to the ModECI Initiative

See [here](https://modeci.org/#aboutPage) for details about the Model Exchange and Convergence Initiative (ModECI).
