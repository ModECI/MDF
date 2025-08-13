<p align="center">
  <img alt="mdf logo" width="402" src="https://raw.githubusercontent.com/ModECI/MDF/main/docs/sphinx/images/logo_light_bg.png"/>
</p>

[![Actions Status][actions-badge]][actions-link]
[![PyPI version](https://badge.fury.io/py/modeci-mdf.svg)](https://badge.fury.io/py/modeci-mdf)
[![Documentation Status][rtd-badge]][rtd-link]
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ModECI Model Description Format (MDF)
[**Click here for the full MDF documentation**](https://mdf.readthedocs.io)

**Note: MDF is still in development! See the [open issues related to the specification](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) or go [here](http://modeci.org/#contactPage) to get in contact regarding MDF.**
*The MDF format was first proposed following a meeting organised at Princeton in July 2019 by Russ Poldrack of the Center for Reproducible Neuroscience (CRN) at Stanford and the [Brain Imaging Data Standard (BIDS)](https://bids.neuroimaging.io/) initiative. For more on the previous work in this area, see [here](https://github.com/OpenSourceBrain/PsyNeuLinkShowcase/tree/master/BIDS-MDF).*


## Overview

MDF is an open source, community-supported standard and associated library of tools for expressing computational models in a form that allows them to be exchanged between diverse programming languages and execution environments. The overarching aim is to provide a common format for models across [computational neuroscience](https://en.wikipedia.org/wiki/Computational_neuroscience), [cognitive science](https://cogsci.jhu.edu/about/#:~:text=What%20Is%20Cognitive%20Science%3F,are%20realized%20in%20the%20brain.) and [machine learning.](https://www.ibm.com/topics/machine-learning)

It consists of a specification for expressing models in serialized formats (currently [JSON](https://developer.mozilla.org/en-US/docs/Glossary/JSON), [YAML](https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/) and [BSON](https://en.wikipedia.org/wiki/BSON) representations are supported, though others such as [HDF5](https://www.geeksforgeeks.org/hdf5-files-in-python/) are planned) and a set of Python tools for implementing a model described using MDF. The serialized formats can be used when importing a model into a supported target environment to execute it; and, conversely, when exporting a model built in a supported environment so that it can be re-used in other environments.

The MDF Python API can be used to create or load an MDF model for inspection and validation. It also includes a basic [execution engine](https://mdf.readthedocs.io/en/latest/api/_autosummary/modeci_mdf.execution_engine.html#module-modeci_mdf.execution_engine) for simulating models in the format. However, this is not intended to provide a efficient, general-purpose simulation environment, nor is MDF intended as a programming language. Rather, the primary purpose of the Python API is to facilitate and validate the exchange of models between existing environments that serve different communities. Accordingly, these Python tools include bi-directional support for importing to and exporting from widely-used programming environments in a range of disciplines, and for easily extending these to other environments.

## Development

The implementation and dissemination of the MDF language and associated tools is being carried out by the [Model Exchange and Convergence Initiative (ModECI)](http://modeci.org/), which has been supported by the [NSF Convergence Accelerator Program](https://www.nsf.gov/od/oia/convergence-accelerator/) (Track D: AI-Driven Innovation via Data and Model Sharing), as a publicly accessible [open-source project](https://github.com/ModECI/MDF). The initial design has been informed by a [series of workshops](http://modeci.org/#communityPage) involving developers of key software environments and other stakeholders in machine learning, cognitive science and neuroscience. Future workshops will address broadening of support to other domains in basic and applied science and technology development (e.g., population biology, medical informatics, structural and environmental monitoring, and complex systems control). Environments for which support is currently being developed include [PyTorch](https://pytorch.org), [ONNX](http://onnx.ai), [WebGME](https://webgme.org), [NeuroML](https://neuroml.org), [PsyNeuLink](http://www.psyneuln.deptcpanel.princeton.edu), and [ACT-R](http://act-r.psy.cmu.edu).

<p align="center">
  <a href="https://github.com/ModECI/MDF/blob/main/examples/README.md">
    <img alt="mdf interactions" width="400" src="https://github.com/ModECI/MDF/raw/main/examples/ModECI_MDF.png"/>
  </a>
<br/><sup><i><b>Fig 1:</b> Some of the current and planned formats which MDF will interact with. Click on the image for more information.</i></sup></p>

Successful interfacing of MDF to existing disciplinary standards (such as [ONNX](http://onnx.ai) in machine learning, and [NeuroML](https://neuroml.org) in neuroscience) as well as general-purpose simulation environments (such as [WebGME](https://webgme.org)) will permit bridging between these environments, and translation to the broader set of environments supported by those standards (such as [Tensorflow](https://www.tensorflow.org) & [Keras](https://keras.io) in the case of ONNX, and [The Virtual Brain](https://www.thevirtualbrain.org) and [SONATA](https://github.com/AllenInstitute/sonata) in the case of NeuroML). Initial investigations have also taken place, in collaboration with projects in the NSF Accelerator Track C (Quantum Technology), to use MDF for facilitating the implementation of computational models on [quantum hardware](https://github.com/ModECI/MDF/blob/readme_update/examples/Quantum).


### The core elements of the MDF standard

**[Models](https://mdf.readthedocs.io/en/latest/api/Specification.html#model)** The highest level construct in MDF is a model that consists of one or more **graphs** and model attributes. The former describe the operational features of the model (its structure and execution), while the latter provide additional information (metadata) useful for executing, evaluating, testing or visualizing it.

**[Graphs](https://mdf.readthedocs.io/en/latest/api/Specification.html#graph)** A graph specifies the structure and process flow of a **model**. The most fundamental element of a graph is a **node**, which specifies some unit of computation in terms of its **parameters** and **functions**. Nodes are connected to other nodes via directed **edges**, which, in the absence of additional **conditions**, define the computational flow of the model.

**[Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node)** These define the core elements of computation in a **graph**, that receive and transmit information via their **input and output ports**. In general, ports represent points of contact between a **node** and the **edges**that connect it to other nodes.

**[Output Ports](https://mdf.readthedocs.io/en/latest/api/Specification.html#outputport)** An output port is the starting point of the data transmission process. After processing the information in a **node**, an output port is used to begin the transmission of information to the next **node** through **edges**.

**[Edges](https://mdf.readthedocs.io/en/latest/api/Specification.html#edge)** These transmit information from the **output port** of one **node** to the **input port** of another, collectively defining a **graph’s** topography. Edges may contain weights that can operate on the information they carry.

**[Input Ports](https://mdf.readthedocs.io/en/latest/api/Specification.html#inputport)** An input port is the endpoint of the data transmission process. It receives the information transmitted through an **edge** and inputs it to the next **node** for further processing.

**[Conditions](https://mdf.readthedocs.io/en/latest/api/Specification.html#condition)** These are a core and distinctive element of the MDF specification, that complement other computational graph-based formats by providing a high-level set of descriptors for specifying conditional execution of **nodes**. This allows models with relatively complex execution requirements (e.g., containing cycles, branches, and/or temporal dependencies) to be expressed as **graphs** in a sufficiently abstract form that facilities exchange among high-level modeling environments without requiring that they be “lowered” to and then recovered from more elaborated procedural descriptions.

**[Parameters](https://mdf.readthedocs.io/en/latest/api/Specification.html#parameter)** Attributes that determine the configuration and operation of **nodes** and **edges**, can be defined in the MDF using parameters. In the case of parameters specifying large data structures (e.g., weight-matrices), arrays in widely used formats (e.g. numpy arrays, TensorFlow tensors) can be used, and serialisation in portable binary formats (e.g. BSON) is supported. Parameters can either be fixed values, which don't change when the **node** is executed, or can change over time (stateful parameters).

**[Functions](https://mdf.readthedocs.io/en/latest/api/Specification.html#function)** A single value which is evaluated as a function of values on **input ports** and other functions and **parameters**. A key distinction with **parameters** is that a function is always stateless.

**[Model metadata](https://mdf.readthedocs.io/en/latest/api/Specification.html#model)** There is the ability to add “metadata” to the **model**, **graph**, **nodes** and many of their sub elements which provide additional information about that element. While the metadata should not be essential to the mathematical description of the behavior/structure of the element, it could be useful for human interpretability of its function/purpose, or used when it is mapped to a specific application for simulation/visualization. Metadata can be added to the top level model to specify contact information, citations, acknowledgements, pointers to sample data and benchmark results, and environments in which the specified model was originally implemented and any that have been validated to support its execution.

<p align="center"><img src="https://github.com/ModECI/MDF/raw/main/docs/images/MDFgraph1.png" width="700px"><br/>
<sup><i><b>Fig 2:</b> A simple graph with 3 nodes and 2 edges expressed in MDF.</i></sup></p>

<p align="center"><img src="https://github.com/ModECI/MDF/raw/main/docs/images/MDFgraph2.png" width="700px"><br/>
<sup><i><b>Fig 3:</b> This graph illustrates the ability to specify behavior that extends beyond the directed flow through the graph. Here, Node 1 generates a random number and transmits that number to Node 2. Node 2 will only run if the number it receives from Node 1 is greater than 10.</i></sup></p>

## Installation

### Requirements
Requires Python >= 3.7

### Quick start
```
pip install modeci-mdf
```
For more detailed installation instructions see [here](https://mdf.readthedocs.io/en/latest/api/Installation.html).

For guidelines on contributing to the development of MDF, see [here](https://github.com/ModECI/MDF/blob/main/CONTRIBUTING.md).

## Examples

To get started, follow the simple example in a Jupyter notebook [here](https://github.com/ModECI/MDF/blob/main/examples/SimpleExample.ipynb)


Multiple examples of serialized MDF files, the Python scripts used to generate them, as well as mappings to target environments can be found [here](https://github.com/ModECI/MDF/blob/main/examples/README.md).

[actions-badge]:            https://github.com/ModECI/MDF/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/ModECI/MDF/actions
[rtd-badge]:                https://readthedocs.org/projects/mdf/badge/?version=latest
[rtd-link]:                 https://mdf.readthedocs.io/en/latest/?badge=latest
