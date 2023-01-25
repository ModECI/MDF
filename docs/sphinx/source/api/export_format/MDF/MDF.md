# MDF Examples

Examples of [Python](https://python.org), [JSON](https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Objects/JSON#:~:text=JavaScript%20Object%20Notation%20(JSON)%20is,page%2C%20or%20vice%20versa) and [YAML](https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/) files to illustrate the structure and usage of MDF.

<a href="#simple"> Simple </a>| <a href="#abcd"> ABCD </a> | <a href="#arrays"> Arrays </a> | <a href="#st"> States </a> | <a href="#conditions"> Conditions </a> | <a href="#parameters-and-functions"> Parameters and Functions </a>

<p id="simple"></p>

## Simple example

[Python Source](https://github.com/ModECI/MDF/blob/main/examples/MDF/simple.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/Simple.yaml)

A simple example with 2 [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) connected by an [Edge](https://mdf.readthedocs.io/en/latest/api/Specification.html#edge):

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/simple.png" width="201" height="35" />


With more detail on [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) (showing [Input Ports](https://mdf.readthedocs.io/en/latest/api/Specification.html#inputport) (green), [Parameters](https://mdf.readthedocs.io/en/latest/api/Specification.html#parameter) (blue) and [Output Ports](https://mdf.readthedocs.io/en/latest/api/Specification.html#output_port)) (red) and [Edges](https://mdf.readthedocs.io/en/latest/api/Specification.html#edge):

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/simple_3.png" width="335" height="236" />

<p id="abcd"></p>

## ABCD

[Python Source](https://github.com/ModECI/MDF/blob/main/examples/MDF/abcd.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/ABCD.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/ABCD.yaml)

Another simple example with more [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node).

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/abcd.png" width="300" height="31" />

&nbsp;

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/abcd_3.png" width="360" height="800" />

<p id="arrays"></p>

## Arrays

[Python Source](https://github.com/ModECI/MDF/blob/main/examples/MDF/arrays.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/Arrays.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/Arrays.yaml)

An example using arrays for [Parameters](https://mdf.readthedocs.io/en/latest/api/Specification.html#parameter) and weights on [Edges](https://mdf.readthedocs.io/en/latest/api/Specification.html#edge).

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/arrays.png" width="329" height="157" />


<p id="st"></p>

## States

[Python Source](https://github.com/ModECI/MDF/blob/main/examples/MDF/states.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/States.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/States.yaml)

An example with [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) containing persistent [States](https://mdf.readthedocs.io/en/latest/api/Specification.html#state).

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/states.png" width="346" height="55" />

<p id="conditions"></p>

## Conditions

[Python Source](https://github.com/ModECI/MDF/blob/main/examples/MDF/abc_conditions.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/abc_conditions.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/abc_conditions.yaml)

A simple 3 [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) graph with scheduling [Conditions](https://mdf.readthedocs.io/en/latest/api/Specification.html#condition). For more examples of conditions see [here](https://github.com/ModECI/MDF/blob/main/examples/MDF/conditions/README.md).

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/abc_conditions.png" width="330" height="198" />

<p id="parameters-and-functions"></p>

## Parameters and Functions

[Python Source](https://github.com/ModECI/MDF/blob/main/examples/MDF/params_funcs.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/ParametersFunctions.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/ParametersFunctions.yaml)

A simple [Node](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) with a number of different types of [Parameters](https://mdf.readthedocs.io/en/latest/api/Specification.html#parameter) (in blue; fixed and **stateful**) and [Functions](https://mdf.readthedocs.io/en/latest/api/Specification.html#function) (in purple; can be built in or ONNX based).

<img src="https://raw.githubusercontent.com/ModECI/MDF/main/examples/MDF/images/params_funcs.png" width="250" height="198" />


## More examples

There are further examples under development, including of a Recurrent Neural Network (RNN), and an Integrate and Fire (IaF) neuron model in [this directory](https://github.com/ModECI/MDF/tree/main/examples/MDF/RNN).
