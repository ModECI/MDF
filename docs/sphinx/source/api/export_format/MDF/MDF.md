# MDF Examples

Examples of Python, JSON and YAML files to illustrate the structure and usage of MDF.

[Simple](https://mdf.readthedocs.io/en/latest/api/export_format/MDF/MDF.html#simple-example) | [ABCD](https://mdf.readthedocs.io/en/latest/api/export_format/MDF/MDF.html#abcd) | [Arrays](https://mdf.readthedocs.io/en/latest/api/export_format/MDF/MDF.html#arrays) | [States](https://mdf.readthedocs.io/en/latest/api/export_format/MDF/MDF.html#states) | [Conditions](https://mdf.readthedocs.io/en/latest/api/export_format/MDF/MDF.html#conditions) | [Parameters and Functions](https://mdf.readthedocs.io/en/latest/api/export_format/MDF/MDF.html#parameters-and-functions)

## Simple example

[Python source](https://github.com/ModECI/MDF/blob/main/examples/MDF/simple.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/ABCD.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/ABCD.yaml)

A simple example with 2 [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) connected by an [Edge](https://mdf.readthedocs.io/en/latest/api/Specification.html#edge):

![1 resimple](https://user-images.githubusercontent.com/100205503/206762808-949c1a33-f3e9-44e1-a1b5-92762b88b2ab.png)


With more detail on [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) (showing [Input Ports](https://mdf.readthedocs.io/en/latest/api/Specification.html#inputport) (green), [Parameters](https://mdf.readthedocs.io/en/latest/api/Specification.html#parameter) (blue) and [Output Ports](https://mdf.readthedocs.io/en/latest/api/Specification.html#output_port)) (red) and [Edges](https://mdf.readthedocs.io/en/latest/api/Specification.html#edge):

 ![2 simple_3](https://user-images.githubusercontent.com/100205503/206763192-ba85d1e6-02d8-477b-961e-ebddf0447f66.png)



## ABCD

[Python source](https://github.com/ModECI/MDF/blob/main/examples/MDF/abcd.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/ABCD.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/ABCD.yaml)

Another simple example with more [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node).

![2 abcd](https://user-images.githubusercontent.com/100205503/206763843-4d3051b5-dd06-4243-8f0c-dd092b1ce99e.png) &nbsp; ![2 abcd_3](https://user-images.githubusercontent.com/100205503/206764130-bfcd2adb-fbfd-4dc2-b933-7931f179370b.png)


## Arrays

[Python source](https://github.com/ModECI/MDF/blob/main/examples/MDF/arrays.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/Arrays.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/Arrays.yaml)

An example using arrays for [Parameters](https://mdf.readthedocs.io/en/latest/api/Specification.html#parameter) and weights on [Edges](https://mdf.readthedocs.io/en/latest/api/Specification.html#edge).

![2 arrays](https://user-images.githubusercontent.com/100205503/206764818-3b82ea6f-ce66-4def-a059-7b11a9040a4f.png)


## States

[Python source](https://github.com/ModECI/MDF/blob/main/examples/MDF/states.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/States.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/States.yaml)

An example with [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) containing persistent [States](https://github.com/ModECI/MDF/blob/main/examples/MDF/States.yaml).

![2 states](https://user-images.githubusercontent.com/100205503/206765361-6a75d5b0-8f32-4be8-ac3c-9a2011f482ca.png)


## Conditions

[Python source](https://github.com/ModECI/MDF/blob/main/examples/MDF/abc_conditions.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/abc_conditions.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/abc_conditions.yaml)

A simple 3 [Nodes](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) graph with scheduling [Conditions](https://mdf.readthedocs.io/en/latest/api/Specification.html#condition). For more examples of conditions see [here](https://github.com/ModECI/MDF/blob/main/examples/MDF/conditions/README.md).

![2 abc_conditions](https://user-images.githubusercontent.com/100205503/206766574-f1afa28f-0e30-48bb-a582-bf7c03caba4b.png)


## Parameters and Functions

[Python source](https://github.com/ModECI/MDF/blob/main/examples/MDF/params_funcs.py) | [JSON](https://github.com/ModECI/MDF/blob/main/examples/MDF/ParametersFunctions.json) | [YAML](https://github.com/ModECI/MDF/blob/main/examples/MDF/ParametersFunctions.yaml)

A simple [Node](https://mdf.readthedocs.io/en/latest/api/Specification.html#node) with a number of different types of [Parameters](https://mdf.readthedocs.io/en/latest/api/Specification.html#parameter) (in blue; fixed and **stateful**) and [Functions](https://mdf.readthedocs.io/en/latest/api/Specification.html#function) (in purple; can be built in or ONNX based).

![2params_funcs](https://user-images.githubusercontent.com/100205503/206767670-58828404-ea5b-4361-95b0-242dc5ecdffa.png)


## More examples

There are further examples under development, including of a Recurrent Neural Network (RNN), and an Integrate and Fire (IaF) neuron model in [this directory](https://github.com/ModECI/MDF/tree/main/examples/MDF/RNN).
