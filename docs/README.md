# Specification of ModECI v0.4
**Note: the ModECI MDF specification is still in development!** See [here](https://github.com/ModECI/MDF/issues) for ongoing discussions.
## Model
The top level construct in MDF is Model, which may contain multiple <a href="#graph">Graph</a> objects and model attribute(s)

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>A unique identifier for this Model</i></td>
 </tr>


  <tr>
    <td><b>format</b></td>
    <td>str</td>
    <td><i>Information on the version of MDF used in this file</i></td>
 </tr>


  <tr>
    <td><b>generating_application</b></td>
    <td>str</td>
    <td><i>Information on what application generated/saved this file</i></td>
 </tr>


  <tr>
    <td><b>onnx_opset_version</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>The ONNX opset used for any ONNX functions in this model.</i></td>
 </tr>


</table>

### Allowed children
<table>
  <tr>
    <td><b>graphs</b></td>
    <td><a href="#graph">Graph</a></td>
    <td><i>The collection of graphs that make up the MDF model.</i></td>
  </tr>


</table>

## Graph
A directed graph consisting of <a href="#node">Node</a>s (with <a href="#parameter">Parameter</a>s and <a href="#function">Function</a>s evaluated internally) connected via <a href="#edge">Edge</a>s.

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>A unique identifier for this Graph</i></td>
 </tr>


  <tr>
    <td><b>parameters</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Dictionary of global parameters for the Graph</i></td>
 </tr>


  <tr>
    <td><b>conditions</b></td>
    <td>Union[ConditionSet, NoneType]</td>
    <td><i>The ConditionSet stored as dictionary for scheduling of the Graph</i></td>
 </tr>


</table>

### Allowed children
<table>
  <tr>
    <td><b>nodes</b></td>
    <td><a href="#node">Node</a></td>
    <td><i>One or more <a href="#node">Node</a>(s) present in the graph</i></td>
  </tr>


  <tr>
    <td><b>edges</b></td>
    <td><a href="#edge">Edge</a></td>
    <td><i>Zero or more <a href="#edge">Edge</a>(s) present in the graph</i></td>
  </tr>


</table>

## Node
A self contained unit of evaluation receiving input from other nodes on <a href="#inputport">InputPort</a>(s). The values from these are processed via a number of <a href="#function">Function</a>(s) and one or more final values
are calculated on the <a href="#outputport">OutputPort</a>(s)

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>A unique identifier for the node.</i></td>
 </tr>


</table>

### Allowed children
<table>
  <tr>
    <td><b>input_ports</b></td>
    <td><a href="#inputport">InputPort</a></td>
    <td><i>Dictionary of the <a href="#inputport">InputPort</a> objects in the Node</i></td>
  </tr>


  <tr>
    <td><b>functions</b></td>
    <td><a href="#function">Function</a></td>
    <td><i>The <a href="#function">Function</a>(s) for computation the node</i></td>
  </tr>


  <tr>
    <td><b>parameters</b></td>
    <td><a href="#parameter">Parameter</a></td>
    <td><i>Dictionary of <a href="#parameter">Parameter</a>(s) for the node</i></td>
  </tr>


  <tr>
    <td><b>output_ports</b></td>
    <td><a href="#outputport">OutputPort</a></td>
    <td><i>The <a href="#outputport">OutputPort</a>(s) containing evaluated quantities from the node</i></td>
  </tr>


</table>

## InputPort
The <a href="#inputport">InputPort</a> is an attribute of a Node which allows external information to be input to the Node

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>The unique (for this Node) id of the input port,</i></td>
 </tr>


  <tr>
    <td><b>shape</b></td>
    <td>Union[Tuple[int, ...], NoneType]</td>
    <td><i>The shape of the input port. This uses the same syntax as numpy ndarray shapes
(e.g., <b>numpy.zeros(shape)</b> would produce an array with the correct shape</i></td>
 </tr>


  <tr>
    <td><b>type</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>The data type of the input received at a port.</i></td>
 </tr>


</table>

## Function
A single value which is evaluated as a function of values on <a href="#inputport">InputPort</a>(s) and other Functions

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>The unique (for this Node) id of the function, which will be used in other <a href="#function">Function</a>s and
the <a href="#outputport">OutputPort</a>s for its value</i></td>
 </tr>


  <tr>
    <td><b>function</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>Which of the in-build MDF functions (<b>linear</b>, etc.). See supported functions:
https://mdf.readthedocs.io/en/latest/api/MDF<a href="#function">function</a>specifications.html</i></td>
 </tr>


  <tr>
    <td><b>args</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Dictionary of values for each of the arguments for the Function, e.g. if the in-built function
is linear(slope),the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}</i></td>
 </tr>


  <tr>
    <td><b>value</b></td>
    <td>Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]</td>
    <td><i>If the function is a value expression, this attribute will contain the expression and the function
and args attributes will be None.</i></td>
 </tr>


</table>

## Parameter
A parameter of the <a href="#node">Node</a>, which can be: 1) a specific fixed <b>value</b> (a constant (int/float) or an array) 2) a string expression for the <b>value</b> referencing other named <a href="#parameter">Parameter</a>(s). which may be stateful (i.e. can change value over multiple executions of the <a href="#node">Node</a>); 3) be evaluated by an
inbuilt <b>function</b> with <b>args</b>; 4) or change from a <b>default_initial_value</b> with a <b>time_derivative</b>.

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i></i></td>
 </tr>


  <tr>
    <td><b>value</b></td>
    <td>Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]</td>
    <td><i>The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values</i></td>
 </tr>


  <tr>
    <td><b>default_initial_value</b></td>
    <td>Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]</td>
    <td><i>The initial value of the parameter, only used when parameter is stateful.</i></td>
 </tr>


  <tr>
    <td><b>time_derivative</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>How the parameter changes with time, i.e. ds/dt. Units of time are seconds.</i></td>
 </tr>


  <tr>
    <td><b>function</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>Which of the in-build MDF functions (linear etc.) this uses, See
https://mdf.readthedocs.io/en/latest/api/MDF<a href="#function">function</a>specifications.html</i></td>
 </tr>


  <tr>
    <td><b>args</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Dictionary of values for each of the arguments for the function of the parameter,
e.g. if the in-build function is <b>linear(slope)</b>, the args here could be <b>{"slope": 3}</b> or
<b>{"slope": "input_port_0 + 2"}</b></i></td>
 </tr>


</table>

### Allowed children
<table>
  <tr>
    <td><b>conditions</b></td>
    <td><a href="#parametercondition">ParameterCondition</a></td>
    <td><i>Parameter specific conditions</i></td>
  </tr>


</table>

## ParameterCondition
A condition to test on a Node's parameters, which if true, sets the value of this Parameter

### Allowed parameters
<table>
  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>A unique identifier for the ParameterCondition</i></td>
 </tr>


  <tr>
    <td><b>test</b></td>
    <td>Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]</td>
    <td><i>The boolean expression to evaluate</i></td>
 </tr>


  <tr>
    <td><b>value</b></td>
    <td>Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]</td>
    <td><i>The new value of the Parameter if the test is true</i></td>
 </tr>


</table>

## OutputPort
The <a href="#outputport">OutputPort</a> is an attribute of a <a href="#node">Node</a> which exports information to another <a href="#node">Node</a> connected by an <a href="#edge">Edge</a>

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>Unique identifier for the output port.</i></td>
 </tr>


  <tr>
    <td><b>value</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>The value of the <a href="#outputport">OutputPort</a> in terms of the <a href="#inputport">InputPort</a>, <a href="#function">Function</a> values, and
<a href="#parameter">Parameter</a> values.</i></td>
 </tr>


  <tr>
    <td><b>shape</b></td>
    <td>Union[Tuple[int, ...], NoneType]</td>
    <td><i>The shape of the output port. This uses the same syntax as numpy ndarray shapes
(e.g., <b>numpy.zeros(shape)</b> would produce an array with the correct shape</i></td>
 </tr>


  <tr>
    <td><b>type</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>The data type of the output sent by a port.</i></td>
 </tr>


</table>

## Edge
An <a href="#edge">Edge</a> is an attribute of a <a href="#graph">Graph</a> that transmits computational results from a sender's <a href="#outputport">OutputPort</a> to a receiver's <a href="#inputport">InputPort</a>.

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>id</b></td>
    <td>str</td>
    <td><i>A unique string identifier for this edge.</i></td>
 </tr>


  <tr>
    <td><b>sender</b></td>
    <td>str</td>
    <td><i>The <b>id</b> of the <a href="#node">Node</a> which is the source of the edge.</i></td>
 </tr>


  <tr>
    <td><b>receiver</b></td>
    <td>str</td>
    <td><i>The <b>id</b> of the <a href="#node">Node</a> which is the target of the edge.</i></td>
 </tr>


  <tr>
    <td><b>sender_port</b></td>
    <td>str</td>
    <td><i>The id of the <a href="#outputport">OutputPort</a> on the sender <a href="#node">Node</a>, whose value should be sent to the
<b>receiver_port</b></i></td>
 </tr>


  <tr>
    <td><b>receiver_port</b></td>
    <td>str</td>
    <td><i>The id of the InputPort on the receiver <a href="#node">Node</a></i></td>
 </tr>


  <tr>
    <td><b>parameters</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Dictionary of parameters for the edge.</i></td>
 </tr>


</table>

## Condition
A set of descriptors which specifies conditional execution of Nodes to meet complex execution requirements.

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>type</b></td>
    <td>str</td>
    <td><i>The type of <a href="#condition">Condition</a> from the library</i></td>
 </tr>


  <tr>
    <td><b>kwargs</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>The dictionary of keyword arguments needed to evaluate the <a href="#condition">Condition</a></i></td>
 </tr>


</table>

## ConditionSet
Specifies the non-default pattern of execution of Nodes

### Allowed parameters
<table>
  <tr>
    <td><b>metadata</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.</i></td>
 </tr>


  <tr>
    <td><b>node_specific</b></td>
    <td>Union[Condition, NoneType]</td>
    <td><i>A dictionary mapping nodes to any non-default run conditions</i></td>
 </tr>


  <tr>
    <td><b>termination</b></td>
    <td>Union[Condition, NoneType]</td>
    <td><i>A dictionary mapping time scales of model execution to conditions indicating when they end</i></td>
 </tr>


</table>
