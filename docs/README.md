# Specification of ModECI v0.3
**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice.** See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.
## Model
The top level construct in MDF is Model, which may contain multiple :class:`.Graph` objects and model attribute(s)

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
A directed graph consisting of Node(s) connected via Edge(s)

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
    <td><i></i></td>
  </tr>


  <tr>
    <td><b>edges</b></td>
    <td><a href="#edge">Edge</a></td>
    <td><i></i></td>
  </tr>


</table>

## Node
The values from these are processed via a number of :class:`Function`\(s) and one or more final values
are calculated on the :class:`OutputPort`\(s)

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
    <td><i>Dictionary of the :class:`InputPort` objects in the Node</i></td>
  </tr>


  <tr>
    <td><b>functions</b></td>
    <td><a href="#function">Function</a></td>
    <td><i>The :class:`Function`\(s) for computation the node</i></td>
  </tr>


  <tr>
    <td><b>parameters</b></td>
    <td><a href="#parameter">Parameter</a></td>
    <td><i>Dictionary of :class:`Parameter`\(s) for the node</i></td>
  </tr>


  <tr>
    <td><b>output_ports</b></td>
    <td><a href="#outputport">OutputPort</a></td>
    <td><i>The :class:`OutputPort`\(s) containing evaluated quantities from the node</i></td>
  </tr>


</table>

## InputPort
The :class:`InputPort` is an attribute of a Node which allows external information to be input to the Node

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
(e.g., :code:`numpy.zeros(shape)` would produce an array with the correct shape</i></td>
 </tr>


  <tr>
    <td><b>type</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>The data type of the input received at a port.</i></td>
 </tr>


</table>

## Function
A single value which is evaluated as a function of values on :class:`InputPort`\(s) and other Functions

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
    <td><i>The unique (for this Node) id of the function, which will be used in other :class:`~Function`s and
the :class:`~OutputPort`s for its value</i></td>
 </tr>


  <tr>
    <td><b>function</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>Which of the in-build MDF functions (:code:`linear`, etc.). See supported functions:
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
referencing other :class:`Parameter`\(s)), be evaluated by an inbuilt function with args, or change from a
:code:`default<a href="#initial">initial</a>value` with a :code:`time<a href="#derivative`.">derivative`.</a>

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
    <td><i>Which of the in-build MDF functions (linear etc.) this uses, See</i></td>
 </tr>


  <tr>
    <td><b>args</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Dictionary of values for each of the arguments for the function of the parameter,
e.g. if the in-build function is :code:`linear(slope)`, the args here could be :code:`{"slope": 3}` or
:code:`{"slope": "input_port_0 + 2"}`</i></td>
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
connected by an :class:`Edge`

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
    <td><i>The value of the :class:`OutputPort` in terms of the :class:`InputPort`, :class:`Function` values, and
:class:`Parameter` values.</i></td>
 </tr>


  <tr>
    <td><b>shape</b></td>
    <td>Union[Tuple[int, ...], NoneType]</td>
    <td><i>The shape of the output port. This uses the same syntax as numpy ndarray shapes
(e.g., :code:`numpy.zeros(shape)` would produce an array with the correct shape</i></td>
 </tr>


  <tr>
    <td><b>type</b></td>
    <td>Union[str, NoneType]</td>
    <td><i>The data type of the output sent by a port.</i></td>
 </tr>


</table>

## Edge
:class:`OutputPort` to a receiver's :class:`InputPort`.

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
    <td><i>The :code:`id` of the :class:`~Node` which is the source of the edge.</i></td>
 </tr>


  <tr>
    <td><b>receiver</b></td>
    <td>str</td>
    <td><i>The :code:`id` of the :class:`~Node` which is the target of the edge.</i></td>
 </tr>


  <tr>
    <td><b>sender_port</b></td>
    <td>str</td>
    <td><i>The id of the :class:`~OutputPort` on the sender :class:`~Node`, whose value should be sent to the
:code:`receiver<a href="#port`">port`</a></i></td>
 </tr>


  <tr>
    <td><b>receiver_port</b></td>
    <td>str</td>
    <td><i>The id of the InputPort on the receiver :class:`~Node'</i></td>
 </tr>


  <tr>
    <td><b>parameters</b></td>
    <td>Union[Any, NoneType]</td>
    <td><i>Dictionary of parameters for the edge.</i></td>
 </tr>


</table>
