# Specification of ModECI v0.2
**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**
## Model
The top level Model containing <a href="#graph">Graph</a>s consisting of <a href="#node">Node</a>s connected via <a href="#edge">Edge</a>s.
#### Allowed parameters
<table><tr><td><b>format</b></td><td>str</td><td><i>Information on the version of MDF used in this file</i></td></tr>

<tr><td><b>generating_application</b></td><td>str</td><td><i>Information on what application generated/saved this file</i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>graphs</b></td><td><a href="#graph">Graph</a></td><td><i>The list of <a href="#graph">Graph</a>s in this Model</i></td></tr>


</table>

## Graph
A directed graph consisting of <a href="#node">Node</a>s connected via <a href="#edge">Edge</a>s.
#### Allowed parameters
<table><tr><td><b>parameters</b></td><td>dict</td><td><i>Dict of global parameters for the Graph</i></td></tr>

<tr><td><b>conditions</b></td><td><a href="#conditionset">ConditionSet</a></td><td><i>The <a href="#conditionset">ConditionSet</a> for scheduling of the Graph</i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>nodes</b></td><td><a href="#node">Node</a></td><td><i>The <a href="#node">Node</a>s present in the Graph</i></td></tr>

<tr><td><b>edges</b></td><td><a href="#edge">Edge</a></td><td><i>The <a href="#edge">Edge</a>s between <a href="#node">Node</a>s in the Graph</i></td></tr>


</table>

## ConditionSet
Specifies the non-default pattern of execution of <a href="#node">Node</a>s
#### Allowed parameters
<table><tr><td><b>node_specific</b></td><td>dict</td><td><i>The <a href="#condition">Condition</a>s corresponding to each <a href="#node">Node</a></i></td></tr>

<tr><td><b>termination</b></td><td>dict</td><td><i>The <a href="#condition">Condition</a>s that indicate when model execution ends</i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>


</table>

## Node
A self contained unit of evaluation receiving input from other Nodes on <a href="#inputport">InputPort</a>s. The values from these are processed via a number of Functions and one or more final values are calculated on the <a href="#outputport">OutputPort</a>s 
#### Allowed parameters
<table><tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>input_ports</b></td><td><a href="#inputport">InputPort</a></td><td><i>The <a href="#inputport">InputPort</a>s into the Node</i></td></tr>

<tr><td><b>functions</b></td><td><a href="#function">Function</a></td><td><i>The <a href="#function">Function</a>s for the Node</i></td></tr>

<tr><td><b>parameters</b></td><td><a href="#parameter">Parameter</a></td><td><i>The <a href="#parameter">Parameter</a>s of the Node</i></td></tr>

<tr><td><b>output_ports</b></td><td><a href="#outputport">OutputPort</a></td><td><i>The <a href="#outputport">OutputPort</a>s containing evaluated quantities from the Node</i></td></tr>


</table>

## InputPort
The InputPort is an attribute of a <a href="#node">Node</a> which allows external information to be input to the <a href="#node">Node</a>
#### Allowed parameters
<table><tr><td><b>shape</b></td><td>str</td><td><i>The shape of the variable (note: there is limited support for this so far...)</i></td></tr>

<tr><td><b>type</b></td><td>str</td><td><i>The type of the variable (note: there is limited support for this so far </i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## Function
A single value which is evaluated as a function of values on <a href="#inputport">InputPort</a>s and other Functions
#### Allowed parameters
<table><tr><td><b>function</b></td><td>str</td><td><i>Which of the in-build MDF functions (linear etc.) this uses</i></td></tr>

<tr><td><b>value</b></td><td>str</td><td><i>evaluable expression</i></td></tr>

<tr><td><b>args</b></td><td>dict</td><td><i>Dictionary of values for each of the arguments for the Function, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## Parameter
A Parameter of the <a href="#node">Node</a>, which can have a specific value (a constant or a string expression referencing other Parameters), be evaluated by an inbuilt function with args, or change from a default<a href="#initial">initial</a>value with a time<a href="#derivative">derivative</a>
#### Allowed parameters
<table><tr><td><b>default_initial_value</b></td><td>str</td><td><i>The initial value of the parameter</i></td></tr>

<tr><td><b>value</b></td><td>EvaluableExpression</td><td><i>The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values</i></td></tr>

<tr><td><b>time_derivative</b></td><td>str</td><td><i>How the parameter with time, i.e. ds/dt. Units of time are seconds.</i></td></tr>

<tr><td><b>function</b></td><td>str</td><td><i>Which of the in-build MDF functions (linear etc.) this uses</i></td></tr>

<tr><td><b>args</b></td><td>dict</td><td><i>Dictionary of values for each of the arguments for the function of the parameter, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}</i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## OutputPort
The OutputPort is an attribute of a <a href="#node">Node</a> which exports information to another <a href="#node">Node</a> connected by an <a href="#edge">Edge</a>
#### Allowed parameters
<table><tr><td><b>value</b></td><td>str</td><td><i>The value of the OutputPort in terms of the <a href="#inputport">InputPort</a> and <a href="#function">Function</a> values</i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## Edge
An Edge is an attribute of a <a href="#graph">Graph</a> that transmits computational results from a sender's <a href="#outputport">OutputPort</a> to a receiver's <a href="#inputport">InputPort</a>
#### Allowed parameters
<table><tr><td><b>parameters</b></td><td>dict</td><td><i>Dict of parameters for the Edge</i></td></tr>

<tr><td><b>sender</b></td><td>str</td><td><i>The id of the <a href="#node">Node</a> which is the source of the Edge</i></td></tr>

<tr><td><b>receiver</b></td><td>str</td><td><i>The id of the <a href="#node">Node</a> which is the target of the Edge</i></td></tr>

<tr><td><b>sender_port</b></td><td>str</td><td><i>The id of the <a href="#outputport">OutputPort</a> on the sender <a href="#node">Node</a>, whose value should be sent to the receiver<a href="#port">port</a></i></td></tr>

<tr><td><b>receiver_port</b></td><td>str</td><td><i>The id of the <a href="#inputport">InputPort</a> on the receiver <a href="#node">Node</a></i></td></tr>

<tr><td><b>metadata</b></td><td>dict</td><td><i>Dict of metadata for the model element</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

