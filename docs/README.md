# Specification of ModECI v0.1
**Note: specification in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**


## Model
The top level Model containing a number of <a href="#graph">Graph</a>s
#### Allowed parameters
<table><tr><td><b>format</b></td><td>str</td><td><i>Information on the version of MDF used in this file</i></td></tr>

<tr><td><b>generating_application</b></td><td>str</td><td><i>Information on what application generated/saved this file</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>graphs</b></td><td><a href="#graph">Graph</a></td><td><i>The list of <a href="#graph">Graph</a>s in this Model</i></td></tr>


</table>

## Graph
#### Allowed parameters
<table><tr><td><b>parameters</b></td><td>dict</td><td><i>Dict of global parameters for the Graph</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>nodes</b></td><td><a href="#node">Node</a></td><td><i>The <a href="#node">Node</a>s present in the Graph</i></td></tr>

<tr><td><b>edges</b></td><td><a href="#edge">Edge</a></td><td><i>The <a href="#edge">Edge</a>s between <a href="#node">Node</a>s in the Graph</i></td></tr>


</table>

## Node
#### Allowed parameters
<table><tr><td><b>type</b></td><td>str</td><td><i>Type...</i></td></tr>

<tr><td><b>parameters</b></td><td>dict</td><td><i>Dict of parameters for the Node</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>input_ports</b></td><td><a href="#inputport">InputPort</a></td><td><i>The <a href="#inputport">InputPort</a>s into the Node</i></td></tr>

<tr><td><b>functions</b></td><td><a href="#function">Function</a></td><td><i>The <a href="#function">Function</a>s for the Node</i></td></tr>

<tr><td><b>output_ports</b></td><td><a href="#outputport">OutputPort</a></td><td><i>The <a href="#outputport">OutputPort</a>s into the Node</i></td></tr>


</table>

## InputPort
#### Allowed parameters
<table><tr><td><b>shape</b></td><td>str</td><td><i>The shape of the variable (limited support so far...)</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## Function
#### Allowed parameters
<table><tr><td><b>function</b></td><td>str</td><td><i>Which of the in-build MDF functions (linear etc.) this uses</i></td></tr>

<tr><td><b>args</b></td><td>dict</td><td><i>Dictionary of arguments for the Function</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## OutputPort
#### Allowed parameters
<table><tr><td><b>value</b></td><td>str</td><td><i>The value of the OutputPort in terms of the <a href="#inputport">InputPort</a> and <a href="#function">Function</a> values</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## Edge
#### Allowed parameters
<table><tr><td><b>sender</b></td><td>str</td><td><i>The <a href="#node">Node</a> which is the source of the Edge</i></td></tr>

<tr><td><b>receiver</b></td><td>str</td><td><i>The <a href="#node">Node</a> which is the target of the Edge</i></td></tr>

<tr><td><b>sender_port</b></td><td>str</td><td><i>The <a href="#outputport">OutputPort</a> on the sender <a href="#node">Node</a></i></td></tr>

<tr><td><b>receiver_port</b></td><td>str</td><td><i>The <a href="#inputport">InputPort</a> on the sender <a href="#node">Node</a></i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

