# Specification of ModECI v0.1
**Note: specification in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**

## Model
#### Allowed parameters
<table><tr><td><b>format</b></td><td>str</td><td><i>Information on verson of MDF</i></td></tr>

<tr><td><b>generating_application</b></td><td>str</td><td><i>Information on what application generated/saved this file</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>graphs</b></td><td><a href="#modelgraph">ModelGraph</a></td><td><i>The definition of top level entry ...</i></td></tr>


</table>

## ModelGraph
#### Allowed parameters
<table><tr><td><b>parameters</b></td><td>dict</td><td><i>Dict of global parameters for the network</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>nodes</b></td><td><a href="#node">Node</a></td><td><i>The definition of node ...</i></td></tr>

<tr><td><b>edges</b></td><td><a href="#edge">Edge</a></td><td><i>The definition of edge...</i></td></tr>


</table>

## Node
#### Allowed parameters
<table><tr><td><b>type</b></td><td>str</td><td><i>Type...</i></td></tr>

<tr><td><b>parameters</b></td><td>dict</td><td><i>Dict of parameters for the node</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

#### Allowed children
<table><tr><td><b>input_ports</b></td><td><a href="#inputport">InputPort</a></td><td><i>Dict of ...</i></td></tr>

<tr><td><b>functions</b></td><td><a href="#function">Function</a></td><td><i>Dict of functions for the node</i></td></tr>

<tr><td><b>output_ports</b></td><td><a href="#outputport">OutputPort</a></td><td><i>Dict of ...</i></td></tr>


</table>

## InputPort
#### Allowed parameters
<table><tr><td><b>shape</b></td><td>str</td><td><i>...</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## Function
#### Allowed parameters
<table><tr><td><b>function</b></td><td>str</td><td><i>...</i></td></tr>

<tr><td><b>args</b></td><td>dict</td><td><i>Dict of args...</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## OutputPort
#### Allowed parameters
<table><tr><td><b>value</b></td><td>str</td><td><i>...</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

## Edge
#### Allowed parameters
<table><tr><td><b>sender</b></td><td>str</td><td><i>...</i></td></tr>

<tr><td><b>receiver</b></td><td>str</td><td><i>...</i></td></tr>

<tr><td><b>sender_port</b></td><td>str</td><td><i>...</i></td></tr>

<tr><td><b>receiver_port</b></td><td>str</td><td><i>...</i></td></tr>

<tr><td><b>id</b></td><td>str</td><td><i>Unique ID of element</i></td></tr>

<tr><td><b>notes</b></td><td>str</td><td><i>Human readable notes</i></td></tr>


</table>

