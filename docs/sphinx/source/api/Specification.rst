============================
Specification of ModECI v0.4
============================

**Note: the ModECI MDF specification is still in development!** See `here <https://github.com/ModECI/MDF/issues>`_ for ongoing discussions.

=====
Model
=====
The top level construct in MDF is Model, which may contain multiple `Graph <#graph>`__ objects and model attribute(s)

**Allowed parameters**

==========================  ===========  =============================================================================================
Allowed field               Data Type    Description
==========================  ===========  =============================================================================================
**metadata**                Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**                      str          A unique identifier for this Model
**format**                  str          Information on the version of MDF used in this file
**generating_application**  str          Information on what application generated/saved this file
**onnx_opset_version**      Optional     The ONNX opset used for any ONNX functions in this model.
==========================  ===========  =============================================================================================

**Allowed children**

===============  ================  ====================================================
Allowed child    Data Type         Description
===============  ================  ====================================================
**graphs**       `List <#list>`__  The collection of graphs that make up the MDF model.
===============  ================  ====================================================

=====
Graph
=====
A directed graph consisting of `Nodes <#node>`__ (with `Parameters <#parameter>`__ and `Functions <#function>`__ evaluated internally) connected via `Edges <#edge>`__.

**Allowed parameters**

===============  ===========  =============================================================================================
Allowed field    Data Type    Description
===============  ===========  =============================================================================================
**metadata**     Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str          A unique identifier for this Graph
**parameters**   Optional     Dictionary of global parameters for the Graph
**conditions**   Optional     The ConditionSet stored as dictionary for scheduling of the Graph
===============  ===========  =============================================================================================

**Allowed children**

===============  ================  =====================================================
Allowed child    Data Type         Description
===============  ================  =====================================================
**nodes**        `List <#list>`__  One or more `Node(s) <#node>`__ present in the graph
**edges**        `List <#list>`__  Zero or more `Edge(s) <#edge>`__ present in the graph
===============  ================  =====================================================

====
Node
====
A self contained unit of evaluation receiving input from other nodes on `InputPort(s) <#inputport>`__. The values from these are processed via a number of `Function(s) <#function>`__ and one or more final values
are calculated on the `OutputPort(s) <#outputport>`__

**Allowed parameters**

===============  ===========  =============================================================================================
Allowed field    Data Type    Description
===============  ===========  =============================================================================================
**metadata**     Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str          A unique identifier for the node.
===============  ===========  =============================================================================================

**Allowed children**

================  ================  =================================================================================
Allowed child     Data Type         Description
================  ================  =================================================================================
**input_ports**   `List <#list>`__  Dictionary of the `InputPort <#inputport>`__ objects in the Node
**functions**     `List <#list>`__  The `Function(s) <#function>`__ for computation the node
**parameters**    `List <#list>`__  Dictionary of `Parameter(s) <#parameter>`__ for the node
**output_ports**  `List <#list>`__  The `OutputPort(s) <#outputport>`__ containing evaluated quantities from the node
================  ================  =================================================================================

=========
InputPort
=========
The `InputPort <#inputport>`__ is an attribute of a Node which allows external information to be input to the Node

**Allowed parameters**

===============  ===========  =============================================================================================
Allowed field    Data Type    Description
===============  ===========  =============================================================================================
**metadata**     Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str          The unique (for this Node) id of the input port,
**shape**        Optional     The shape of the input port. This uses the same syntax as numpy ndarray shapes
                              (e.g., **numpy.zeros(shape)** would produce an array with the correct shape
**type**         Optional     The data type of the input received at a port.
===============  ===========  =============================================================================================

========
Function
========
A single value which is evaluated as a function of values on `InputPort(s) <#inputport>`__ and other Functions

**Allowed parameters**

===============  ===========  ========================================================================================================
Allowed field    Data Type    Description
===============  ===========  ========================================================================================================
**metadata**     Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str          The unique (for this Node) id of the function, which will be used in other `Functions <#function>`__ and
                              the `OutputPorts <#outputport>`__ for its value
**function**     Optional     Which of the in-build MDF functions (**linear**, etc.). See supported functions:
                              https://mdf.readthedocs.io/en/latest/api/MDF_function_specifications.html
**args**         Optional     Dictionary of values for each of the arguments for the Function, e.g. if the in-built function
                              is linear(slope),the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}
**value**        Union        If the function is a value expression, this attribute will contain the expression and the function
                              and args attributes will be None.
===============  ===========  ========================================================================================================

=========
Parameter
=========
A parameter of the `Node <#node>`__, which can be: 1) a specific fixed **value** (a constant (int/float) or an array) 2) a string expression for the **value** referencing other named `Parameter(s) <#parameter>`__. which may be stateful (i.e. can change value over multiple executions of the `Node <#node>`__); 3) be evaluated by an
inbuilt **function** with **args**; 4) or change from a **default_initial_value** with a **time_derivative**.

**Allowed parameters**

=========================  ===========  ================================================================================================
Allowed field              Data Type    Description
=========================  ===========  ================================================================================================
**metadata**               Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**                     str
**value**                  Union        The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values
**default_initial_value**  Union        The initial value of the parameter, only used when parameter is stateful.
**time_derivative**        Optional     How the parameter changes with time, i.e. ds/dt. Units of time are seconds.
**function**               Optional     Which of the in-build MDF functions (linear etc.) this uses, See
                                        https://mdf.readthedocs.io/en/latest/api/MDF_function_specifications.html
**args**                   Optional     Dictionary of values for each of the arguments for the function of the parameter,
                                        e.g. if the in-build function is **linear(slope)**, the args here could be **{"slope": 3}** or
                                        **{"slope": "input_port_0 + 2"}**
=========================  ===========  ================================================================================================

**Allowed children**

===============  ================  =============================
Allowed child    Data Type         Description
===============  ================  =============================
**conditions**   `List <#list>`__  Parameter specific conditions
===============  ================  =============================

==================
ParameterCondition
==================
A condition to test on a Node's parameters, which if true, sets the value of this Parameter

**Allowed parameters**

===============  ===========  ==================================================
Allowed field    Data Type    Description
===============  ===========  ==================================================
**id**           str          A unique identifier for the ParameterCondition
**test**         Union        The boolean expression to evaluate
**value**        Union        The new value of the Parameter if the test is true
===============  ===========  ==================================================

==========
OutputPort
==========
The `OutputPort <#outputport>`__ is an attribute of a `Node <#node>`__ which exports information to another `Node <#node>`__ connected by an `Edge <#edge>`__

**Allowed parameters**

===============  ===========  ==============================================================================================================================
Allowed field    Data Type    Description
===============  ===========  ==============================================================================================================================
**metadata**     Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str          Unique identifier for the output port.
**value**        Optional     The value of the `OutputPort <#outputport>`__ in terms of the `InputPort <#inputport>`__, `Function <#function>`__ values, and
                              `Parameter <#parameter>`__ values.
**shape**        Optional     The shape of the output port. This uses the same syntax as numpy ndarray shapes
                              (e.g., **numpy.zeros(shape)** would produce an array with the correct shape
**type**         Optional     The data type of the output sent by a port.
===============  ===========  ==============================================================================================================================

====
Edge
====
An `Edge <#edge>`__ is an attribute of a `Graph <#graph>`__ that transmits computational results from a sender's `OutputPort <#outputport>`__ to a receiver's `InputPort <#inputport>`__.

**Allowed parameters**

=================  ===========  ============================================================================================================
Allowed field      Data Type    Description
=================  ===========  ============================================================================================================
**metadata**       Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**             str          A unique string identifier for this edge.
**sender**         str          The **id** of the `Node <#node>`__ which is the source of the edge.
**receiver**       str          The **id** of the `Node <#node>`__ which is the target of the edge.
**sender_port**    str          The id of the `OutputPort <#outputport>`__ on the sender `Node <#node>`__, whose value should be sent to the
                                **receiver_port**
**receiver_port**  str          The id of the InputPort on the receiver `Node <#node>`__
**parameters**     Optional     Dictionary of parameters for the edge.
=================  ===========  ============================================================================================================

=========
Condition
=========
A set of descriptors which specifies conditional execution of Nodes to meet complex execution requirements.

**Allowed parameters**

===============  ===========  =============================================================================================
Allowed field    Data Type    Description
===============  ===========  =============================================================================================
**metadata**     Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**type**         str          The type of `Condition <#condition>`__ from the library
**kwargs**       Optional     The dictionary of keyword arguments needed to evaluate the `Condition <#condition>`__
===============  ===========  =============================================================================================

============
ConditionSet
============
Specifies the non-default pattern of execution of Nodes

**Allowed parameters**

=================  ===========  =============================================================================================
Allowed field      Data Type    Description
=================  ===========  =============================================================================================
**metadata**       Optional     Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**node_specific**  Optional     A dictionary mapping nodes to any non-default run conditions
**termination**    Optional     A dictionary mapping time scales of model execution to conditions indicating when they end
=================  ===========  =============================================================================================
