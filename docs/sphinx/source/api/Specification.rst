================================
Specification of ModECI v0.4 RST
================================

**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice.** See `here <https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification>`_ for ongoing discussions.

=====
Model
=====
The top level construct in MDF is Model, which may contain multiple `Graph <#graph>`_ objects and model attribute(s)

**Allowed parameters**

==========================  ====================  =============================================================================================
Allowed field               Data Type             Description
==========================  ====================  =============================================================================================
**metadata**                Union[Any, NoneType]  Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**                      str                   A unique identifier for this Model
**format**                  str                   Information on the version of MDF used in this file
**generating_application**  str                   Information on what application generated/saved this file
**onnx_opset_version**      Union[str, NoneType]  The ONNX opset used for any ONNX functions in this model.
==========================  ====================  =============================================================================================

**Allowed children**

===============  =================  ====================================================
Allowed child    Data Type          Description
===============  =================  ====================================================
**graphs**       `Graph <#graph>`_  The collection of graphs that make up the MDF model.
===============  =================  ====================================================

=====
Graph
=====
A directed graph consisting of Node(s) connected via Edge(s)

**Allowed parameters**

===============  =============================  =============================================================================================
Allowed field    Data Type                      Description
===============  =============================  =============================================================================================
**metadata**     Union[Any, NoneType]           Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str                            A unique identifier for this Graph
**parameters**   Union[Any, NoneType]           Dictionary of global parameters for the Graph
**conditions**   Union[ConditionSet, NoneType]  The ConditionSet stored as dictionary for scheduling of the Graph
===============  =============================  =============================================================================================

**Allowed children**

===============  ===============  ====================================================
Allowed child    Data Type        Description
===============  ===============  ====================================================
**nodes**        `Node <#node>`_  One or more `Node(s) <#node>`_ present in the graph
**edges**        `Edge <#edge>`_  Zero or more `Edge(s) <#edge>`_ present in the graph
===============  ===============  ====================================================

====
Node
====
A self contained unit of evaluation receiving input from other nodes on `InputPort(s) <#inputport>`_. The values from these are processed via a number of `Function(s) <#function>`_ and one or more final values
are calculated on the `OutputPort(s) <#outputport>`_

**Allowed parameters**

===============  ====================  =============================================================================================
Allowed field    Data Type             Description
===============  ====================  =============================================================================================
**metadata**     Union[Any, NoneType]  Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str                   A unique identifier for the node.
===============  ====================  =============================================================================================

**Allowed children**

================  ===========================  ================================================================================
Allowed child     Data Type                    Description
================  ===========================  ================================================================================
**input_ports**   `InputPort <#inputport>`_    Dictionary of the `InputPort <#inputport>`_ objects in the Node
**functions**     `Function <#function>`_      The `Function(s) <#function>`_ for computation the node
**parameters**    `Parameter <#parameter>`_    Dictionary of `Parameter(s) <#parameter>`_ for the node
**output_ports**  `OutputPort <#outputport>`_  The `OutputPort(s) <#outputport>`_ containing evaluated quantities from the node
================  ===========================  ================================================================================

=========
InputPort
=========
The `InputPort <#inputport>`_ is an attribute of a Node which allows external information to be input to the Node

**Allowed parameters**

===============  ================================  =============================================================================================
Allowed field    Data Type                         Description
===============  ================================  =============================================================================================
**metadata**     Union[Any, NoneType]              Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str                               The unique (for this Node) id of the input port,
**shape**        Union[Tuple[int, ...], NoneType]  The shape of the input port. This uses the same syntax as numpy ndarray shapes
                                                   (e.g., **numpy.zeros(shape)** would produce an array with the correct shape
**type**         Union[str, NoneType]              The data type of the input received at a port.
===============  ================================  =============================================================================================

========
Function
========
A single value which is evaluated as a function of values on `InputPort(s) <#inputport>`_ and other Functions

**Allowed parameters**

===============  ==========================================================================  =======================================================================================================
Allowed field    Data Type                                                                   Description
===============  ==========================================================================  =======================================================================================================
**metadata**     Union[Any, NoneType]                                                        Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str                                                                         The unique (for this Node) id of the function, which will be used in other `Functions <#function>`_ and
                                                                                             the `OutputPorts <#outputport>`_ for its value
**function**     Union[str, NoneType]                                                        Which of the in-build MDF functions (**linear**, etc.). See supported functions:
                                                                                             https://mdf.readthedocs.io/en/latest/api/MDF_function_specifications.html
**args**         Union[Any, NoneType]                                                        Dictionary of values for each of the arguments for the Function, e.g. if the in-built function
                                                                                             is linear(slope),the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}
**value**        Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  If the function is a value expression, this attribute will contain the expression and the function
                                                                                             and args attributes will be None.
===============  ==========================================================================  =======================================================================================================

=========
Parameter
=========
A parameter of the `Node <#node>`_, which can be: 1) a specific fixed **value** (a constant (int/float) or an array) 2) a string expression for the **value** referencing other named `Parameter(s) <#parameter>`_. which may be stateful (i.e. can change value over multiple executions of the `Node <#node>`_); 3) be evaluated by an
inbuilt **function** with **args**; 4) or change from a **default_initial_value** with a **time_derivative**.

**Allowed parameters**

=========================  ==========================================================================  ================================================================================================
Allowed field              Data Type                                                                   Description
=========================  ==========================================================================  ================================================================================================
**metadata**               Union[Any, NoneType]                                                        Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**                     str
**value**                  Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values
**default_initial_value**  Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  The initial value of the parameter, only used when parameter is stateful.
**time_derivative**        Union[str, NoneType]                                                        How the parameter changes with time, i.e. ds/dt. Units of time are seconds.
**function**               Union[str, NoneType]                                                        Which of the in-build MDF functions (linear etc.) this uses, See
**args**                   Union[Any, NoneType]                                                        Dictionary of values for each of the arguments for the function of the parameter,
                                                                                                       e.g. if the in-build function is **linear(slope)**, the args here could be **{"slope": 3}** or
                                                                                                       **{"slope": "input_port_0 + 2"}**
=========================  ==========================================================================  ================================================================================================

**Allowed children**

===============  ===========================================  =============================
Allowed child    Data Type                                    Description
===============  ===========================================  =============================
**conditions**   `ParameterCondition <#parametercondition>`_  Parameter specific conditions
===============  ===========================================  =============================

==================
ParameterCondition
==================
A condition to test on a Node's parameters, which if true, sets the value of this Parameter

**Allowed parameters**

===============  ==========================================================================  ==================================================
Allowed field    Data Type                                                                   Description
===============  ==========================================================================  ==================================================
**id**           str                                                                         A unique identifier for the ParameterCondition
**test**         Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  The boolean expression to evaluate
**value**        Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  The new value of the Parameter if the test is true
===============  ==========================================================================  ==================================================

==========
OutputPort
==========
The `OutputPort <#outputport>`_ is an attribute of a `Node <#node>`_ which exports information to another `Node <#node>`_ connected by an `Edge <#edge>`_

**Allowed parameters**

===============  ================================  ===========================================================================================================================
Allowed field    Data Type                         Description
===============  ================================  ===========================================================================================================================
**metadata**     Union[Any, NoneType]              Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**           str                               Unique identifier for the output port.
**value**        Union[str, NoneType]              The value of the `OutputPort <#outputport>`_ in terms of the `InputPort <#inputport>`_, `Function <#function>`_ values, and
                                                   `Parameter <#parameter>`_ values.
**shape**        Union[Tuple[int, ...], NoneType]  The shape of the output port. This uses the same syntax as numpy ndarray shapes
                                                   (e.g., **numpy.zeros(shape)** would produce an array with the correct shape
**type**         Union[str, NoneType]              The data type of the output sent by a port.
===============  ================================  ===========================================================================================================================

====
Edge
====
An `Edge <#edge>`_ is an attribute of a `Graph <#graph>`_ that transmits computational results from a sender's `OutputPort <#outputport>`_ to a receiver's `InputPort <#inputport>`_.

**Allowed parameters**

=================  ====================  ==========================================================================================================
Allowed field      Data Type             Description
=================  ====================  ==========================================================================================================
**metadata**       Union[Any, NoneType]  Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.
**id**             str                   A unique string identifier for this edge.
**sender**         str                   The **id** of the `Node <#node>`_ which is the source of the edge.
**receiver**       str                   The **id** of the `Node <#node>`_ which is the target of the edge.
**sender_port**    str                   The id of the `OutputPort <#outputport>`_ on the sender `Node <#node>`_, whose value should be sent to the
                                         **receiver_port**
**receiver_port**  str                   The id of the InputPort on the receiver `Node <#node>`_
**parameters**     Union[Any, NoneType]  Dictionary of parameters for the edge.
=================  ====================  ==========================================================================================================
