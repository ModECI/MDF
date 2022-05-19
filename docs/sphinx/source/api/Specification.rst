================================
Specification of ModECI v0.4 RST
================================

**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice.** See `here <https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification>`_ for ongoing discussions.

=====
Model
=====
The top level construct in MDF is Model, which may contain multiple <a href="#graph">Graph</a> objects and model attribute(s)

**Allowed parameters**

==========================  ====================  ===============================================================================================
Allowed field               Data Type             Description
==========================  ====================  ===============================================================================================
**metadata**                Union[Any, NoneType]  *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**                      str                   *A unique identifier for this Model*
**format**                  str                   *Information on the version of MDF used in this file*
**generating_application**  str                   *Information on what application generated/saved this file*
**onnx_opset_version**      Union[str, NoneType]  *The ONNX opset used for any ONNX functions in this model.*
==========================  ====================  ===============================================================================================

**Allowed children**

===============  =================  ======================================================
Allowed child    Data Type          Description
===============  =================  ======================================================
**graphs**       `Graph <#graph>`_  *The collection of graphs that make up the MDF model.*
===============  =================  ======================================================

=====
Graph
=====
A directed graph consisting of Node(s) connected via Edge(s)

**Allowed parameters**

===============  =============================  ===============================================================================================
Allowed field    Data Type                      Description
===============  =============================  ===============================================================================================
**metadata**     Union[Any, NoneType]           *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**           str                            *A unique identifier for this Graph*
**parameters**   Union[Any, NoneType]           *Dictionary of global parameters for the Graph*
**conditions**   Union[ConditionSet, NoneType]  *The ConditionSet stored as dictionary for scheduling of the Graph*
===============  =============================  ===============================================================================================

**Allowed children**

===============  ===============  ================================================================
Allowed child    Data Type        Description
===============  ===============  ================================================================
**nodes**        `Node <#node>`_  *One or more <a href="#node">Node</a>\(s) present in the graph*
**edges**        `Edge <#edge>`_  *Zero or more <a href="#edge">Edge</a>\(s) present in the graph*
===============  ===============  ================================================================

====
Node
====
The values from these are processed via a number of <a href="#function">Function</a>\(s) and one or more final values
are calculated on the <a href="#outputport">OutputPort</a>\(s)

**Allowed parameters**

===============  ====================  ===============================================================================================
Allowed field    Data Type             Description
===============  ====================  ===============================================================================================
**metadata**     Union[Any, NoneType]  *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**           str                   *A unique identifier for the node.*
===============  ====================  ===============================================================================================

**Allowed children**

================  ===========================  ============================================================================================
Allowed child     Data Type                    Description
================  ===========================  ============================================================================================
**input_ports**   `InputPort <#inputport>`_    *Dictionary of the <a href="#inputport">InputPort</a> objects in the Node*
**functions**     `Function <#function>`_      *The <a href="#function">Function</a>\(s) for computation the node*
**parameters**    `Parameter <#parameter>`_    *Dictionary of <a href="#parameter">Parameter</a>\(s) for the node*
**output_ports**  `OutputPort <#outputport>`_  *The <a href="#outputport">OutputPort</a>\(s) containing evaluated quantities from the node*
================  ===========================  ============================================================================================

=========
InputPort
=========
The <a href="#inputport">InputPort</a> is an attribute of a Node which allows external information to be input to the Node

**Allowed parameters**

===============  ================================  ===============================================================================================
Allowed field    Data Type                         Description
===============  ================================  ===============================================================================================
**metadata**     Union[Any, NoneType]              *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**           str                               *The unique (for this Node) id of the input port,*
**shape**        Union[Tuple[int, ...], NoneType]  *The shape of the input port. This uses the same syntax as numpy ndarray shapes
                                                   (e.g., <b>numpy.zeros(shape)</b> would produce an array with the correct shape*
**type**         Union[str, NoneType]              *The data type of the input received at a port.*
===============  ================================  ===============================================================================================

========
Function
========
A single value which is evaluated as a function of values on <a href="#inputport">InputPort</a>\(s) and other Functions

**Allowed parameters**

===============  ==========================================================================  =================================================================================================================
Allowed field    Data Type                                                                   Description
===============  ==========================================================================  =================================================================================================================
**metadata**     Union[Any, NoneType]                                                        *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**           str                                                                         *The unique (for this Node) id of the function, which will be used in other <a href="#function">Function</a>s and
                                                                                             the <a href="#outputport">OutputPort</a>s for its value*
**function**     Union[str, NoneType]                                                        *Which of the in-build MDF functions (<b>linear</b>, etc.). See supported functions:
                                                                                             https://mdf.readthedocs.io/en/latest/api/MDFfunctionspecifications.html*
**args**         Union[Any, NoneType]                                                        *Dictionary of values for each of the arguments for the Function, e.g. if the in-built function
                                                                                             is linear(slope),the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}*
**value**        Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  *If the function is a value expression, this attribute will contain the expression and the function
                                                                                             and args attributes will be None.*
===============  ==========================================================================  =================================================================================================================

=========
Parameter
=========
referencing other named <a href="#parameter">Parameter</a>\(s). which may be stateful (i.e. can change value over multiple executions of the <a href="#node">Node</a>); 3) be evaluated by an
inbuilt <b>function</b> with <b>args</b>; 4) or change from a <b>default_initial_value</b> with a <b>time_derivative</b>.

**Allowed parameters**

=========================  ==========================================================================  ====================================================================================================
Allowed field              Data Type                                                                   Description
=========================  ==========================================================================  ====================================================================================================
**metadata**               Union[Any, NoneType]                                                        *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**                     str                                                                         **
**value**                  Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  *The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values*
**default_initial_value**  Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  *The initial value of the parameter, only used when parameter is stateful.*
**time_derivative**        Union[str, NoneType]                                                        *How the parameter changes with time, i.e. ds/dt. Units of time are seconds.*
**function**               Union[str, NoneType]                                                        *Which of the in-build MDF functions (linear etc.) this uses, See*
**args**                   Union[Any, NoneType]                                                        *Dictionary of values for each of the arguments for the function of the parameter,
                                                                                                       e.g. if the in-build function is <b>linear(slope)</b>, the args here could be <b>{"slope": 3}</b> or
                                                                                                       <b>{"slope": "input_port_0 + 2"}</b>*
=========================  ==========================================================================  ====================================================================================================

**Allowed children**

===============  ===========================================  ===============================
Allowed child    Data Type                                    Description
===============  ===========================================  ===============================
**conditions**   `ParameterCondition <#parametercondition>`_  *Parameter specific conditions*
===============  ===========================================  ===============================

==================
ParameterCondition
==================
A condition to test on a Node's parameters, which if true, sets the value of this Parameter

**Allowed parameters**

===============  ==========================================================================  ====================================================
Allowed field    Data Type                                                                   Description
===============  ==========================================================================  ====================================================
**id**           str                                                                         *A unique identifier for the ParameterCondition*
**test**         Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  *The boolean expression to evaluate*
**value**        Union[EvaluableExpression, List, Dict, ndarray, int, float, str, NoneType]  *The new value of the Parameter if the test is true*
===============  ==========================================================================  ====================================================

==========
OutputPort
==========
connected by an <a href="#edge">Edge</a>

**Allowed parameters**

===============  ================================  =======================================================================================================================================================
Allowed field    Data Type                         Description
===============  ================================  =======================================================================================================================================================
**metadata**     Union[Any, NoneType]              *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**           str                               *Unique identifier for the output port.*
**value**        Union[str, NoneType]              *The value of the <a href="#outputport">OutputPort</a> in terms of the <a href="#inputport">InputPort</a>, <a href="#function">Function</a> values, and
                                                   <a href="#parameter">Parameter</a> values.*
**shape**        Union[Tuple[int, ...], NoneType]  *The shape of the output port. This uses the same syntax as numpy ndarray shapes
                                                   (e.g., <b>numpy.zeros(shape)</b> would produce an array with the correct shape*
**type**         Union[str, NoneType]              *The data type of the output sent by a port.*
===============  ================================  =======================================================================================================================================================

====
Edge
====
<a href="#outputport">OutputPort</a> to a receiver's <a href="#inputport">InputPort</a>.

**Allowed parameters**

=================  ====================  =============================================================================================================================
Allowed field      Data Type             Description
=================  ====================  =============================================================================================================================
**metadata**       Union[Any, NoneType]  *Optional metadata field, an arbitrary dictionary of string keys and JSON serializable values.*
**id**             str                   *A unique string identifier for this edge.*
**sender**         str                   *The <b>id</b> of the <a href="#node">Node</a> which is the source of the edge.*
**receiver**       str                   *The <b>id</b> of the <a href="#node">Node</a> which is the target of the edge.*
**sender_port**    str                   *The id of the <a href="#outputport">OutputPort</a> on the sender <a href="#node">Node</a>, whose value should be sent to the
                                         <b>receiver_port</b>*
**receiver_port**  str                   *The id of the InputPort on the receiver <a href="#node">Node</a>*
**parameters**     Union[Any, NoneType]  *Dictionary of parameters for the edge.*
=================  ====================  =============================================================================================================================
