================================
Specification of ModECI v0.3 RST
================================

**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice.** See `here <https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification>`_ for ongoing discussions.

=====
Model
=====
The top level Model containing Graphs consisting of Nodes connected via Edges.

**Allowed parameters**

==========================  ===========  ===========================================================
Allowed field               Data Type    Description
==========================  ===========  ===========================================================
**format**                  str          *Information on the version of MDF used in this file*
**generating_application**  str          *Information on what application generated/saved this file*
**metadata**                dict         *Dict of metadata for the model element*
**id**                      str          *Unique ID of element*
**notes**                   str          *Human readable notes*
==========================  ===========  ===========================================================

**Allowed children**

===============  =================  ==================================
Allowed child    Data Type          Description
===============  =================  ==================================
**graphs**       `Graph <#graph>`_  *The list of Graphs in this Model*
===============  =================  ==================================

=====
Graph
=====
A directed graph consisting of Nodes connected via Edges.

**Allowed parameters**

===============  ===============================  ==============================================
Allowed field    Data Type                        Description
===============  ===============================  ==============================================
**parameters**   dict                             *Dict of global parameters for the Graph*
**conditions**   `ConditionSet <#conditionset>`_  *The ConditionSet for scheduling of the Graph*
**metadata**     dict                             *Dict of metadata for the model element*
**id**           str                              *Unique ID of element*
**notes**        str                              *Human readable notes*
===============  ===============================  ==============================================

**Allowed children**

===============  ===============  ======================================
Allowed child    Data Type        Description
===============  ===============  ======================================
**nodes**        `Node <#node>`_  *The Nodes present in the Graph*
**edges**        `Edge <#edge>`_  *The Edges between Nodes in the Graph*
===============  ===============  ======================================

============
ConditionSet
============
Specifies the non-default pattern of execution of Nodes

**Allowed parameters**

=================  ===========  ========================================================
Allowed field      Data Type    Description
=================  ===========  ========================================================
**node_specific**  dict         *The Conditions corresponding to each Node*
**termination**    dict         *The Conditions that indicate when model execution ends*
**metadata**       dict         *Dict of metadata for the model element*
=================  ===========  ========================================================

====
Node
====
A self contained unit of evaluation receiving input from other Nodes on InputPorts. The values from these are processed via a number of Functions and one or more final values are calculated on the OutputPorts

**Allowed parameters**

===============  ===========  ========================================
Allowed field    Data Type    Description
===============  ===========  ========================================
**metadata**     dict         *Dict of metadata for the model element*
**id**           str          *Unique ID of element*
**notes**        str          *Human readable notes*
===============  ===========  ========================================

**Allowed children**

================  ===========================  ===============================================================
Allowed child     Data Type                    Description
================  ===========================  ===============================================================
**input_ports**   `InputPort <#inputport>`_    *The InputPorts into the Node*
**functions**     `Function <#function>`_      *The Functions for the Node*
**parameters**    `Parameter <#parameter>`_    *The Parameters of the Node*
**output_ports**  `OutputPort <#outputport>`_  *The OutputPorts containing evaluated quantities from the Node*
================  ===========================  ===============================================================

=========
InputPort
=========
The InputPort is an attribute of a Node which allows external information to be input to the Node

**Allowed parameters**

===============  ===========  ===============================================================================
Allowed field    Data Type    Description
===============  ===========  ===============================================================================
**shape**        str          *The shape of the variable (note: there is limited support for this so far...)*
**type**         str          *The type of the variable (note: there is limited support for this so far *
**metadata**     dict         *Dict of metadata for the model element*
**id**           str          *Unique ID of element*
**notes**        str          *Human readable notes*
===============  ===========  ===============================================================================

========
Function
========
A single value which is evaluated as a function of values on InputPorts and other Functions

**Allowed parameters**

===============  ===========  =====================================================================================================================================================================================
Allowed field    Data Type    Description
===============  ===========  =====================================================================================================================================================================================
**function**     dict         *Which of the in-build MDF functions (linear etc.) this uses*
**value**        str          *evaluable expression*
**args**         dict         *Dictionary of values for each of the arguments for the Function, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}*
**id**           str          *Unique ID of element*
**metadata**     dict         *Dict of metadata for the model element*
**notes**        str          *Human readable notes*
===============  ===========  =====================================================================================================================================================================================

=========
Parameter
=========
A Parameter of the Node, which can have a specific value (a constant or a string expression referencing other Parameters), be evaluated by an inbuilt function with args, or change from a defaultinitialvalue with a timederivative

**Allowed parameters**

=========================  ===================  ======================================================================================================================================================================================================
Allowed field              Data Type            Description
=========================  ===================  ======================================================================================================================================================================================================
**default_initial_value**  EvaluableExpression  *The initial value of the parameter*
**value**                  EvaluableExpression  *The next value of the parameter, in terms of the inputs, functions and PREVIOUS parameter values*
**time_derivative**        str                  *How the parameter with time, i.e. ds/dt. Units of time are seconds.*
**function**               str                  *Which of the in-build MDF functions (linear etc.) this uses*
**args**                   dict                 *Dictionary of values for each of the arguments for the function of the parameter, e.g. if the in-build function is linear(slope), the args here could be {"slope":3} or {"slope":"input_port_0 + 2"}*
**metadata**               dict                 *Dict of metadata for the model element*
**id**                     str                  *Unique ID of element*
**notes**                  str                  *Human readable notes*
=========================  ===================  ======================================================================================================================================================================================================

==========
OutputPort
==========
The OutputPort is an attribute of a Node which exports information to another Node connected by an Edge

**Allowed parameters**

===============  ===========  ===========================================================================
Allowed field    Data Type    Description
===============  ===========  ===========================================================================
**value**        str          *The value of the OutputPort in terms of the InputPort and Function values*
**metadata**     dict         *Dict of metadata for the model element*
**id**           str          *Unique ID of element*
**notes**        str          *Human readable notes*
===============  ===========  ===========================================================================

====
Edge
====
An Edge is an attribute of a Graph that transmits computational results from a sender's OutputPort to a receiver's InputPort

**Allowed parameters**

=================  ===========  =============================================================================================
Allowed field      Data Type    Description
=================  ===========  =============================================================================================
**parameters**     dict         *Dict of parameters for the Edge*
**sender**         str          *The id of the Node which is the source of the Edge*
**receiver**       str          *The id of the Node which is the target of the Edge*
**sender_port**    str          *The id of the OutputPort on the sender Node, whose value should be sent to the receiverport*
**receiver_port**  str          *The id of the InputPort on the receiver Node*
**metadata**       dict         *Dict of metadata for the model element*
**id**             str          *Unique ID of element*
**notes**          str          *Human readable notes*
=================  ===========  =============================================================================================
