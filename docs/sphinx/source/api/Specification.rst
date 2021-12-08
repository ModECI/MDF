=======================================
Specification of ModECI v0.3
=======================================
**Note: the ModECI MDF specification is still in development! Subject to change without (much) notice. See [here](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) for ongoing discussions.**
=====
Model
=====
The top level Model containing `Graph <#graph>`_s consisting of `Node <#node>`_s connected via `Edge <#edge>`_s.

Allowed parameters
==================

==========================  ===========  ===========================================================
Allowed field               Data Type    Description
==========================  ===========  ===========================================================
**format**                  str          *Information on the version of MDF used in this file*
**generating_application**  str          *Information on what application generated/saved this file*
**metadata**                dict         *Dict of metadata for the model element*
**id**                      str          *Unique ID of element*
**notes**                   str          *Human readable notes*
==========================  ===========  ===========================================================

Allowed children
================

===============  =================  ==============================================
Allowed child    Data Type          Description
===============  =================  ==============================================
**graphs**       `Graph <#graph>`_  *The list of `Graph <#graph>`_s in this Model*
===============  =================  ==============================================

=====
Graph
=====
A directed graph consisting of `Node <#node>`_s connected via `Edge <#edge>`_s.

Allowed parameters
==================

===============  ===============================  =================================================================
Allowed field    Data Type                        Description
===============  ===============================  =================================================================
**parameters**   dict                             *Dict of global parameters for the Graph*
**conditions**   `ConditionSet <#conditionset>`_  *The `ConditionSet <#conditionset>`_ for scheduling of the Graph*
**metadata**     dict                             *Dict of metadata for the model element*
**id**           str                              *Unique ID of element*
**notes**        str                              *Human readable notes*
===============  ===============================  =================================================================

Allowed children
================

===============  ===============  ============================================================
Allowed child    Data Type        Description
===============  ===============  ============================================================
**nodes**        `Node <#node>`_  *The `Node <#node>`_s present in the Graph*
**edges**        `Edge <#edge>`_  *The `Edge <#edge>`_s between `Node <#node>`_s in the Graph*
===============  ===============  ============================================================

============
ConditionSet
============
Specifies the non-default pattern of execution of `Node <#node>`_s

Allowed parameters
==================

=================  ===========  ========================================================================
Allowed field      Data Type    Description
=================  ===========  ========================================================================
**node_specific**  dict         *The `Condition <#condition>`_s corresponding to each `Node <#node>`_*
**termination**    dict         *The `Condition <#condition>`_s that indicate when model execution ends*
**metadata**       dict         *Dict of metadata for the model element*
=================  ===========  ========================================================================

====
Node
====
A self contained unit of evaluation receiving input from other Nodes on `InputPort <#inputport>`_s. The values from these are processed via a number of Functions and one or more final values are calculated on the `OutputPort <#outputport>`_s

Allowed parameters
==================

===============  ===========  ========================================
Allowed field    Data Type    Description
===============  ===========  ========================================
**metadata**     dict         *Dict of metadata for the model element*
**id**           str          *Unique ID of element*
**notes**        str          *Human readable notes*
===============  ===========  ========================================

Allowed children
================

================  ===========================  ================================================================================
Allowed child     Data Type                    Description
================  ===========================  ================================================================================
**input_ports**   `InputPort <#inputport>`_    *The `InputPort <#inputport>`_s into the Node*
**functions**     `Function <#function>`_      *The `Function <#function>`_s for the Node*
**parameters**    `Parameter <#parameter>`_    *The `Parameter <#parameter>`_s of the Node*
**output_ports**  `OutputPort <#outputport>`_  *The `OutputPort <#outputport>`_s containing evaluated quantities from the Node*
================  ===========================  ================================================================================

=========
InputPort
=========
The InputPort is an attribute of a `Node <#node>`_ which allows external information to be input to the `Node <#node>`_

Allowed parameters
==================

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
A single value which is evaluated as a function of values on `InputPort <#inputport>`_s and other Functions

Allowed parameters
==================

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
A Parameter of the `Node <#node>`_, which can have a specific value (a constant or a string expression referencing other Parameters), be evaluated by an inbuilt function with args, or change from a default`initial <#initial>`_value with a time`derivative <#derivative>`_

Allowed parameters
==================

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
The OutputPort is an attribute of a `Node <#node>`_ which exports information to another `Node <#node>`_ connected by an `Edge <#edge>`_

Allowed parameters
==================

===============  ===========  ==========================================================================================================
Allowed field    Data Type    Description
===============  ===========  ==========================================================================================================
**value**        str          *The value of the OutputPort in terms of the `InputPort <#inputport>`_ and `Function <#function>`_ values*
**metadata**     dict         *Dict of metadata for the model element*
**id**           str          *Unique ID of element*
**notes**        str          *Human readable notes*
===============  ===========  ==========================================================================================================

====
Edge
====
An Edge is an attribute of a `Graph <#graph>`_ that transmits computational results from a sender's `OutputPort <#outputport>`_ to a receiver's `InputPort <#inputport>`_

Allowed parameters
==================

=================  ===========  ====================================================================================================================================
Allowed field      Data Type    Description
=================  ===========  ====================================================================================================================================
**parameters**     dict         *Dict of parameters for the Edge*
**sender**         str          *The id of the `Node <#node>`_ which is the source of the Edge*
**receiver**       str          *The id of the `Node <#node>`_ which is the target of the Edge*
**sender_port**    str          *The id of the `OutputPort <#outputport>`_ on the sender `Node <#node>`_, whose value should be sent to the receiver`port <#port>`_*
**receiver_port**  str          *The id of the `InputPort <#inputport>`_ on the receiver `Node <#node>`_*
**metadata**       dict         *Dict of metadata for the model element*
**id**             str          *Unique ID of element*
**notes**          str          *Human readable notes*
=================  ===========  ====================================================================================================================================
