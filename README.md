[![Build Status](https://travis-ci.com/ModECI/MDF.svg?branch=main)](https://travis-ci.com/ModECI/MDF)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ModECI Model Description Format (MDF)

**Work in progress!!! See the [open issues related to the specification](https://github.com/ModECI/MDF/issues?q=is%3Aissue+is%3Aopen+label%3Aspecification) or go [here](https://modeci.github.io/Website/About.html) to get in contact regarding MDF.**

*The MDF format was first proposed following a meeting organised at Princeton in July 2019 by Russ Poldrack of the Center for Reproducible Neuroscience (CRN) at Stanford and the [Brain Imaging Data Standard (BIDS)](https://bids.neuroimaging.io/) initiative. For more on the previous work in this area, see [here](https://github.com/OpenSourceBrain/PsyNeuLinkShowcase/tree/master/BIDS-MDF).*


Overview
--------

The purpose of the MDF is to provide a standard, JSON-based format (though other serializations like YAML, HDF5 are envisioned)
for describing computational models of brain and/or mental function.  The goal is to provide a common exchange format that allows models created in one
environment that supports the standard to be expressed in a form - and in sufficient detail - that it can be
imported into another modeling environment that supports the standard, and then executed in that environment with
identical results (within a certain tolerance), and/or integrated with other models in that environment.

The format assumes that models can be expressed as graphs, in which each **node** is a computational component, and **edges**
specify connections between them that (at least partially) determine the flow of computation.  In this respect,
they are similar to more generic forms of computational graphs.  However, the standard supports the expression of
more specific elements within nodes and edges that are central to models of brain function (e.g., the inclusion of
**ports** in nodes that are dedicated to processing the input and/or output of a node; and the **weight** and/or **function**
of an edge that allows it to do more than simply relay information unmodified from one node to another.  Finally, the
standard allows elements that are specific to a particular modeling environment to be expressed in a circumscribed
form, so that the format can be used to "serialize" models from that environment, and makes them accessible to other
environments that specifically support those constructs from the origin environment.  This latter capability provides
a way not only for extending the standard to accommodate the specific needs of individual environments, but also as a
path toward extending the standard: recurring extensions that serve similar purposes help identify targets for the
definition of new components of the core standard itself.

The format is currently under development, and should be considered a prototype for a more complete standard.
Among the targets for extension are standards for specifying process control flow.  At present, to execute a model
specified in the format, the environment must either assume that process control flow is determined by the structure
of the graph (i.e., edges are directed, the graph is acyclic, and nodes execute in the order of dependencies), or
that process control flow is specified in environment-specific entries.  A goal of further development is to provide
standard ways of expressing both time-based and condition-based specifications for execution timing and order. Progress on this topic is outlined under [Conditions](#conditions) and [Graphs](#graphs) below.

Basic Constructs
----------------

MDF assumes that a model can be expressed as a **graph** made up of the following four basic types of
objects:

* **nodes** - the basic computational elements of a model;

* **edges** - directed connections between **nodes** that help determine the flow of computations;

* **ports** - specifically **input_ports** and **output_ports**: components that belong to a **node** and mediate its **functions** and its incoming and outgoing
              **edge(s)**;

* **functions** - can belong to any of the other components, and specify the particular computation(s) carried
                  out by that component.

In addition, the standard allows the specification of **parameters** for **nodes**, **edges** and **ports**,
**arguments** for **functions**, and run **conditions**.  The format for specifying these components is described below.

Overall structure
-----------------

MDF is a hierarchically organized format using JSON-compliant syntax, that can be used to describe one or
more models in a single text file. Equivalent (lossless) serializations of the format in YAML and other
format will be supported, but the language elements are described below using JSON for clarity.

The outermost level of the specification is a dictionary with a single entry named
``graphs``, the value of which is a list of  **graph** objects.  Each **graph** object is a dictionary that
defines a single model.  Each **graph** dictionary must have at least two entries, named ``nodes`` and ``edges``,
each of which is a dictionary with entries describing the **nodes** and **edges** of the graph, respectively.
Each entry in the **node** and **edge** dictionaries must be another dictionary that contains object-specific
entries.   The ``nodes`` dictionary, in addition to entries describing nodes, can also contain entries that are
themselves graphs, which can be used to describe hierarchically-structured models.  In addition to the ``nodes``
and ``edges`` entries, a ``graph`` object can have additional entries, some of which are generic (such as the ``names``
entry in the example below;  see ``Entries common to all objects`` for a full listing);  it can also include a
``parameters`` entry, that contains specifications required by specific environments.  The following provides an
example of the overall scheme of a MDF specification:

[//]: # "Should we add schedulers as a standard/generic parameter type to deal with aspects of computational flow
that are not determined by the structure of the graph itself??;  controller is another example of this (that we have
 added below to show how it would be done... but should probably be discussed further"

    {
        "graphs": [
            {
                "name": {                                   # Optional generic entry
                    "Example Model"
                }
                "nodes": {                                  # Required
                    ... dictionaries for node objects
                }
                "edges": {                                  # Required
                    ... dictionaries for edge objects
                }
                "parameters": {                             # Optional (for parameters of the graph)
                    ... dictionaries for graph parameters

                }
                "conditions": {                             # Optional
                    ... dictionaries for run conditions
                }
             }
            {
                "name": {
                    "Another Model"
                }
                "nodes": {
                    ... dictionaries for node objects
                }
                "edges": {
                    ... dictionaries for edge objects
                }
                "parameters": {
                    ... dictionaries for graph parameters
                }
                "conditions": {
                    ... dictionaries for run conditions
                }
             }
         ]
     }

In addition to these standard entries, a modeling environment that requires objects or other information to be
specified that is not (yet) supported by the standard, can include entries for such information using a name of the
environment as the key.  In the examples below, PsyNeuLink (PNL) is used to demonstrate such environment-specific
entries.

Entries common to all objects
-----------------------------

The following entries can be used in any MDF object (using the strings shown below as their keys):

* ``name`` : a label for the object

* ``type`` : a dictionary used to describe the type of object specified by the entry.  The ``generic`` entry
  contains types supported by the standard (e.g., certain common types of functions;  see XXX);  in addition, the
  type can include environment-specific entries.  For example, the following specifies a graph that has two **nodes**,
  one of which is a nested **graph**, as specified by its ``type`` entry:

      "graphs": [
          {
              "nodes": {
                  "Processing Unit": {
                      ... node specifications
                  }
                  "A Nested Graph": {
                      "type": {
                          "generic": "graph"
                          "PNL": "Composition"
                      }
                  }
              }

  Here, the ``generic`` entry of ``type`` is used to specify that it is a graph (recognized by the MDF standard),
  while the ``PNL`` entry is used to specify the PNL-specific designation of a graph ("Composition").


[//]: # "JDC: ??IS THERE ANY REASON TO DISTINGUISH THESE, OR SHOULD THESE BE COMBINED TO MAKE IT SIMPLER"

[//]: # "KDM: No, there's no reason on my part, that was just how the original spec listed it, so I implemented it that way.
I think it would be better to choose one name"

* ``parameters`` (for non-**function** entries) or ``args`` (for **functions** entries) : this is used to specificy
  attributes of the object.  For all objects other than **functions**, these are called ``parameters``, and for
  **functions** they are called ``args``.  For example, the following contains an entry for a **node** ("Processing
  Unit"), that has specifications for one of its standard **parameters** (``initializer``), as well as a ``functions``
  entry that specifies the list of functions for the node well as the **type** and **args** for each function.  For
  example, the node for **Processing Unit** in the example below has a single parameter specified, and a single
  function that is specified as follows:

    [//]: # "JDC: Are there any standard-defined parameters for graphs, nodes?  Here, we assume that ``initializer`` is such
    parameter and included it in the example as such.  However, I'm not sure we ever discussed or agreed to any such
    standard node parameters.  If so, we should list them somehwher.  If are not any, then ``parameters`` should be
    removed from this example, and ``initializer`` (if retained) should be moved to the ``PNL`` entry in the next
    example)"

        "nodes": {
            "Processing Unit": {
                "parameters": {
                    "initializer": [[0]]
                }
                "functions": [
                    {
                        "type": "Linear"
                        "args": {
                            "bounds": null,
                            "intercept": 0.0,
                            "slope": 1.0
                        }
                    }
                ]
            }
        }

    [//]: # "JDC: Are there any standard-defined parameters for graphs, nodes?  Here, we assume that ``initializer`` is such
    parameter and included it in the example as such.  However, I'm not sure we ever discussed or agreed to any such
    standard node parameters.  If so, we should list them somehwher.  If are not any, then ``parameters`` should be
    removed from this example, and ``initializer`` (if retained) should be moved to the ``PNL`` entry in the next
    example)"

    [//]: # "JDC: The example below is predicated on the assumption that ``controller`` should be a standard parameter for a
    graph;  if not, can delete the following text and example, but might want to include another one that shows how
    a node can be referenced by another entry"

    The following example specifies a standard **parameter** for a graph -- ``controller`` -- by referencing the
    **node** that serves as the controller for the model:

       "graphs": [{
            {
                "parameters": {
                    "controller":"My controller"
                }
                nodes:{
                    "My Contoller": {
                        ... node ``functions`` and ``parameters`` entries here
                    }
                }
            }]
        }





The ``parameters`` entry of a **node** (or ``args`` entry of a *function**) can also include a subdictionary of
environment-specific parameter-value (or arg-value) pairs.  For example, the ``parameters`` entry below adds entries
for two parameters -- ``execution_count`` and ``has_initializers`` -- that are specific to the PsyNeuLink (PNL)
environment:

      "parameters": {
            "initializer": [[0]]
            "PNL": {                      # This is a subdictionary of PNL-specific parameters and their values
                "execution_count": 0,
                "has_initializers": false,
            }
        }

Object-specific entries
-----------------------

#### *Nodes, edges, and ports*

These objects can all include a ``functions`` entry, that specifies one or more **functions** that belong to the object:

* ``functions`` : a list of **function** objects. As noted above, each of these can have an ``args`` entry that,
  in turn, can contain entries that are either simple parameter-value pairs (as in the example above), or a dictionary
  that provides more detailed information about an argument, including its **type** and **value** as in the example
  below:

        "functions": [
            {
                "name": "Linear Function-1",
                "type": {
                    "generic": "Linear"
                "args": {
                    "intercept": {
                        "source": "A.input_ports.intercept",
                        "type": "float",
                        "value": 2.0
                    },
                    "slope": {
                        "source": "A.input_ports.slope",
                        "type": "float",
                        "value": 5.0
                    }
                },
                }
            }
        ]

    Note that the ``source`` entry can reference a ``port`` used to determine the value of the argument when the
    model is executed (see [ports](#ports)] below);  the reference must use dot-delimited notation, beginning with the
    name of the **node** to which the **port** belongs (``A`` in the example above), followed by ``input_ports
    `` entry, and then name of the input_port from which the argument should receive its value.

    [//]: # "??PNL ALLOWS THE NAME OF THE NODE (MECHANISM) TO BE USED IN PLACE OF THE INPUTPORT.  THIS IS SEEMS TO BE
    SUPPORTED FOR EDGES (SEE BELOW).  SHOULD WE DO THE SAME FOR SOURCE SPECIFICATIONS?"

    [//]: # "KDM: maybe, but I'd wait and see what others say. Suppose there might be a use for a 'combined' input port with a
     name that isn't the same as the arg name"

    [//]: # "JDC: Not sure I follow;  should discuss"

#### Non-**graph** **nodes**

These can have one or both of the following entries that specify the **node**'s **ports**

* ``input_ports`` : a list of **port** objects that can be referenced by the ``source`` entry of an ``args`` dictionary
  (see [parameters and args](#paramters-or-args) above), or the ``sender_port`` entry of an ``edges`` dictionary (see
   [edges](#edges) below).

* ``output_ports`` : a list of **port** objects that can be referenced by the ``receiver_port`` entry of an ``edges
`` dictionary (see [edges](#edges) below).

#### **Ports**

These are used to specify entries in the ``input_ports`` and/or ``output_ports`` entry of a **node**, and can
be referenced, respectively, by the ``sender_port`` and ``receiver_port`` entries of an **edge**, and by ``source``
in the ``args`` entry of a **function**.

* ``dtype`` : the type of the input received by a **port** listed in the ``input_ports`` entry of a **node**,
  or of the output sent by a **port** listed in the ``output_ports`` entry of a **node**;  this uses the same
  syntax as [numpy.dtype](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html).

* ``shape`` : the shape of the input or output of a **port**. This uses the same syntax as numpy ndarray shapes
  (e.g., `numpy.zeros(<shape>)` would produce an array with the correct shape).

#### **Edges**

These must contain the following entries that specify edge's sender and receiver. This can include the **port** to
which the edge connects on a **node**; if either of the **port** specifications is omitted for an **edge**, then
the environment must not use ports, or it must be able to assign a suitable default for the referenced ``sender``
or ``receiver`` **node**.

* ``sender`` : the name of its source **node**.

* ``sender_port`` : the name of the **port** on the ``sender`` **node** to which it connects.

* ``receiver`` : the name of its destination **node**.

* ``receiver_port`` : the name of the **port** on the ``receiver`` **node** to which it connects.

#### **Conditions**

These must include a **type**, among options of an accepted library of
conditions (TBD), as well as any **args** necessary to
determine satisfaction of the relevant condition.

* ``type`` : the name of the condition
* ``args`` : named arguments used to evaluate the condition

#### **Graphs**

These objects may include **condition**s to specify a non-default
pattern of execution.

* ``conditions`` : a dictionary with entries
  * ``node_specific`` : a dictionary mapping **node**s to any non-default run **condition**s
  * ``termination`` : a dictionary mapping time scales of model execution to **condition**s indicating when they end

  Each node or time scale may have exactly one condition, however binary
  conditions like `And`, `Or`, and `Not`, may combine multiple
  conditions into one in an unambiguous way. This is sufficient to
  describe a wide variety of execution patterns that go beyond an
  implied ordering based on the structure of the graph, as in the
  example below:

        "conditions": {
            "node_specific": {
                "A": {
                    "type": "Always",
                    "args": {}
                },
                "B": {
                    "type": "EveryNCalls",
                    "args": {
                        "dependency": "A",
                        "n": 1
                    }
                },
                "C": {
                    "type": "EveryNCalls",
                    "args": {
                        "dependency": "A",
                        "n": 3
                    }
                }
            },
            "termination": {
                "trial": {
                    "type": "And",
                    "args": {
                        "dependencies": [
                            {
                                "type": "AfterNCalls",
                                "args": {
                                    "dependency": "C",
                                    "n": 2
                                }
                            },
                            {
                                "type": "JustRan",
                                "args": {
                                    "dependency": "A"
                                }
                            }
                        ]
                    }
                }
            }
        },

    In this example, node `C` runs only while its condition
    `EveryNCalls` with `dependency` `A` and `n = 3` is true. This
    results in `C` executing only every third time `A` executes. The
    `trial` finishes when the trial termination condition - `And` - is
    satisfied. This `And` is comprised of an `AfterNCalls` and a
    `JustRan`, creating a condition that is only true after both `C` has
    run at least twice and `A` has run in the last `time step`.
