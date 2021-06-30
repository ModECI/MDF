.. modeci-mdf documentation master file, created by
   sphinx-quickstart on Thu May 13 17:09:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/ModECI/MDF



:tocdepth: 5

.. |logo| image:: _static/logo_light_bg.png
    :width: 10%
    :target: http://modeci.org


|logo| Documentation
----------------------

MDF is intended to be an open source, community-supported standard and associated library
of tools for expressing computational models in a form that allows them to be exchanged
between diverse programming languages and execution environments. It consists of a
specification for expressing models in a serialized format (currently a JSON
representation, though others such as YAML and HDF5 are planned) and a set of Python
tools for implementing a model described using MDF. The serialized format can be used
when importing a model into a supported target environment to execute it; and,
conversely, when exporting a model built in a supported environment so that it can be
re-used in other environments.



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   api/Introduction

.. toctree::
   :maxdepth: 2
   :caption: Function:

   api/MDF_function_specifications

.. toctree::
   :caption: API Reference:

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
