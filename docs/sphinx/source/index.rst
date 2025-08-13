.. modeci-mdf documentation master file, created by
   sphinx-quickstart on Thu May 13 17:09:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/ModECI/MDF



.. |logo| image:: ../../sphinx/images/logo_dark_bg.png
    :width: 15%
    :target: http://modeci.org

|logo| ModECI Model Description Format (MDF)
============================================

MDF is an open source, community-supported standard and associated library
of tools for expressing computational models in a form that allows them to be exchanged
between diverse programming languages and execution environments. The overarching aim is
to provide a common format for models across **computational neuroscience, cognitive science and machine learning**.

It consists of a
specification for expressing models in serialized form (currently JSON, YAML or BSON
representations, though others such as HDF5 are planned) and a set of Python
tools for implementing a model described using MDF. The serialized formats can be used
when importing a model into a supported target environment to execute it; and,
conversely, when exporting a model built in a supported environment so that it can be
re-used in other environments.



.. toctree::
   :maxdepth: 1
   :caption: Contents

   api/Introduction
   api/QuickStart
   api/MDFpaper
   api/Installation
   api/Contributing
   api/Contributors


.. toctree::
   :maxdepth: 2
   :caption: Specification

   api/Specification

.. toctree::
   :maxdepth: 1
   :caption: Examples

   api/export_format/MDF/MDF.md

.. toctree::
   :maxdepth: 1
   :caption: Export Formats

   api/export_format/ACT-R/ACT-R.md
   api/export_format/NeuroML/NeuroML.md
   api/export_format/ONNX/ONNX.md
   api/export_format/PsyNeuLink/PsyNeuLink.md
   api/export_format/PyTorch/PyTorch.md
   api/export_format/Quantum/Quantum.md
   api/export_format/WebGME/WebGME.md

.. toctree::
   :maxdepth: 2
   :caption: Functions

   api/MDF_function_specifications


.. toctree::
   :caption: API Reference

   api/_autosummary/modeci_mdf


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
