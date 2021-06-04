.. modeci-mdf documentation master file, created by
   sphinx-quickstart on Thu May 13 17:09:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. figure:: /Image/logo_light_bg.png
   :scale: 40%
Welcome to modeci-mdf's documentation!
======================================
The purpose of the MDF is to provide a standard, JSON-based format (though other serializations like YAML, HDF5 are
envisioned) for describing computational models of brain and/or mental function. The goal is to provide a common exchange
format that allows models created in one environment that supports the standard to be expressed in a form - and in sufficient
detail - that it can be imported into another modeling environment that supports the standard, and then executed in that environment
with identical results (within a certain tolerance), and/or integrated with other models in that environment.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 4
   :caption: ReadMeFile:

   api/README


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
