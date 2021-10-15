"""
MDF is intended to be an open source, community-supported standard and associated library of tools for expressing
computational models in a form that allows them to be exchanged between diverse programming languages and execution
environments. The MDF Python API can be used to create or load an MDF model for inspection and validation. It also
includes a basic execution engine for simulating models in the format. However, this is not intended as a general
purpose simulation environment, nor is MDF intended as a programming language.  Rather, the primary purpose of the
Python API is to facilitate and validate the exchange of models between existing environments that serve different
communities. Accordingly, these Python tools include bi-directional support for importing to and exporting from
widely-used programming environments in a range of disciplines, and for easily extending these to other environments.
"""

# Version of the specification for MDF
MODECI_MDF_VERSION = "0.2"

# Version of the python module. Use MDF version here and just change minor version
__version__ = "%s.1" % MODECI_MDF_VERSION
