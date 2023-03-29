#!/usr/bin/env python

from setuptools import setup

# Most of this packages settings are defined in setup.cfg
# FIXME: Not sure of the best way to setup extras_require from setup.cfg
extras = {
    "psyneulink": ["grpcio-tools==1.42.0", "psyneulink"],
    "neuroml": ["pyNeuroML>=0.5.20", "neuromllite>=0.5.2"],
    "tensorflow": ["tensorflow", "keras_visualizer", "pydot"],
    "test": [
        "pytest",
        "pytest-benchmark",
        "pytest-mock",
        "typing_extensions; python_version<'3.8'",
    ],
    "optional": [
        "Sphinx~=3.0",
        "recommonmark>=0.5.0",
        "nbsphinx",
        "sphinx_copybutton",
        "sphinx-rtd-theme",
        "myst_parser",
        "sphinx_markdown_tables",
        "sphinx-autoapi",
        "pytorch-sphinx-theme==0.0.19",
        "sphinxcontrib-versioning",
        "Jinja2<3.1",
        "torchviz",
        "netron",
        "torch>=1.11.0",
        "torchvision<=0.12.0",
        "h5py",
    ],
    "dev": [],
}
extras["all"] = sum(extras.values(), [])
extras["dev"] += extras["test"]

setup(extras_require=extras)
