#!/usr/bin/env python

from setuptools import setup

# Most of this packages settings are defined in setup.cfg
# FIXME: Not sure of the best way to setup extras_require from setup.cfg
extras = {
    "test": [
        "pytest",
        "pytest-benchmark",
        "typing_extensions; python_version<'3.8'",
    ],
    "docs": [
        "Sphinx~=3.0",
        "recommonmark>=0.5.0",
        "sphinx_book_theme==0.38.0",
        "nbsphinx",
        "sphinx_copybutton",
    ],
    "dev": [],
}
extras["all"] = sum(extras.values(), [])
extras["dev"] += extras["test"]

setup(extras_require=extras)
