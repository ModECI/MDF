# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys

# Docs require Python 3.6+ to generate
from pathlib import Path

DIR = Path(__file__).parent.parent.parent.resolve()
BASEDIR = DIR.parent

sys.path.append(str(BASEDIR / "src"))

# -- Project information -----------------------------------------------------

project = "modeci-mdf"
copyright = "2021, ModECI project"
author = "ModECI project"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "sphinx_markdown_tables",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "myst_parser",
    "sphinx.ext.githubpages",
]

autodoc_member_order = "bysource"
autosummary_generate = True
master_doc = "index"
napoleon_use_ivar = True
autosectionlabel_prefix_document = True

html_logo = "../../sphinx/images/logo_dark_bg.png"

# Source Suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
    "navigation_depth": -1,
    "includehidden": True,
}

# Output file base name for HTML help builder.
htmlhelp_basename = "ModECI_MDFdoc"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
}

from sphinx.ext.autodoc import ClassLevelDocumenter, InstanceAttributeDocumenter


def add_directive_header(self, sig):
    ClassLevelDocumenter.add_directive_header(self, sig)


InstanceAttributeDocumenter.add_directive_header = add_directive_header
