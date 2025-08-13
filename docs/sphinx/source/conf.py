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
copyright = "2023, ModECI project"
author = "ModECI project"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
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
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)

# Exclusions
# To exclude a module, use autodoc_mock_imports. Note this may increase build time, a lot.
# (Also, when installing on readthedocs.org, we omit installing Tensorflow and
# Tensorflow Probability so mock them here instead.)
autodoc_mock_imports = [
    "modeci_mdf.version",
    "modeci_mdf.interfaces.pytorch.mod_torch_builtins",
    "modeci_mdf.interfaces.pytorch.builtins",
]

# To exclude a class, function, method or attribute, use autodoc-skip-member. (Note this can also
# be used in reverse, ie. to re-include a particular member that has been excluded.)
# 'Private' and 'special' members (_ and __) are excluded using the Jinja2 templates; from the main
# doc by the absence of specific autoclass directives (ie. :private-members:), and from summary
# tables by explicit 'if-not' statements. Re-inclusion is effective for the main doc though not for
# the summary tables.
# See: https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion
def autodoc_skip_member_callback(app, what, name, obj, skip, options):
    # Classes and functions to exclude
    exclusions = "MdfBase"

    # Things to reinclude
    inclusions = ()

    if name in exclusions:
        return True
    elif name in inclusions:
        return False
    else:
        return skip


def setup(app):
    # Entry point to autodoc-skip-member
    app.connect("autodoc-skip-member", autodoc_skip_member_callback)


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
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
}

from sphinx.ext.autodoc import ClassLevelDocumenter, AttributeDocumenter


def add_directive_header(self, sig):
    ClassLevelDocumenter.add_directive_header(self, sig)


AttributeDocumenter.add_directive_header = add_directive_header


import os

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_css_files = ["rtd-custom.css"]  # Override some CSS settings
