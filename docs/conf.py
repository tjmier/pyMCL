# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyMCL'
copyright = '2024, Tristen Jay Mier'
author = 'Tristen Jay Mier'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",     # Read the Docs theme
    "sphinx.ext.autodoc",   # Include documentation from docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.napoleon",  # Parse Google-style docstrings
    "sphinx_copybutton",    # Add copy buttons to code blocks
    "nbsphinx",             # Execute Jupyter notebooks in documentation
    "myst_parser",          # Use MyST markdown parser
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_context = {
    "sidebar_external_links_caption": "Links",
    "sidebar_external_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source code',
            "https://github.com/tjmier/pyMCL",
        ),
        (
            '<i class="fa fa-bug fa-fw"></i> Issue tracker',
            "https://github.com/tjmier/pyMCL/issues",
        ),
    ],
}
