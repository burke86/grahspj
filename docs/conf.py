from __future__ import annotations

import os
import sys
from datetime import datetime

ROOT = os.path.abspath("..")
SRC = os.path.join(ROOT, "src")
DOC_EXT = os.path.abspath("_ext")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if DOC_EXT not in sys.path:
    sys.path.insert(0, DOC_EXT)

project = "JAXSEDFit"
author = "JAXSEDFit contributors"
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "filter_registry",
    "nbsphinx",
    "nbsphinx_link",
]

autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Keep docs build light and robust on RTD.
autodoc_mock_imports = [
    "corner",
    "diffmah",
    "diffstar",
    "dsps",
    "dustmaps",
    "extinction",
    "jax",
    "jax_cosmo",
    "jaxlib",
    "matplotlib",
    "numpyro",
    "optax",
    "tensorflow_probability",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["filter-reference.css"]
html_title = "JAXSEDFit Documentation"
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/burke86/jaxsedfit",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

# Render tutorial notebooks as documentation pages. Do not execute notebooks by
# default: several examples query remote services or run expensive samplers.
# Set NBSPHINX_EXECUTE=always locally or in CI to pre-execute notebooks.
nbsphinx_execute = os.environ.get("NBSPHINX_EXECUTE", "never")
nbsphinx_allow_errors = False
