# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "efficient-transformers"
copyright = "2025, Qualcomm"

# The full version, including alpha/beta/rc tags
release = os.getenv("DOC_VERSION", "main")


def _sort_versions(versions):
    def key(version):
        if version == "main":
            return (0, 0, 0, 0, "")
        match = re.fullmatch(r"(?:release-)?v?(\d+)(?:\.(\d+))?(?:\.(\d+))?([\-\+].*)?", version)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2) or 0)
            patch = int(match.group(3) or 0)
            suffix = match.group(4) or ""
            return (1, -major, -minor, -patch, suffix)
        return (2, 0, 0, 0, version)

    return sorted(set(versions), key=key)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx_multiversion",
    "sphinx.ext.napoleon",
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]  # This tells Sphinx to process both .rst and .md files

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# source = [".md"] # This line was commented out/incorrect syntax for source_suffix
todo_include_todos = True
available_versions = [v.strip() for v in os.getenv("DOCS_AVAILABLE_VERSIONS", "").split(",") if v.strip()]
if release not in available_versions:
    available_versions.append(release)
available_versions = _sort_versions(available_versions)
html_context = {
    "doc_version": release,
    "available_versions": available_versions,
    "versions_index_url": os.getenv("DOCS_VERSIONS_INDEX_URL", "versions/index.html"),
    "latest_docs_url": os.getenv("DOCS_LATEST_URL", "./"),
}

suppress_warnings = [
    "ref.rst_pilog",  # Suppress warnings about excluded toctree entries
]

# OpenCompute blocks automated bots and returns 403 to linkcheck,
# so we keep the public URL but ignore it during CI link verification.
linkcheck_ignore = [r"https://www\.opencompute\.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf"]


def setup(app):
    app.add_css_file("my_theme.css")
