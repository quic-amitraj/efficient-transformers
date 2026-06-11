# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "efficient-transformers"
author = "Qualcomm"
copyright = "2025, Qualcomm"
release = os.getenv("DOC_VERSION", "main")
html_title = f"{project} ({release})"


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

extensions = [
    "myst_parser",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
language = "en"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}


# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "style_external_links": True,
}

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
    "ref.rst_pilog",
]

# OpenCompute blocks automated bots and returns 403 to linkcheck,
# so we keep the public URL but ignore it during CI link verification.
linkcheck_ignore = [
    r"https://www\.opencompute\.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf",
]
linkcheck_timeout = 10
linkcheck_retries = 2


def setup(app):
    app.add_css_file("my_theme.css")
