"""Sphinx configuration for the mtl5-python documentation site.

Build locally (needs the compiled extension and the `docs` extra):

    pip install -e ".[docs]"
    sphinx-build -E -W --keep-going -b html docs docs/_build/html

-E discards the saved environment, so every page, including the executable
getting-started page, is re-read and re-executed against the current package.

The notebooks under notebooks/ are the single source of truth. They are
committed with their outputs stripped (tests/test_notebooks.py enforces it),
copied into docs/notebooks/ at the start of every build, and executed by
MyST-NB, so the site always shows outputs from the current code.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mtl5
import mtl5.io
import mtl5.sparse

# mtl5.mixed is a submodule created by the compiled extension: an attribute of
# mtl5, not a module the import system can find, so autodoc's
# `import mtl5.mixed` fails. Registering it lets autodoc document its functions.
sys.modules.setdefault("mtl5.mixed", mtl5.mixed)

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent

# -- Project -----------------------------------------------------------------

project = "mtl5-python"
author = "Stillwater Supercomputing, Inc."
copyright = "2026, Stillwater Supercomputing, Inc."
release = mtl5.__version__
version = ".".join(release.split(".")[:2])

# -- General -----------------------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]
templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
root_doc = "index"

# docs/ also holds internal notes that are not part of the site.
exclude_patterns = [
    "_build",
    "designs/**",
    "sessions/**",
    "gap-analysis-*.md",
    "pypi-release.md",
    "notebooks/data/**",
    "**/.ipynb_checkpoints",
]

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

# -- Notebooks ---------------------------------------------------------------

# Committed notebooks have no outputs, so always execute. A failing cell fails
# the build, which is the point: the site cannot show stale numbers.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 600
nb_execution_show_tb = True
nb_merge_streams = True


def _copy_notebooks() -> None:
    """Mirror notebooks/ into docs/notebooks/ (generated, git-ignored)."""
    src, dst = REPO / "notebooks", DOCS / "notebooks"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    # copy, not copy2: a fresh mtime makes Sphinx re-read, and so re-execute,
    # every notebook on every build. Keeping the source mtime would let an
    # incremental build reuse cached outputs after a change to the package.
    for nb in sorted(src.glob("*.ipynb")):
        shutil.copy(nb, dst / nb.name)
    # The examples ported from MTL5's examples/ (docs/examples/index.md).
    (dst / "examples").mkdir()
    for nb in sorted(src.glob("examples/*.ipynb")):
        shutil.copy(nb, dst / "examples" / nb.name)
    if (src / "data").is_dir():
        shutil.copytree(src / "data", dst / "data")


_copy_notebooks()

# -- API reference -----------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_typehints = "signature"
# nanobind overload signatures such as `shape=(*, *)` are not Python syntax, so
# Sphinx cannot parse them as signatures. _move_overload_signatures shows them
# verbatim instead.
autodoc_docstring_signature = False
# The extension's docstrings are written for help(), not as reStructuredText:
# `x` is code, not a cross-reference. Render it as code.
default_role = "code"


def _move_overload_signatures(app, what, name, obj, options, lines):
    """Move nanobind's leading overload signatures out of the docstring body.

    A nanobind function's __doc__ starts with one line per overload, such as
    `lu(A: ndarray[dtype=float64, shape=(*, *), ...]) -> ...`. autodoc turns
    the lines it can parse into signatures; the rest would be parsed as
    reStructuredText, where `(*` opens an emphasis that never closes, and
    autosummary would take a signature as the summary line. They move into a
    literal block at the end, so the description comes first and the
    signatures stay visible verbatim.
    """
    short = name.rsplit(".", 1)[-1]
    sigs = []
    while lines and lines[0].startswith(short + "("):
        sigs.append(lines.pop(0))
    if not sigs:
        return
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip() == "Overloaded function.":
        return  # nanobind lists each overload, with its own text, below this line
    lines.extend(["", "**Signatures:**", "", ".. code-block:: text", ""])
    lines.extend("   " + sig for sig in sigs)
    lines.append("")


def _wrap_nanobind_functions() -> None:
    """Present each compiled function to autodoc as a plain Python function.

    Sphinx recognizes a function through inspect.isroutine(), which is False
    for nanobind's nb_func type, so autosummary would file every compiled
    function under `autodata` and render it as a value. Inside this Sphinx
    process only, each one is replaced by a thin Python function with the same
    name and docstring. The notebooks execute in their own kernel and never
    see these wrappers.
    """
    import functools

    for module in (mtl5, mtl5.sparse, mtl5.io, mtl5.mixed):
        for name in dir(module):
            obj = getattr(module, name)
            # A Python wrapper of an nb_func (mtl5.solve, #152) is documented
            # as the nb_func it wraps, from the docstring functools.wraps kept.
            nanobind = type(obj).__name__ == "nb_func"
            nanobind = nanobind or type(getattr(obj, "__wrapped__", None)).__name__ == "nb_func"
            if name.startswith("_") or not nanobind:
                continue

            def wrapper(*args, _f=obj, **kwargs):
                return _f(*args, **kwargs)

            functools.update_wrapper(wrapper, obj, assigned=("__doc__",), updated=())
            wrapper.__name__ = wrapper.__qualname__ = name
            wrapper.__module__ = module.__name__
            wrapper._mtl5_nanobind = True
            setattr(module, name, wrapper)


_wrap_nanobind_functions()


def _nanobind_signature(app, what, name, obj, options, signature, return_annotation):
    # The overloads are listed in the docstring, by _move_overload_signatures.
    if getattr(obj, "_mtl5_nanobind", False):
        return "(...)", None
    return None


def _numpy_sections(app, what, name, obj, options, lines):
    """Parse NumPy-style Parameters/Returns sections, for callables only.

    This is sphinx.ext.napoleon's parser, applied here instead of through the
    extension, because the extension also parses every property and attribute
    docstring as "type: description". Properties such as SparseLU.L carry
    plain prose with colons in it (A[row_perm][:, col_perm]), which that
    split would cut apart.
    """
    if what not in ("function", "method", "class"):
        return
    from sphinx.ext.napoleon import Config
    from sphinx.ext.napoleon.docstring import NumpyDocstring

    config = Config(napoleon_use_param=True, napoleon_use_rtype=True)
    lines[:] = NumpyDocstring(lines, config, app, what, name, obj, options).lines()


def setup(app):
    app.connect("autodoc-process-signature", _nanobind_signature)
    app.connect("autodoc-process-docstring", _move_overload_signatures)
    app.connect("autodoc-process-docstring", _numpy_sections)


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML --------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"mtl5-python {release}"
html_theme_options = {
    "github_url": "https://github.com/stillwater-sc/mtl5-python",
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_context = {
    "github_user": "stillwater-sc",
    "github_repo": "mtl5-python",
    "github_version": "main",
    "doc_path": "docs",
}
