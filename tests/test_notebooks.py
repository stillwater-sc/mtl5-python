"""Execute every notebook's code cells, so documentation cannot rot silently.

Every file read and write here pins encoding="utf-8". The notebooks contain
non-ASCII prose, and the default is locale-dependent: on the Windows runners it
currently survives only because a cp1252 decode/encode round-trips the original
UTF-8 bytes unchanged, which is luck rather than design — an ASCII or CJK
locale would raise instead.

Runs the cells as a plain script in a subprocess rather than through Jupyter:
the notebooks are linear demonstrations with no cell-order tricks, so this
catches the failure that matters (an API drifted and the notebook no longer
runs) without adding jupyter to the test dependencies.

Skipped where matplotlib is absent — it is a `notebooks` extra, not a core one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import mtl5

pytest.importorskip("matplotlib", reason="notebooks need the [notebooks] extra")

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
# The guide's notebooks, and the examples ported from MTL5 in examples/.
NOTEBOOKS = (
    sorted([*NOTEBOOK_DIR.glob("*.ipynb"), *NOTEBOOK_DIR.glob("examples/*.ipynb")])
    if NOTEBOOK_DIR.is_dir()
    else []
)

# Resolve the child against the same mtl5 this process imported — under
# `pip install .` the repo root holds a source package with no compiled _core.
MTL5_IMPORT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(mtl5.__file__)))


def notebook_to_script(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = [
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as _plt",
        "_plt.show = lambda *a, **k: _plt.close('all')",
    ]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            parts.append("".join(cell["source"]))
    return "\n".join(parts)


def test_notebooks_exist():
    assert NOTEBOOKS, "expected at least one notebook under notebooks/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.relative_to(NOTEBOOK_DIR).as_posix())
def test_notebook_runs(path: Path):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = MTL5_IMPORT_ROOT + os.pathsep + existing if existing else MTL5_IMPORT_ROOT
    env["MPLBACKEND"] = "Agg"
    # Jupyter's output is UTF-8 on every platform, and the notebooks print
    # non-ASCII text (ε, ∞). A script's stdout follows the locale instead,
    # which is cp1252 on the Windows runners and can't encode those.
    env["PYTHONIOENCODING"] = "utf-8"

    with tempfile.TemporaryDirectory() as work:
        script = Path(work) / "cells.py"
        script.write_text(notebook_to_script(path), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            # The notebook's own directory, as under Jupyter, so a notebook
            # reads `data/...` by the same relative path in both. Safe for the
            # import: a script's directory, not the cwd, goes on sys.path, and
            # the script lives in `work`, so notebooks/ cannot shadow mtl5.
            cwd=path.parent,
            env=env,
            timeout=600,
        )
    assert result.returncode == 0, (
        f"{path.name} failed to execute:\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.relative_to(NOTEBOOK_DIR).as_posix())
def test_notebook_outputs_are_stripped(path: Path):
    """Committed outputs bloat diffs and go stale; the runner above is what
    proves the notebook works."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"{path.name} cell {i} has stored outputs"
            assert cell.get("execution_count") is None, (
                f"{path.name} cell {i} has an execution count"
            )


def _scipy_floor(requirements: list[str]) -> tuple[int, ...]:
    (spec,) = [r for r in requirements if r.startswith("scipy")]
    return tuple(int(part) for part in spec.split(">=")[1].split("."))


def test_notebooks_extra_allows_only_the_scipy_the_notebooks_need():
    """The notebooks run under both the `notebooks` and the `docs` extra. The
    docs extra requires SciPy 1.12, the first whose iterative solvers take
    `rtol=`, which the multigrid notebook passes to `scipy.sparse.linalg.cg`;
    with the notebooks extra at 1.10 it raised TypeError on SciPy 1.10-1.11."""
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert _scipy_floor(extras["notebooks"]) >= _scipy_floor(extras["docs"]) >= (1, 12)
