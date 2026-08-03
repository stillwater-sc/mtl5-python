"""Execute every notebook's code cells, so documentation cannot rot silently.

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
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb")) if NOTEBOOK_DIR.is_dir() else []

# Resolve the child against the same mtl5 this process imported — under
# `pip install .` the repo root holds a source package with no compiled _core.
MTL5_IMPORT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(mtl5.__file__)))


def notebook_to_script(path: Path) -> str:
    nb = json.loads(path.read_text())
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


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path: Path):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = MTL5_IMPORT_ROOT + os.pathsep + existing if existing else MTL5_IMPORT_ROOT
    env["MPLBACKEND"] = "Agg"

    with tempfile.TemporaryDirectory() as work:
        script = Path(work) / "cells.py"
        script.write_text(notebook_to_script(path))
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=work,
            env=env,
            timeout=600,
        )
    assert result.returncode == 0, (
        f"{path.name} failed to execute:\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_outputs_are_stripped(path: Path):
    """Committed outputs bloat diffs and go stale; the runner above is what
    proves the notebook works."""
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"{path.name} cell {i} has stored outputs"
            assert cell.get("execution_count") is None, (
                f"{path.name} cell {i} has an execution count"
            )
