"""Keep the Examples section of the site consistent with its notebooks.

The notebooks in notebooks/examples/ port MTL5's examples/ (issue #147).
docs/examples/index.md lists every MTL5 example and where it went; each
notebook names the MTL5 programs it ports in its metadata (`mtl5_examples`)
and links to them. These tests check that the three agree, and that every
link to MTL5 points at the tag this build fetches, so the links can't drift
from the code they describe.

With MTL5_SOURCE_DIR set to an MTL5 checkout at that tag, they also check
that the index lists exactly the example programs MTL5 has.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "notebooks" / "examples"
INDEX = REPO / "docs" / "examples" / "index.md"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))

LINK = re.compile(
    r"https://github\.com/stillwater-sc/mtl5/(?:blob|tree)/([^/\s)]+)/examples/?([^\s)]*)"
)
ROW = re.compile(r"^\| \[`([^`]+\.cpp)`\]\(", re.MULTILINE)


def mtl5_tag() -> str:
    """The MTL5 tag CMakeLists.txt fetches."""
    cmake = (REPO / "CMakeLists.txt").read_text(encoding="utf-8")
    block = re.search(r"FetchContent_Declare\(\s*mtl5\b(.*?)\)", cmake, re.DOTALL)
    assert block, "no FetchContent_Declare(mtl5 ...) in CMakeLists.txt"
    tag = re.search(r"GIT_TAG\s+(\S+)", block.group(1))
    assert tag, "the mtl5 FetchContent block has no GIT_TAG"
    return tag.group(1)


def index_text() -> str:
    return INDEX.read_text(encoding="utf-8")


def index_rows() -> list[str]:
    return ROW.findall(index_text())


def notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_there_are_example_notebooks():
    assert NOTEBOOKS, "expected notebooks under notebooks/examples/"


def test_toctree_lists_exactly_the_notebooks():
    toctree = re.findall(r"^\.\./notebooks/examples/(\S+)$", index_text(), re.MULTILINE)
    assert sorted(toctree) == sorted(p.stem for p in NOTEBOOKS)


def test_index_lists_each_mtl5_example_once():
    rows = index_rows()
    assert rows, "the index has no table of MTL5 examples"
    duplicates = sorted({r for r in rows if rows.count(r) > 1})
    assert not duplicates, f"listed more than once: {duplicates}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_names_and_links_its_sources(path: Path):
    nb = notebook(path)
    sources = nb["metadata"].get("mtl5_examples")
    assert sources, f"{path.name}: metadata has no mtl5_examples"

    rows = set(index_rows())
    missing = [s for s in sources if s not in rows]
    assert not missing, f"{path.name}: not in the index's table: {missing}"

    # The index row of each source points at this notebook.
    for source in sources:
        row = re.search(rf"^\| \[`{re.escape(source)}`\].*$", index_text(), re.MULTILINE)
        assert f"examples/{path.name}" in row.group(0), (
            f"the index row for {source} doesn't link to {path.name}"
        )

    intro = "".join(nb["cells"][0]["source"])
    linked = {m.group(2) for m in LINK.finditer(intro)}
    assert set(sources) <= linked, f"{path.name}: first cell doesn't link {set(sources) - linked}"


@pytest.mark.parametrize("path", [INDEX, *NOTEBOOKS], ids=lambda p: p.relative_to(REPO).as_posix())
def test_mtl5_links_use_the_fetched_tag(path: Path):
    tag = mtl5_tag()
    tags = {m.group(1) for m in LINK.finditer(path.read_text(encoding="utf-8"))}
    assert tags, f"{path.name} has no links to MTL5's examples"
    assert tags == {tag}, f"{path.name} links MTL5 at {sorted(tags)}, but the build fetches {tag}"


def test_index_matches_an_mtl5_checkout():
    root = os.environ.get("MTL5_SOURCE_DIR")
    if not root:
        pytest.skip("set MTL5_SOURCE_DIR to an MTL5 checkout to compare the index against it")
    examples = Path(root) / "examples"
    programs = sorted(p.relative_to(examples).as_posix() for p in examples.rglob("*.cpp"))
    assert sorted(index_rows()) == programs
