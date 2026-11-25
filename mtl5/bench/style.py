"""The house style for benchmark figures.

One style for every figure, so figures from different questions read as one
set:

- **Colour:** the Okabe-Ito palette, which stays distinguishable under the
  common colour-vision deficiencies. Its yellow is left out because it
  disappears on white.
- **Grayscale:** every series also gets its own marker and dash pattern, so a
  figure printed in gray still separates its series.
- **Sizes:** figures are sized in inches for a journal column
  (:data:`WIDTH_SINGLE`) or a full page (:data:`WIDTH_DOUBLE`), with fonts
  set for that size, not scaled down from a screen.
- **Output:** vector PDF and SVG with the text kept as text in the PDF
  (TrueType, not Type 3) and drawn as paths in the SVG, and a 300 dpi PNG.
  The font is DejaVu Sans, which ships with matplotlib, so a figure renders
  the same on any machine.
- **Reproducible files:** the writers' timestamps are switched off and the SVG
  ids are seeded, so the same DataFrame gives the same bytes.

Use :func:`house_style` as a context manager around figure construction, and
:func:`save` to write the files.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = [
    "PALETTE",
    "MARKERS",
    "LINESTYLES",
    "WIDTH_SINGLE",
    "WIDTH_DOUBLE",
    "FORMATS",
    "rc_params",
    "house_style",
    "series_style",
    "save",
]

#: Okabe-Ito, without yellow, in the order series take them.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES: tuple = (
    "-",
    "--",
    ":",
    "-.",
    (0, (5, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (1, 1)),
)

#: Figure widths in inches: one journal column, and a full page.
WIDTH_SINGLE = 3.5
WIDTH_DOUBLE = 7.2

#: The formats :func:`save` writes by default.
FORMATS: tuple[str, ...] = ("pdf", "svg", "png")

# Text colour for secondary text: the provenance footer, annotations.
MUTED = "#555555"


def series_style(i: int) -> dict:
    """The colour, marker and dash pattern of the ``i``-th series."""
    n = len(PALETTE)
    return {"color": PALETTE[i % n], "marker": MARKERS[i % n], "linestyle": LINESTYLES[i % n]}


def rc_params() -> dict:
    """The matplotlib rcParams of the house style."""
    from cycler import cycler

    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=list(PALETTE))
        + cycler(marker=list(MARKERS))
        + cycler(linestyle=list(LINESTYLES)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "grid.color": "#b0b0b0",
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "mtl5-bench",
        "path.simplify": False,
    }


@contextlib.contextmanager
def house_style() -> Iterator[None]:
    """Apply the house style for the duration of the block."""
    import matplotlib

    with matplotlib.rc_context(rc_params()):
        yield


# Writers' metadata that would make two renders of one figure differ.
_METADATA = {
    "pdf": {"Creator": "mtl5.bench", "Producer": None, "CreationDate": None},
    "svg": {"Creator": "mtl5.bench", "Date": None},
    "png": {"Software": "mtl5.bench"},
}


def save(fig, stem: str | os.PathLike, formats: Iterable[str] = FORMATS) -> list[Path]:
    """Write ``fig`` as ``stem.<format>`` for each format; return the paths.

    The files carry no timestamp, so the same figure gives the same bytes.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with house_style():
        for fmt in formats:
            if fmt not in _METADATA:
                raise ValueError(f"unsupported format {fmt!r}; use one of {sorted(_METADATA)}")
            path = stem.with_name(f"{stem.name}.{fmt}")
            fig.savefig(path, format=fmt, metadata=_METADATA[fmt])
            paths.append(path)
    return paths
