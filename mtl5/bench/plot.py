"""The core benchmark figures, drawn from the frames of :mod:`mtl5.bench.analyze`.

Each function takes a DataFrame and returns a :class:`BenchFigure`: the
matplotlib figure, and the frame it was drawn from. A figure reads nothing but
its frame, so it can be drawn again from a saved frame (:func:`from_csv`),
and the two renders are the same file byte for byte.

``capability_card``
    What a machine can do: CPU, build, FMA peaks, bandwidths, and the roofline
    those make.
``roofline``
    Every entry of one run at its arithmetic intensity and rate, under the
    machine's compute and bandwidth ceilings.
``thread_scaling``
    Speedup against threads, with the ideal line.
``rate_vs_size``
    A rate against problem size, one line per operation and backend.

Every figure is in the house style of :mod:`mtl5.bench.style` and carries a
provenance footer: the machines, the commits, the run count and dates, and a
warning when a run's provenance is incomplete.

The figures are built on :class:`matplotlib.figure.Figure` directly, without
pyplot, so they leave pyplot's global state alone. In a notebook a
:class:`BenchFigure` displays itself.
"""

from __future__ import annotations

import io
import math
import os
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import style
from .analyze import PROVENANCE_COLUMNS, _matches

__all__ = [
    "BenchFigure",
    "FIGURES",
    "capability_card",
    "roofline",
    "thread_scaling",
    "rate_vs_size",
    "from_csv",
    "provenance",
]


@dataclass
class BenchFigure:
    """A figure and the DataFrame it was drawn from."""

    kind: str
    figure: Any  # matplotlib.figure.Figure
    data: Any  # pandas.DataFrame

    def save(
        self,
        stem: str | os.PathLike,
        formats: Iterable[str] = style.FORMATS,
        *,
        data: bool = True,
    ) -> list[Path]:
        """Write ``stem.<format>`` for each format, and ``stem.csv`` with the data.

        :func:`from_csv` draws the same figure again from ``stem.csv``.
        """
        paths = style.save(self.figure, stem, formats)
        if data:
            stem = Path(stem)
            csv = stem.with_name(f"{stem.name}.csv")
            self.data.to_csv(csv, index=False)
            paths.append(csv)
        return paths

    def _render(self, fmt: str) -> bytes:
        buf = io.BytesIO()
        with style.house_style():
            self.figure.savefig(buf, format=fmt, metadata=style._METADATA[fmt])
        return buf.getvalue()

    def _repr_png_(self) -> bytes:
        return self._render("png")


# --- provenance ---------------------------------------------------------------


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return not (isinstance(v, str) and not v.strip())


def provenance(df) -> str:
    """The provenance footer of a frame: machines, commits, runs, dates, tier."""
    import pandas as pd

    missing = [c for c in PROVENANCE_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"the frame has no provenance columns {missing}")
    runs = df.drop_duplicates("run_id").copy()
    runs["_t"] = pd.to_datetime(runs["run_timestamp"], errors="coerce")
    runs = runs.sort_values(["_t", "run_id"], na_position="last")

    machines = sorted({m for m in runs["machine"] if _present(m)})
    unattributed = sum(not _present(m) for m in runs["machine"])
    parts = []
    if machines:
        parts.append(("machine " if len(machines) == 1 else "machines ") + ", ".join(machines))
    if unattributed:
        parts.append(f"{unattributed} run{'s' if unattributed > 1 else ''} with no machine")

    commits = list(dict.fromkeys(str(c)[:7] for c in runs["git_commit"] if _present(c)))
    if not commits:
        parts.append("commit unrecorded")
    elif len(commits) == 1:
        parts.append(f"commit {commits[0]}")
    else:
        parts.append(f"{len(commits)} commits {commits[0]}…{commits[-1]}")

    n = len(runs)
    times = runs["_t"].dropna()
    when = ""
    if not times.empty:
        first, last = times.iloc[0].date(), times.iloc[-1].date()
        when = f", {first}" if first == last else f", {first} to {last}"
    parts.append(f"{n} run{'s' if n != 1 else ''}{when}")

    incomplete = int((runs["tier"] != "complete").sum())
    if incomplete:
        parts.append(f"INCOMPLETE PROVENANCE in {incomplete} of {n} runs")
    return " · ".join(parts)


def _finish(fig, df) -> None:
    """Add the provenance footer, then fix the layout.

    Constrained layout is computed once, here, and then switched off. Left on,
    it runs again at every draw and starts from where the last one left the
    axes, so writing a PNG before an SVG would move the SVG's axes.
    """
    fig.text(
        0.0,
        0.0,
        provenance(df),
        ha="left",
        va="top",
        fontsize=5.5,
        color=style.MUTED,
        transform=fig.transFigure,
    )
    fig.draw_without_rendering()
    fig.set_layout_engine("none")


def _figure(width: float, height: float):
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), layout="constrained")


def _point_style(i: int) -> dict:
    """A colour and marker pair, distinct for the first 49 series."""
    n = len(style.PALETTE)
    return {"color": style.PALETTE[i % n], "marker": style.MARKERS[(i + i // n) % n]}


def _line_style(i: int) -> dict:
    n = len(style.PALETTE)
    j = (i + i // n) % n
    return {
        "color": style.PALETTE[i % n],
        "marker": style.MARKERS[j],
        "linestyle": style.LINESTYLES[j],
    }


def _check_series(n: int, max_series: int, what: str) -> None:
    if n == 0:
        raise ValueError(f"nothing to plot: the frame has no {what}")
    if n > max_series:
        raise ValueError(
            f"{n} {what} is more than a legible figure holds ({max_series}); "
            "select fewer, or raise max_series"
        )


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _g(v: Any, digits: int = 3) -> str:
    x = _num(v)
    return "n/a" if x is None else f"{x:.{digits}g}"


# --- roofline ceilings, shared by the card and the roofline -------------------


def _ceilings(row) -> list[tuple[str, float]]:
    """(label, peak GFLOP/s) for each compute ceiling a run has.

    The hardware FMA peaks only. The ``build_fma_peak_*`` probes, the same FMA
    loop compiled for the run's SIMD backend, are the card's text, not a
    ceiling: in a scalar build they sit far below what blocked kernels reach,
    so a line there would read as a bound the points break.
    """
    out = []
    for dtype in ("f64", "f32"):
        peak = _num(row.get(f"peak_{dtype}_gflops"))
        if peak:
            out.append((f"{dtype} FMA peak", peak))
    return out


def _draw_roof(ax, row, xlo: float, xhi: float) -> None:
    import numpy as np

    bw = _num(row.get("bandwidth_gbytes_per_s"))
    x = np.geomspace(xlo, xhi, 256)
    for label, peak in _ceilings(row):
        y = np.minimum(peak, bw * x) if bw else np.full_like(x, peak)
        ax.plot(x, y, color=style.MUTED, linestyle="-", linewidth=0.9, marker="", zorder=1)
        ax.annotate(
            f"{label} {_g(peak)}",
            (x[-1], peak),
            xytext=(-2, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6,
            color=style.MUTED,
        )
    if bw:
        # Up and to the left of a point on the slope, which is above the line.
        xb = xlo * 8
        ax.annotate(
            f"{_g(bw)} GB/s",
            (xb, bw * xb),
            xytext=(-3, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6,
            color=style.MUTED,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")


def _title(row) -> str:
    parts = [str(row.get("machine"))]
    if _present(row.get("label")):
        parts.append(str(row.get("label")))
    t = _num(row.get("threads"))
    if t is not None:
        parts.append(f"{int(t)} thread{'s' if int(t) != 1 else ''}")
    return " · ".join(parts)


# --- the figures ----------------------------------------------------------------


def capability_card(df) -> BenchFigure:
    """A machine's capability card, from :func:`mtl5.bench.analyze.capability`."""
    if len(df) != 1:
        raise ValueError(f"a capability card describes one run; the frame has {len(df)}")
    row = df.iloc[0]
    with style.house_style():
        fig = _figure(style.WIDTH_DOUBLE, 2.5)
        text_ax, roof_ax = fig.subplots(1, 2, width_ratios=(1.25, 1))
        text_ax.set_axis_off()

        def per(dtype: str) -> str:
            hw = _g(row.get(f"fma_peak_{dtype}_gflops"))
            build = _g(row.get(f"build_fma_peak_{dtype}_gflops"))
            return f"{hw} {dtype}" + (f" (this build {build})" if build != hw else "")

        def joined(*parts) -> str:
            return ", ".join(str(p) for p in parts if _present(p))

        def tagged(tag: str, key: str) -> str | None:
            return f"{tag} {row.get(key)}" if _present(row.get(key)) else None

        llc = _num(row.get("last_level_cache_bytes"))
        cores = _num(row.get("cpu_logical_cores"))
        compiler = " ".join(
            str(v) for v in (row.get("compiler"), row.get("compiler_version")) if _present(v)
        )
        os_text = row.get("os_version") if _present(row.get("os_version")) else row.get("os_name")
        lines = [
            ("CPU", row.get("cpu_brand")),
            (
                "Architecture",
                joined(row.get("cpu_arch"), f"{int(cores)} logical cores" if cores else None),
            ),
            ("SIMD", row.get("cpu_simd")),
            ("FMA unit", row.get("hw_fma_isa")),
            (
                "Build",
                joined(compiler, tagged("ISA", "build_isa"), tagged("SIMD", "simd_backend")),
            ),
            ("OS", os_text),
            ("FMA peak, GFLOP/s", f"{per('f64')}, {per('f32')}"),
            (
                "Bandwidth, GB/s",
                f"read {_g(row.get('read_gbytes_per_s'))}, "
                f"update {_g(row.get('update_gbytes_per_s'))}, "
                f"triad {_g(row.get('triad_gbytes_per_s'))}",
            ),
            ("Last-level cache", f"{llc / 2**20:.3g} MiB" if llc else "n/a"),
        ]
        y = 0.97
        for key, value in lines:
            value = str(value) if _present(value) else "n/a"
            wrapped = textwrap.wrap(value, 58) or ["n/a"]
            text_ax.text(0.0, y, key, fontsize=7, fontweight="bold", va="top")
            text_ax.text(0.33, y, "\n".join(wrapped), fontsize=7, va="top")
            y -= 0.105 * len(wrapped)
        text_ax.set_title(_title(row), loc="left")

        roof_row = {
            "peak_f64_gflops": row.get("fma_peak_f64_gflops"),
            "peak_f32_gflops": row.get("fma_peak_f32_gflops"),
            "bandwidth_gbytes_per_s": row.get("bandwidth_gbytes_per_s"),
        }
        _draw_roof(roof_ax, roof_row, 1 / 32, 256)
        roof_ax.set_title("roofline", loc="left")
        _finish(fig, df)
    return BenchFigure("card", fig, df)


def roofline(df, *, label_entries: Sequence[str] = ()) -> BenchFigure:
    """One run's entries on its roofline, from :func:`mtl5.bench.analyze.roofline`.

    Points are grouped by area. ``label_entries`` names entries to label on
    the plot, by id or by a prefix of whole segments (``"blas3/gemm"`` labels
    ``blas3/gemm/f64/...`` but not ``blas3/gemmt/...``).
    """
    if df["run_id"].nunique() != 1:
        raise ValueError("a roofline places the entries of one run")
    areas = sorted(df["area"].dropna().unique())
    _check_series(len(areas), 49, "areas")
    row = df.iloc[0]
    with style.house_style():
        fig = _figure(style.WIDTH_DOUBLE, 3.6)
        ax = fig.subplots()
        lo = float(df["intensity"].min())
        hi = float(df["intensity"].max())
        _draw_roof(ax, row, 2 ** math.floor(math.log2(lo) - 1), 2 ** math.ceil(math.log2(hi) + 1))
        for i, area in enumerate(areas):
            pts = df[df["area"] == area]
            ax.scatter(
                pts["intensity"],
                pts["gflops"],
                s=12,
                linewidths=0.4,
                edgecolors="white",
                label=area,
                zorder=3,
                **_point_style(i),
            )
        for _, p in df.iterrows():
            if _matches(p["entry_id"], label_entries):
                ax.annotate(
                    f"{p['op']} {p['dtype']}",
                    (p["intensity"], p["gflops"]),
                    xytext=(3, -3),
                    textcoords="offset points",
                    fontsize=6,
                    va="top",
                )
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=1 + len(areas) // 18)
        ax.set_title(_title(row), loc="left")
        _finish(fig, df)
    return BenchFigure("roofline", fig, df)


def _entry_labels(entry_ids: Sequence[str]) -> dict[str, str]:
    """Entry ids without the shape, where that keeps them distinct."""
    short = {e: "/".join(e.split("/")[:3]) for e in entry_ids}
    if len(set(short.values())) == len(short):
        return short
    return {e: e for e in entry_ids}


def thread_scaling(df, *, max_series: int = 12) -> BenchFigure:
    """Speedup against threads, from :func:`mtl5.bench.analyze.thread_scaling`."""
    entries = sorted(df["entry_id"].unique())
    _check_series(len(entries), max_series, "entries")
    labels = _entry_labels(entries)
    threads = sorted(int(t) for t in df["threads"].unique())
    with style.house_style():
        fig = _figure(style.WIDTH_SINGLE, 2.8)
        ax = fig.subplots()
        top = max(threads)
        ax.plot(
            [1, top],
            [1, top],
            color=style.MUTED,
            linestyle="--",
            marker="",
            linewidth=0.8,
            label="ideal",
        )
        for i, e in enumerate(entries):
            pts = df[df["entry_id"] == e].sort_values("threads")
            ax.plot(pts["threads"], pts["speedup"], label=labels[e], **_line_style(i))
        ax.set_xticks(threads)
        ax.set_xlim(0.5, top + 0.5)
        ax.set_ylim(0, max(top, float(df["speedup"].max())) * 1.05)
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup over 1 thread")
        ax.legend(loc="upper left")
        row = df.iloc[0]
        ax.set_title(
            " · ".join(str(x) for x in (row.get("machine"), row.get("label")) if _present(x)),
            loc="left",
        )
        _finish(fig, df)
    return BenchFigure("scaling", fig, df)


def rate_vs_size(df, *, max_series: int = 12) -> BenchFigure:
    """A rate against problem size, from :func:`mtl5.bench.analyze.rate_vs_size`."""
    import pandas as pd

    df = df.copy()
    df["_machine"] = [m if _present(m) else "no machine" for m in df["machine"]]
    keys = sorted(
        df[["_machine", "operation", "backend"]].drop_duplicates().itertuples(index=False)
    )
    _check_series(len(keys), max_series, "series (machine, operation, backend)")
    several = len({k[0] for k in keys}) > 1
    units = sorted(df["unit"].dropna().unique())
    metric = df["metric"].iloc[0] if "metric" in df else "rate"
    with style.house_style():
        fig = _figure(style.WIDTH_SINGLE, 2.8)
        ax = fig.subplots()
        for i, (machine, op, backend) in enumerate(keys):
            mask = (df["_machine"] == machine) & (df["operation"] == op)
            pts = df[mask & (df["backend"] == backend)]
            pts = pts.groupby("size", as_index=False)["value"].median().sort_values("size")
            label = f"{op} ({backend})" + (f", {machine}" if several else "")
            ax.plot(pd.to_numeric(pts["size"]), pts["value"], label=label, **_line_style(i))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("problem size n")
        ax.set_ylabel(units[0] if len(units) == 1 else metric)
        ax.legend(loc="best")
        if not several:
            ax.set_title(keys[0][0], loc="left")
        _finish(fig, df.drop(columns="_machine"))
    return BenchFigure("rate", fig, df.drop(columns="_machine"))


#: The figures by kind, as :func:`from_csv` and the command line name them.
FIGURES = {
    "card": capability_card,
    "roofline": roofline,
    "scaling": thread_scaling,
    "rate": rate_vs_size,
}


def from_csv(path: str | os.PathLike, kind: str, **kwargs) -> BenchFigure:
    """Draw a figure again from the frame :meth:`BenchFigure.save` wrote."""
    import pandas as pd

    if kind not in FIGURES:
        raise ValueError(f"unknown figure {kind!r}; use one of {sorted(FIGURES)}")
    return FIGURES[kind](pd.read_csv(path), **kwargs)
