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
    "machine_balance",
    "cross_machine",
    "from_csv",
    "read_frame",
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
    """A colour, marker and dash, with a distinct marker and dash pair for the
    first 49 series, so no two of them differ by colour alone."""
    n = len(style.PALETTE)
    return {
        "color": style.PALETTE[i % n],
        "marker": style.MARKERS[(i + i // n) % n],
        "linestyle": style.LINESTYLES[i % n],
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
    return "n/a" if x is None else style.number(x, digits)


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
    df = df.copy()  # the figure keeps the frame it was drawn from
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


def roofline(df, *, label_entries: Sequence[str] | None = None) -> BenchFigure:
    """One run's entries on its roofline, from :func:`mtl5.bench.analyze.roofline`.

    Points are grouped by area. ``label_entries`` names entries to label on
    the plot, by id or by a prefix of whole segments (``"blas3/gemm"`` labels
    ``blas3/gemm/f64/...`` but not ``blas3/gemmt/...``). The selection is kept
    in the frame as a ``labelled`` column, so a redraw from the saved frame
    labels the same entries; ``label_entries`` given here overrides it.
    """
    if df["run_id"].nunique() != 1:
        raise ValueError("a roofline places the entries of one run")
    areas = sorted(df["area"].dropna().unique())
    _check_series(len(areas), 49, "areas")
    df = df.copy()
    if label_entries is not None:
        df["labelled"] = df["entry_id"].map(lambda e: _matches(e, label_entries))
    elif "labelled" in df:
        df["labelled"] = df["labelled"].map(lambda v: v is True or str(v) == "True")
    else:
        df["labelled"] = False
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
            if p["labelled"]:
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
    df = df.copy()  # the figure keeps the frame it was drawn from
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
    # A series is one run's measurements of an operation: two runs of one
    # backend (two builds, two sessions) are two lines, never one median.
    keys = sorted(
        df[["_machine", "operation", "backend", "run_id"]].drop_duplicates().itertuples(index=False)
    )
    _check_series(len(keys), max_series, "series (machine, operation, backend, run)")
    several = len({k[0] for k in keys}) > 1
    # Runs of one (machine, operation, backend) are told apart in the legend
    # by their commit or date, and by their id where those collide.
    tags: dict[tuple, str] = {}
    by_series: dict[tuple, list[str]] = {}
    for m, op, b, run in keys:
        by_series.setdefault((m, op, b), []).append(run)
    for series, runs in by_series.items():
        if len(runs) < 2:
            continue
        short = {run: _run_tag(df[df["run_id"] == run].iloc[0]) for run in runs}
        for run in runs:
            clash = sum(t == short[run] for t in short.values()) > 1
            tags[(*series, run)] = f"{short[run]} ({str(run)[:7]})" if clash else short[run]
    units = sorted(df["unit"].dropna().unique())
    metric = df["metric"].iloc[0] if "metric" in df else "rate"
    with style.house_style():
        fig = _figure(style.WIDTH_SINGLE, 2.8)
        ax = fig.subplots()
        for i, (machine, op, backend, run) in enumerate(keys):
            pts = df[df["run_id"] == run]
            pts = pts[(pts["operation"] == op) & (pts["backend"] == backend)]
            pts = pts.groupby("size", as_index=False)["value"].median().sort_values("size")
            label = f"{op} ({backend})" + (f", {machine}" if several else "")
            if (machine, op, backend, run) in tags:
                label += f", {tags[(machine, op, backend, run)]}"
            ax.plot(pd.to_numeric(pts["size"]), pts["value"], label=label, **_line_style(i))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("problem size n")
        ax.set_ylabel(units[0] if len(units) == 1 else metric)
        ax.legend(loc="best")
        if not several:
            ax.set_title(keys[0][0], loc="left")
        _finish(fig, df.drop(columns="_machine"))
    return BenchFigure("rate", fig, df.drop(columns="_machine"))


def _run_tag(row) -> str:
    """A short name for a run in a legend: its commit, else its date, else its id."""
    import pandas as pd

    if _present(row.get("git_commit")):
        return str(row["git_commit"])[:7]
    t = pd.to_datetime(row.get("run_timestamp"), errors="coerce")
    return str(t.date()) if not pd.isna(t) else f"run {str(row['run_id'])[:7]}"


def _true(v) -> bool:
    """A flag as saved: True, or the text "True" after a CSV round trip."""
    return v is True or str(v) == "True"


def _machine_labels(df) -> dict[str, str]:
    """Each machine with the thread count(s) its runs used."""
    threads = df.groupby("machine")["threads"].agg(lambda t: sorted({int(x) for x in t}))
    return {
        m: f"{m} ({'/'.join(map(str, t))} thread{'s' if t != [1] else ''})"
        for m, t in threads.items()
    }


def machine_balance(df, *, precision: str = "f64", max_series: int = 12) -> BenchFigure:
    """FMA peak against bandwidth per machine, from :func:`mtl5.bench.analyze.machines`.

    Each machine is the median of its runs, with bars from the least to the
    most its runs measured. The diagonals are constant arithmetic intensity:
    a machine on the ``4 FLOP/byte`` line needs a kernel to do 4 FLOP per
    byte moved before its FMA peak, not its memory, is the limit (its ridge
    point). Runs without a positive peak and bandwidth are left off the axes,
    and a machine with no such run is named in the title.
    """
    import numpy as np

    if precision not in ("f64", "f32"):
        raise ValueError(f"precision must be 'f64' or 'f32', not {precision!r}")
    peak_col = f"fma_peak_{precision}_gflops"
    for col in (peak_col, "bandwidth_gbytes_per_s"):
        if col not in df:
            raise ValueError(f"the frame has no {col} column")
    import pandas as pd

    df = df.copy()  # the figure keeps the frame it was drawn from
    # Only runs with both probes, positive and finite, go on a log-log plot.
    x_all = pd.to_numeric(df["bandwidth_gbytes_per_s"], errors="coerce")
    y_all = pd.to_numeric(df[peak_col], errors="coerce")
    usable = (x_all > 0) & (y_all > 0) & np.isfinite(x_all) & np.isfinite(y_all)
    plotted = df[usable]
    names = sorted(plotted["machine"].dropna().unique())
    if not names:
        raise ValueError(
            f"nothing to plot: no run has a positive {peak_col} and bandwidth_gbytes_per_s"
        )
    _check_series(len(names), max_series, "machines")
    dropped = sorted(set(df["machine"].dropna()) - set(names))
    labels = _machine_labels(plotted)
    stats = {}
    for m in names:
        g = plotted[plotted["machine"] == m]
        x = x_all[g.index]
        y = y_all[g.index]
        stats[m] = (x.median(), x.min(), x.max(), y.median(), y.min(), y.max())
    xs = [v for s_ in stats.values() for v in s_[:3]]
    ys = [v for s_ in stats.values() for v in s_[3:]]
    xlo, xhi = min(xs) / 1.6, max(xs) * 1.6
    ylo, yhi = min(ys) / 1.6, max(ys) * 1.6
    with style.house_style():
        fig = _figure(style.WIDTH_SINGLE + 1.4, 3.2)
        ax = fig.subplots()
        # Constant-intensity guides, a power of two apart.
        lo_k = math.floor(math.log2(ylo / xhi))
        hi_k = math.ceil(math.log2(yhi / xlo))
        x = np.geomspace(xlo, xhi, 64)
        for k in range(lo_k, hi_k + 1):
            if k % 2:
                continue
            intensity = 2.0**k
            ax.plot(x, intensity * x, color="#c8c8c8", linestyle="-", marker="",
                    linewidth=0.6, zorder=1)  # fmt: skip
            # Label where the line leaves the plot.
            xe = min(xhi, yhi / intensity)
            if xlo < xe <= xhi and ylo < intensity * xe:
                ax.annotate(
                    f"{style.number(intensity)} FLOP/B",
                    (xe, intensity * xe),
                    xytext=(-2, -2),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=5.5,
                    color=style.MUTED,
                )
        for i, m in enumerate(names):
            xm, x0, x1, ym, y0, y1 = stats[m]
            st = _point_style(i)
            ax.errorbar(
                [xm],
                [ym],
                xerr=[[xm - x0], [x1 - xm]],
                yerr=[[ym - y0], [y1 - ym]],
                color=st["color"],
                marker=st["marker"],
                markersize=5,
                linestyle="",
                elinewidth=0.8,
                capsize=1.5,
                label=labels[m],
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("memory bandwidth (GB/s)")
        ax.set_ylabel(f"{precision} FMA peak (GFLOP/s)")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        title = f"machine balance · {precision}"
        if dropped:
            title += f" · no probes: {', '.join(dropped)}"
        ax.set_title(title, loc="left")
        _finish(fig, df)
    return BenchFigure("balance", fig, df)


def cross_machine(df, *, max_series: int = 12, max_entries: int = 30) -> BenchFigure:
    """Entries across machines, from :func:`mtl5.bench.analyze.cross_machine`.

    A dot plot: one row per entry, one marker per machine at the median of
    its runs, with a bar from the least to the most its runs measured. Rates
    are on a log axis, so a factor reads the same everywhere; percentages of
    the roofline are on a linear one. A hollow marker is a result MTL5 flagged
    as cache-resident or with a suspect work model in any of the machine's
    runs, which is why a result can sit above its roofline.
    """
    df = df.copy()  # the figure keeps the frame it was drawn from
    metrics = sorted(df["metric"].dropna().unique()) if "metric" in df else []
    if len(metrics) != 1:
        raise ValueError(f"a comparison shows one metric; the frame has {metrics or 'none'}")
    metric = metrics[0]
    names = sorted(df["machine"].dropna().unique())
    entries = sorted(df["entry_id"].unique())
    _check_series(len(names), max_series, "machines")
    _check_series(len(entries), max_entries, "entries")
    labels = _machine_labels(df)
    entry_labels = _entry_labels(entries)
    stats = df.groupby(["entry_id", "machine"])["value"].agg(["median", "min", "max", "count"])
    flags = [c for c in ("cache_resident", "suspect_model") if c in df]
    flagged = None
    if flags:
        # Series.map, column by column: DataFrame.map needs pandas 2.1.
        hit = df[flags[0]].map(_true)
        for col in flags[1:]:
            hit = hit | df[col].map(_true)
        flagged = df.assign(_f=hit.astype(bool)).groupby(["entry_id", "machine"])["_f"].any()
    n = len(names)
    step = min(0.8 / n, 0.14)
    with style.house_style():
        fig = _figure(style.WIDTH_DOUBLE, 0.9 + 0.3 * len(entries))
        ax = fig.subplots()
        for j, m in enumerate(names):
            st = _point_style(j)
            off = (j - (n - 1) / 2) * step
            ys, xs, hollow = [], [], []
            for i, e in enumerate(entries):
                if (e, m) not in stats.index:
                    continue
                r = stats.loc[(e, m)]
                ys.append(i + off)
                xs.append(r["median"])
                hollow.append(flagged is not None and bool(flagged.loc[(e, m)]))
                if r["count"] > 1:
                    ax.plot([r["min"], r["max"]], [i + off] * 2, color=st["color"],
                            linestyle="-", marker="", linewidth=0.9, zorder=2)  # fmt: skip
            solid = [k for k, h in enumerate(hollow) if not h]
            ax.plot([xs[k] for k in solid], [ys[k] for k in solid], linestyle="",
                    label=labels[m], markersize=4, zorder=3, **st)  # fmt: skip
            open_ = [k for k, h in enumerate(hollow) if h]
            if open_:
                # A label starting with "_" stays out of the legend.
                ax.plot([xs[k] for k in open_], [ys[k] for k in open_], linestyle="",
                        label=f"_{labels[m]}, flagged", markersize=4, zorder=3,
                        markerfacecolor="white", **st)  # fmt: skip
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([entry_labels[e] for e in entries])
        ax.set_ylim(len(entries) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        unit = df["unit"].dropna().iloc[0] if "unit" in df and df["unit"].notna().any() else metric
        if metric == "roofline_pct":
            ax.set_xlim(0, max(105.0, float(stats["max"].max()) * 1.05))
            ax.axvline(100, color=style.MUTED, linestyle="--", linewidth=0.8, zorder=1)
        else:
            ax.set_xscale("log")
        ax.set_xlabel(unit)
        handles, texts = ax.get_legend_handles_labels()
        if flagged is not None and flagged.any():
            from matplotlib.lines import Line2D

            handles.append(Line2D([], [], color=style.MUTED, marker="o", linestyle="",
                                  markersize=4, markerfacecolor="white"))  # fmt: skip
            texts.append("hollow: flagged by MTL5\n(cache-resident or suspect model)")
        ax.legend(handles, texts, loc="lower left", bbox_to_anchor=(1.01, 0.0))
        ax.set_title(f"across machines · {unit}", loc="left")
        _finish(fig, df)
    return BenchFigure("compare", fig, df)


#: The figures by kind, as :func:`from_csv` and the command line name them.
FIGURES = {
    "card": capability_card,
    "roofline": roofline,
    "scaling": thread_scaling,
    "rate": rate_vs_size,
    "balance": machine_balance,
    "compare": cross_machine,
}


# Columns that are text even when they look like numbers or like "NA": an
# all-digit commit stays a string with its leading zeros, a machine named NA
# stays named.
TEXT_COLUMNS = frozenset(
    {
        "run_id", "machine", "machine_dir", "machine_fingerprint", "label", "backend",
        "build_config", "state_class", "git_commit", "run_timestamp", "tier",
        "entry_id", "area", "op", "dtype", "shape", "operation", "metric", "unit",
        "cpu_vendor", "cpu_brand", "cpu_arch", "cpu_simd", "build_isa", "simd_backend",
        "compiler", "compiler_version", "os_name", "os_version", "hw_fma_isa",
        "blas_vendor",
    }
)  # fmt: skip


def read_frame(path: str | os.PathLike):
    """Read a frame :meth:`BenchFigure.save` wrote, as the figure had it.

    Only an empty field is missing. The text columns stay text; every other
    column is numeric where all its values are.
    """
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    for col in df.columns:
        if col in TEXT_COLUMNS:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            pass
    return df


def from_csv(path: str | os.PathLike, kind: str, **kwargs) -> BenchFigure:
    """Draw a figure again from the frame :meth:`BenchFigure.save` wrote."""
    if kind not in FIGURES:
        raise ValueError(f"unknown figure {kind!r}; use one of {sorted(FIGURES)}")
    return FIGURES[kind](read_frame(path), **kwargs)
