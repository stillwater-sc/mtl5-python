"""Comparison tables across machines, from the frames of :mod:`mtl5.bench.analyze`.

The frames of :func:`~mtl5.bench.analyze.machines` and
:func:`~mtl5.bench.analyze.cross_machine` have one row per run. A table
reduces each machine's runs (its sessions) to their median, and keeps the
spread and the run count beside it, so a table never shows one session as if
it were the machine.

:func:`to_markdown` writes a table as a Markdown pipe table, with the frame's
provenance line under it. It needs no extra dependency, and the same frame
always gives the same text.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

from .style import number

__all__ = ["machine_table", "compare_table", "to_markdown"]

# (column in the machines frame, heading, digits); None digits is text.
_MACHINE_COLUMNS = (
    ("cpu_brand", "CPU", None),
    ("cpu_arch", "Arch", None),
    ("threads", "Threads", 0),
    ("hw_fma_isa", "FMA unit", None),
    ("fma_peak_f64_gflops", "f64 peak, GFLOP/s", 3),
    ("fma_peak_f32_gflops", "f32 peak, GFLOP/s", 3),
    ("bandwidth_gbytes_per_s", "Bandwidth, GB/s", 3),
    ("ridge_f64", "f64 ridge, FLOP/byte", 3),
    ("last_level_cache_bytes", "LLC, MiB", 3),
)


def _text(values) -> Any:
    seen = [v for v in dict.fromkeys(values) if _present(v)]
    return " / ".join(str(v) for v in seen) if seen else None


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return not (isinstance(v, str) and not v.strip())


def machine_table(df):
    """One row per machine: what it is, and the median of its runs' probes.

    The numeric columns are each machine's median over its runs; ``runs``
    counts them, and ``commit`` is the commit they share.
    """
    import pandas as pd

    rows = []
    for machine, g in df.groupby("machine", sort=True):
        row: dict[str, Any] = {"Machine": machine}
        for col, heading, digits in _MACHINE_COLUMNS:
            if col not in g:
                row[heading] = None
            elif digits is None:
                row[heading] = _text(g[col])
            else:
                values = pd.to_numeric(g[col], errors="coerce")
                if col == "last_level_cache_bytes":
                    # The probe reports 0 when it found no cache size.
                    values = values.where(values > 0) / 2**20
                row[heading] = values.median() if values.notna().any() else None
        row["Runs"] = g["run_id"].nunique()
        row["Commit"] = _text(str(c)[:7] for c in g["git_commit"] if _present(c))
        rows.append(row)
    return pd.DataFrame(rows)


def _entry_label(entry_id: str) -> str:
    return "/".join(entry_id.split("/")[:3])


def compare_table(df, *, spread: bool = False):
    """Entries by machines: each machine's median ``value`` over its runs.

    One row per entry and one column per machine, labelled with its thread
    count. A machine that did not run an entry has no value there. With
    ``spread``, each cell is the text ``median (min–max)`` wherever a machine
    has more than one run, instead of the number.
    """
    import pandas as pd

    threads = df.groupby("machine")["threads"].agg(lambda t: sorted({int(x) for x in t}))
    heading = {m: f"{m} ({'/'.join(map(str, t))} thr)" for m, t in threads.items()}
    stats = df.groupby(["entry_id", "machine"])["value"].agg(["median", "min", "max", "count"])
    if spread:
        cells = stats.apply(
            lambda r: (
                _g(r["median"])
                if r["count"] < 2
                else f"{_g(r['median'])} ({_g(r['min'])}–{_g(r['max'])})"
            ),
            axis=1,
        )
    else:
        cells = stats["median"]
    table = cells.unstack("machine")
    table = table.reindex(columns=sorted(table.columns))
    table.columns = [heading[m] for m in table.columns]
    labels = {e: _entry_label(e) for e in table.index}
    if len(set(labels.values())) < len(labels):
        labels = {e: e for e in table.index}
    table.index = [labels[e] for e in table.index]
    table.index.name = "Entry"
    table = table.reset_index()
    unit = _text(df["unit"]) if "unit" in df else None
    table.attrs["unit"] = unit
    return pd.DataFrame(table)


def _g(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "" if not _present(v) else str(v)
    if not math.isfinite(x):
        return ""
    return number(x, digits)


def to_markdown(table, *, digits: int = 3, footer: str | None = None) -> str:
    """A table as a Markdown pipe table; numbers to ``digits`` significant digits.

    ``footer``, such as :func:`mtl5.bench.plot.provenance` of the frame the
    table came from, follows the table as an italic line.
    """
    cols = list(table.columns)

    def is_number(v) -> bool:
        return isinstance(v, numbers.Real) and not isinstance(v, bool)

    numeric = {
        c: all(is_number(v) or not _present(v) for v in table[c])
        and any(_present(v) for v in table[c])
        for c in cols
    }

    def cell(c, v) -> str:
        if not _present(v):
            return ""
        if numeric[c]:
            if isinstance(v, numbers.Integral):
                return str(int(v))
            return _g(v, digits)
        return str(v).replace("|", "\\|")

    lines = [
        "| " + " | ".join(str(c).replace("|", "\\|") for c in cols) + " |",
        "|" + "|".join("---:" if numeric[c] else "---" for c in cols) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(cell(c, row[c]) for c in cols) + " |")
    text = "\n".join(lines) + "\n"
    if footer:
        text += f"\n*{footer}*\n"
    return text
