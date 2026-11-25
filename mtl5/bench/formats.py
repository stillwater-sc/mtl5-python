"""The legacy benchmark formats: MTL5's CSVs and ``bench_blas.py`` JSON.

``mtl5_bench`` writes one self-describing JSON schema. Everything before it
wrote CSV, one dialect per harness, and this repository's ``bench_blas.py``
writes its own JSON. Each dialect is described here declaratively, as a
:class:`Format`: the columns that identify it, the columns that name a
measurement (its parameters), the columns that hold measured values and their
units, and the columns that index repeated samples.

Every format flattens to the same long table, ``measurements``: one row per
(file row, metric), with the parameters in a map. That keeps one schema for a
dozen dialects, and a new harness needs a ``Format`` entry, not a table.

A format is matched by its header, not by its file name. A header no format
matches is recorded as ``deferred`` with the header in the reason, so a new
dialect shows up in the ingest report instead of disappearing.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field

__all__ = ["Format", "FORMATS", "BENCH_BLAS", "match_csv", "ParsedTable", "parse_csv"]


@dataclass(frozen=True)
class Format:
    """One benchmark dialect.

    ``signature``
        Columns the header must contain for this format to match. The first
        format in :data:`FORMATS` whose signature the header contains wins, so
        a more specific format comes before a more general one.
    ``params``
        Columns that identify what was measured. Together with the run's
        comparability key they make the series key. A column listed here but
        missing from a file (an older variant) is simply absent from its keys.
    ``metrics``
        Measured columns and their units.
    ``repeat``
        A column that numbers repeated samples (``round``, ``run``). It is
        stored on each measurement but is not part of the series, so repeats
        are points of one series.
    ``valid_if``
        For a metric, a ``(column, value)`` pair the row must have for the
        value to count, as in the solver scoreboards, where ``native_s`` means
        something only when ``native_status`` is ``ok``.
    ``exempt``
        Sidecar keys this producer does not record because they do not apply
        or the file itself carries them, so their absence is not a provenance
        gap.
    ``from_columns``
        Sidecar keys a column of the file supplies (``compiler_version`` in
        the interleave CSVs). Every row must agree.
    ``aliases``
        Sidecar keys this producer spells differently.
    ``file_in_entry``
        Put the file's name (without suffix) into each entry, for a format
        whose file name is the only thing naming what was compared: the KLU
        and SuperLU scoreboards share columns, and only ``klu_scoreboard``
        and ``superlu_scoreboard`` tell them apart.
    ``version``
        The format definition's version; part of every series key.
    """

    name: str
    signature: tuple[str, ...]
    params: tuple[str, ...]
    metrics: dict[str, str]
    repeat: str | None = None
    valid_if: dict[str, tuple[str, str]] = field(default_factory=dict)
    exempt: frozenset[str] = frozenset()
    from_columns: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    file_in_entry: bool = False
    version: int = 1

    @property
    def kind(self) -> str:
        return f"{self.name}/{self.version}"


# Keys no legacy producer writes: they are mtl5_bench's own (its benchmark
# name, its machine label, its size class and its SIMD backend report).
_NOT_LEGACY = frozenset({"benchmark", "machine", "class", "simd_backend"})

# The shell harnesses' sidecar dialect (run_column_sweep.sh, the SuperLU and
# KLU scoreboards) spells some keys its own way.
_HARNESS_ALIASES = {"host": "preflight_host", "cpu": "cpu_brand", "commit": "git_commit"}

_SCOREBOARD_METRICS = {
    f"{solver}_{m}": unit
    for solver in ("native", "ext", "snode")
    for m, unit in (("s", "s"), ("fill", "entries"), ("resid", "1"))
}

#: The CSV dialects, most specific first.
FORMATS: tuple[Format, ...] = (
    Format(
        name="nc_model_timing",
        signature=("model", "best_seconds", "gflops"),
        params=("model", "dtype", "m", "n", "k", "threads"),
        metrics={"best_seconds": "s", "gflops": "GFLOP/s"},
        exempt=_NOT_LEGACY,
    ),
    Format(
        # The model's predictions only: no column is a measurement.
        name="nc_model_sweep",
        signature=("model", "nc", "packedB_bytes"),
        params=("model", "dtype", "m", "n", "k", "threads"),
        metrics={},
        exempt=_NOT_LEGACY,
    ),
    Format(
        name="blocking_ab",
        signature=("arm", "dtype", "m", "n", "k", "threads", "min_s", "gflops"),
        params=("arm", "dtype", "m", "n", "k", "threads"),
        metrics={"min_s": "s", "gflops": "GFLOP/s"},
        exempt=_NOT_LEGACY,
    ),
    Format(
        name="blocking_sweep",
        signature=("kc", "mc", "threads", "round", "gflops"),
        params=("kc", "mc", "threads"),
        metrics={"gflops": "GFLOP/s"},
        repeat="round",
        exempt=_NOT_LEGACY,
    ),
    Format(
        name="vendor_nested",
        signature=("vendor", "pool", "round", "arm", "n", "batch", "seconds"),
        params=("vendor", "pool", "arm", "n", "batch"),
        metrics={"seconds": "s", "gflops": "GFLOP/s"},
        repeat="round",
        exempt=_NOT_LEGACY,
    ),
    Format(
        name="pool_latency",
        signature=("arm", "threads", "gap_us", "round", "p50_ns"),
        params=("arm", "threads", "gap_us", "spin", "metric"),
        metrics={k: "ns" for k in ("p50_ns", "p90_ns", "p99_ns", "p999_ns", "max_ns")},
        repeat="round",
        exempt=_NOT_LEGACY,
    ),
    Format(
        # bench_all and the int_arms sweep share it.
        name="bench_all",
        signature=("operation", "backend", "size", "median_ns"),
        params=("operation", "backend", "size"),
        metrics={
            "median_ns": "ns",
            "min_ns": "ns",
            "max_ns": "ns",
            "mean_ns": "ns",
            "stddev_ns": "ns",
            "gflops": "GFLOP/s",
        },
        exempt=_NOT_LEGACY,
    ),
    Format(
        name="interleave_rows",
        signature=("session", "run", "kernel", "type", "width", "ms", "gbps"),
        params=("compiler", "cpu", "vector_registers", "n", "kernel", "type", "width"),
        metrics={"ms": "ms", "gbps": "GB/s"},
        repeat="run",
        # One pinned core and no thread pool.
        exempt=_NOT_LEGACY | {"pool_size"},
        from_columns={"compiler_version": "compiler_version"},
    ),
    Format(
        name="column_sweep",
        signature=("kernel", "type", "m", "n", "sweep_ns", "rows_ns"),
        params=("kernel", "type", "m", "n"),
        metrics={"sweep_ns": "ns", "sweep1_ns": "ns", "rows_ns": "ns", "public_ns": "ns"},
        exempt=_NOT_LEGACY | {"pool_size"},
        aliases=_HARNESS_ALIASES | {"build_flags": "cxx_flags"},
    ),
    Format(
        # The KLU and SuperLU scoreboards; the snode columns are newer.
        name="solver_scoreboard",
        signature=("matrix", "native_status", "native_s", "ext_status", "ext_s"),
        params=("matrix",),
        metrics=_SCOREBOARD_METRICS,
        valid_if={m: (f"{m.split('_')[0]}_status", "ok") for m in _SCOREBOARD_METRICS},
        exempt=_NOT_LEGACY | {"pool_size"},
        aliases=_HARNESS_ALIASES,
        file_in_entry=True,
    ),
)

#: ``bench_blas.py --json``: ``{"meta": ..., "results": [...]}``.
BENCH_BLAS = Format(
    name="bench_blas",
    signature=("kernel", "dtype", "n", "status"),
    params=("kernel", "dtype", "n", "accumulator"),
    metrics={"seconds_per_op": "s", "gflops": "GFLOP/s"},
    valid_if={"seconds_per_op": ("status", "ok"), "gflops": ("status", "ok")},
    exempt=_NOT_LEGACY,
)


@dataclass
class ParsedTable:
    header: list[str]
    rows: list[dict[str, str]]
    comments: list[str]


def parse_csv(text: str) -> ParsedTable:
    """Split leading ``#`` comment lines from a CSV and parse the rest.

    Raises ``ValueError`` for an empty table, a row with more fields than the
    header, or a duplicate column name.
    """
    lines = text.splitlines()
    comments = []
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        line = lines.pop(0)
        if line.strip():
            comments.append(line.lstrip("#").strip())
    if not lines:
        raise ValueError("no header row")
    reader = csv.reader(io.StringIO("\n".join(lines)))
    header = [h.strip() for h in next(reader)]
    if len(set(header)) != len(header):
        raise ValueError("the header repeats a column")
    rows = []
    for lineno, fields in enumerate(reader, start=2 + len(comments)):
        if not fields or not any(f.strip() for f in fields):
            continue
        if len(fields) > len(header):
            raise ValueError(f"line {lineno} has {len(fields)} fields, the header {len(header)}")
        rows.append(
            {h: (fields[i].strip() if i < len(fields) else "") for i, h in enumerate(header)}
        )
    return ParsedTable(header, rows, comments)


def match_csv(header: list[str]) -> Format | None:
    """The first format whose signature the header contains."""
    cols = set(header)
    for fmt in FORMATS:
        if set(fmt.signature) <= cols:
            return fmt
    return None


def number(text: str) -> float | None:
    """A finite float from a CSV field, else None (empty, 'nan', 'n/a')."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
