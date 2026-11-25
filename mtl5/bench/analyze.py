"""DataFrames for the core figures, read from the benchmark database.

Each function here answers one question and returns a pandas DataFrame with
one row per plotted point. The figures in :mod:`mtl5.bench.plot` take these
frames and nothing else, so a figure is reproducible from its frame: save the
frame, and the same figure can be drawn again from it without the database.

Every frame carries the provenance columns (:data:`PROVENANCE_COLUMNS`): the
run, its machine, its commit, its time and its provenance tier. The figures'
provenance footers are computed from them.

A database of another schema version is refused with ``ValueError``: it is
a cache, rebuilt with ``mtl5-bench ingest --rebuild``.

Which runs are read
-------------------
Only current runs are read, and by default only runs in the ``complete``
tier; pass ``tiers=("complete", "incomplete")`` to include runs whose
provenance has gaps (most of MTL5's older CSVs). The footer of a figure says
when it does. Results whose correctness check did not pass are never read.

A ``machine`` is matched against the run's machine label, the data directory
it was committed under, and the machine fingerprint. Where a question needs
one run (a capability card, a roofline), the latest matching run is used
unless ``run_id`` picks one.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

__all__ = [
    "METRICS",
    "trend",
    "PROVENANCE_COLUMNS",
    "machines",
    "cross_machine",
    "capability",
    "roofline",
    "thread_scaling",
    "rate_vs_size",
]

#: The columns every frame carries, for the provenance footer.
PROVENANCE_COLUMNS = ("run_id", "machine", "git_commit", "run_timestamp", "tier")

_RUN_COLUMNS = """
    r.run_id,
    coalesce(r.machine, r.machine_dir) AS machine,
    r.machine_dir,
    r.machine_fingerprint,
    r.label,
    r.class,
    r.threads,
    r.backend,
    r.build_config,
    r.state_class,
    r.git_commit,
    r.git_dirty,
    r.run_timestamp,
    r.tier
"""

_MATCH_MACHINE = "(? IS NULL OR ? IN (r.machine, r.machine_dir, r.machine_fingerprint))"

# The probes a roofline needs, and the columns they become.
_ROOF_PROBES = {
    "fma_peak_f64_gflops": "peak_f64_gflops",
    "fma_peak_f32_gflops": "peak_f32_gflops",
    "build_fma_peak_f64_gflops": "build_peak_f64_gflops",
    "build_fma_peak_f32_gflops": "build_peak_f32_gflops",
    "bandwidth_gbytes_per_s": "bandwidth_gbytes_per_s",
}

# bench_all columns that describe one sample rather than the point measured.
_SAMPLE_COLUMNS = frozenset({"iterations"})

# Sysinfo fields a capability card shows.
_CARD_SYSINFO = (
    "cpu_simd",
    "build_isa",
    "simd_backend",
    "compiler",
    "compiler_version",
    "os_name",
    "os_version",
)


def _connection(db: Any):
    """A DuckDB connection: ``db`` itself, or the database file it names."""
    if isinstance(db, (str, os.PathLike)):
        import duckdb

        return duckdb.connect(os.fspath(db), read_only=True), True
    return db, False


def _check_schema(con, db: Any) -> None:
    """Refuse a database of another schema version, or not a benchmark one.

    The database is a cache, and a column can change its meaning between
    versions (in version 3, ``git_dirty`` came to cover ``tree_git_dirty``),
    so an older database would answer with stale values.
    """
    from .schema import SCHEMA_VERSION

    name = os.fspath(db) if isinstance(db, (str, os.PathLike)) else "the database"
    try:
        meta = dict(
            con.execute(
                "SELECT key, value FROM meta WHERE key IN ('application', 'schema_version')"
            ).fetchall()
        )
    except Exception as exc:  # duckdb.CatalogException: no meta table
        raise ValueError(f"{name} is not an mtl5.bench database") from exc
    if meta.get("application") != "mtl5.bench":
        raise ValueError(f"{name} is not an mtl5.bench database")
    if meta.get("schema_version") != str(SCHEMA_VERSION):
        raise ValueError(
            f"{name} has benchmark schema version {meta.get('schema_version')}, this "
            f"mtl5.bench reads version {SCHEMA_VERSION}; rebuild it with "
            "'mtl5-bench ingest ROOT --db ... --rebuild'"
        )


class _Db:
    def __init__(self, db: Any):
        self.con, self.owned = _connection(db)
        try:
            _check_schema(self.con, db)
        except BaseException:
            if self.owned:
                self.con.close()
            raise

    def __enter__(self):
        return self.con

    def __exit__(self, *exc):
        if self.owned:
            self.con.close()


def _machines(con, tiers: Sequence[str]) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT coalesce(machine, machine_dir) FROM current_runs "
        "WHERE schema = 'mtl5-bench/1' AND list_contains(?::VARCHAR[], tier) ORDER BY 1",
        [list(tiers)],
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def _runs(
    con,
    machine: str | None,
    *,
    label: str | None = None,
    threads: int | None = None,
    run_id: str | None = None,
    tiers: Sequence[str] = ("complete",),
):
    """The matching mtl5-bench runs, latest first."""
    df = con.execute(
        f"""
        SELECT {_RUN_COLUMNS}
        FROM current_runs r
        WHERE r.schema = 'mtl5-bench/1'
          AND list_contains(?::VARCHAR[], r.tier)
          AND {_MATCH_MACHINE}
          AND (? IS NULL OR r.label = ?)
          AND (? IS NULL OR r.threads = ?)
          AND (? IS NULL OR r.run_id = ?)
        ORDER BY r.run_timestamp DESC NULLS LAST, r.run_id
        """,
        [list(tiers), machine, machine, label, label, threads, threads, run_id, run_id],
    ).df()
    return df


def _one_run(con, machine, *, label, threads, run_id, tiers):
    runs = _runs(con, machine, label=label, threads=threads, run_id=run_id, tiers=tiers)
    if runs.empty:
        what = [f"machine={machine!r}"]
        if label is not None:
            what.append(f"label={label!r}")
        if threads is not None:
            what.append(f"threads={threads}")
        if run_id is not None:
            what.append(f"run_id={run_id!r}")
        raise LookupError(
            f"no {'/'.join(tiers)} mtl5-bench run with {', '.join(what)}; "
            f"machines: {', '.join(_machines(con, tiers)) or 'none'}"
        )
    return runs.iloc[[0]].reset_index(drop=True)


def _probes(con, run_ids: list[str]):
    import pandas as pd

    rows = con.execute(
        "SELECT run_id, name, value, text FROM probes WHERE list_contains(?::VARCHAR[], run_id)",
        [run_ids],
    ).fetchall()
    out: dict[str, dict[str, Any]] = {r: {} for r in run_ids}
    for run, name, value, text in rows:
        out[run][name] = value if value is not None else text
    return pd.DataFrame.from_dict(out, orient="index")


def capability(
    db,
    machine: str,
    *,
    label: str | None = "native",
    threads: int | None = None,
    run_id: str | None = None,
    tiers: Sequence[str] = ("complete",),
):
    """One row describing a machine: its CPU, its build and its probes.

    The probes are the run's own measurements of the machine: the FMA peaks
    (``fma_peak_*``, ``hw_fma_peak_*``, ``build_fma_peak_*``), the memory
    bandwidths, the last-level cache. Each probe is a column.
    """
    with _Db(db) as con:
        run = _one_run(con, machine, label=label, threads=threads, run_id=run_id, tiers=tiers)
        rid = run.at[0, "run_id"]
        m = con.execute(
            "SELECT cpu_vendor, cpu_brand, cpu_arch, cpu_logical_cores FROM machines "
            "WHERE fingerprint = ?",
            [run.at[0, "machine_fingerprint"]],
        ).fetchone() or (None, None, None, None)
        sysinfo = con.execute("SELECT sysinfo FROM runs WHERE run_id = ?", [rid]).fetchone()[0]
        sysinfo = dict(sysinfo or {})
        probes = _probes(con, [rid])
    for col, value in zip(("cpu_vendor", "cpu_brand", "cpu_arch", "cpu_logical_cores"), m):
        run[col] = value
    for key in _CARD_SYSINFO:
        run[key] = sysinfo.get(key)
    for name in sorted(probes.columns):
        if name != "threads":  # the run's own column already
            run[name] = probes.at[rid, name]
    return run


def roofline(
    db,
    machine: str,
    *,
    label: str | None = "native",
    threads: int | None = 1,
    run_id: str | None = None,
    areas: Sequence[str] | None = None,
    tiers: Sequence[str] = ("complete",),
):
    """Every entry of one run placed on its machine's roofline.

    One row per entry with a work model (``flops`` and ``bytes`` both
    positive): its arithmetic intensity (``flops / bytes``), its rate, MTL5's
    roofline bound for it and the percentage reached, plus the run's peaks and
    bandwidth as columns, so the frame holds the ceilings too.
    """
    with _Db(db) as con:
        run = _one_run(con, machine, label=label, threads=threads, run_id=run_id, tiers=tiers)
        rid = run.at[0, "run_id"]
        df = con.execute(
            """
            SELECT entry_id, area, op, dtype, shape, flops, bytes,
                   flops / bytes AS intensity, gflops, roofline_gflops, roofline_pct,
                   cache_resident, suspect_model
            FROM results
            WHERE run_id = ? AND check_ok IS TRUE
              AND flops > 0 AND bytes > 0 AND gflops > 0
              AND (?::VARCHAR[] IS NULL OR list_contains(?::VARCHAR[], area))
            ORDER BY entry_id
            """,
            [rid, None if areas is None else list(areas), None if areas is None else list(areas)],
        ).df()
        probes = _probes(con, [rid])
    for probe, col in _ROOF_PROBES.items():
        df[col] = probes.at[rid, probe] if probe in probes.columns else None
    for col in run.columns:
        df[col] = run.at[0, col]
    return df


def _named_code(runs):
    """The runs that name their code: a recorded commit and a clean tree."""
    import pandas as pd

    known = runs["git_commit"].map(lambda c: isinstance(c, str) and bool(c.strip()))
    clean = runs["git_dirty"].map(lambda d: d is not None and d is not pd.NA and d == 0)
    return runs[known & clean.astype(bool)]


def _state_without_placement(con, run_id: str) -> str:
    from .sysinfo import state_class

    fields = con.execute("SELECT sysinfo FROM runs WHERE run_id = ?", [run_id]).fetchone()[0]
    return state_class(dict(fields or {}), placement=False)


def _matches(entry_id: str, wanted: Sequence[str]) -> bool:
    """Whether ``entry_id`` is one of ``wanted`` or under one, by whole segments.

    The area may be left out: ``"gemm/f64"`` matches ``blas3/gemm/f64/...``.
    """
    without_area = entry_id.split("/", 1)[1] if "/" in entry_id else entry_id
    for w in wanted:
        w = w.rstrip("/")
        for e in (entry_id, without_area):
            if e == w or e.startswith(w + "/"):
                return True
    return False


def thread_scaling(
    db,
    machine: str,
    *,
    label: str | None = "native",
    entries: Sequence[str] | None = None,
    tiers: Sequence[str] = ("complete",),
):
    """Speedup and parallel efficiency against the 1-thread run.

    Only runs with a recorded commit and a clean tree are compared. The
    reference is the latest multi-threaded run. The runs compared are the
    latest run for each thread count among runs of the same physical machine,
    label, problem class, backend (with the BLAS vendor), commit, build
    configuration and
    machine state (less the CPUs it ran on) as the reference, so only the
    thread count differs. ``entries`` keeps the entries whose id is, or starts
    with, one of the given prefixes
    (``"blas3/gemm"`` keeps ``blas3/gemm/f64/...`` and ``blas3/gemm/f32/...``).

    One row per (entry, thread count), the 1-thread rows included, with
    ``speedup = median_s(1 thread) / median_s`` and
    ``efficiency = speedup / threads``.
    """
    with _Db(db) as con:
        runs = _runs(con, machine, label=label, tiers=tiers)
        if runs.empty:
            _one_run(con, machine, label=label, threads=None, run_id=None, tiers=tiers)
        # Two runs with no commit, or two dirty builds of one commit, may run
        # different code, and their ratio would not be a speedup.
        runs = _named_code(runs)
        multi = runs[runs["threads"] > 1]
        if multi.empty:
            raise LookupError(
                f"no multi-threaded {'/'.join(tiers)} mtl5-bench run of {machine!r}"
                + (f" with label={label!r}" if label is not None else "")
                + " with a recorded commit and a clean tree"
            )
        # The reference: the latest multi-threaded run, the most threads first
        # among runs of one time (a session's runs share its directory's time).
        ref = multi.sort_values(
            ["run_timestamp", "threads"], ascending=False, na_position="last", kind="stable"
        ).iloc[0]
        runs["_state"] = [_state_without_placement(con, rid) for rid in runs["run_id"]]
        # The backend too: the label is "blas" for every vendor, and only the
        # backend says which.
        key = [
            "machine_fingerprint",
            "label",
            "class",
            "backend",
            "build_config",
            "_state",
            "git_commit",
        ]
        # As strings, so a missing value matches a missing value.
        k = runs[key].astype(str)
        same = runs[(k == k.loc[ref.name]).all(axis=1)]
        # Latest first already, so the first row per thread count is its latest.
        chosen = (
            same.drop_duplicates("threads")
            .drop(columns="_state")
            .sort_values("threads")
            .reset_index(drop=True)
        )
        if 1 not in set(chosen["threads"]):
            raise LookupError(
                f"no 1-thread run comparable with the latest multi-threaded {machine!r} run "
                f"({ref['run_id']}): thread scaling needs a 1-thread run of the "
                "same commit, build and machine state"
            )
        res = con.execute(
            """
            SELECT run_id, entry_id, area, op, dtype, shape, median_s, gflops
            FROM results
            WHERE list_contains(?::VARCHAR[], run_id) AND check_ok IS TRUE AND median_s > 0
            """,
            [list(chosen["run_id"])],
        ).df()
    df = res.merge(chosen, on="run_id")
    if entries is not None:
        # As bool: an empty frame maps to an empty object Series, which as an
        # indexer selects columns, not rows.
        df = df[df["entry_id"].map(lambda e: _matches(e, entries)).astype(bool)]
    base = df[df["threads"] == 1].set_index("entry_id")["median_s"]
    df = df[df["entry_id"].isin(base.index)].copy()
    df["speedup"] = df["entry_id"].map(base) / df["median_s"]
    df["efficiency"] = df["speedup"] / df["threads"]
    return df.sort_values(["entry_id", "threads"]).reset_index(drop=True)


def rate_vs_size(
    db,
    machine: str | None = None,
    *,
    operations: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    metric: str = "gflops",
    tiers: Sequence[str] = ("complete",),
):
    """A rate (GFLOP/s by default) against problem size, per operation and backend.

    Read from the ``bench_all`` CSVs (the BLAS and LAPACK sweeps, the scaling
    sweeps), which measure each operation over a range of sizes. Repeated
    samples of one point are reduced to their median, with the count in
    ``samples``; rows of one point that differ in another column are refused
    with ``ValueError``, since they are different points. ``machine=None``
    reads every machine, including files committed outside a machine
    directory.
    """
    import json

    ops = None if operations is None else list(operations)
    bks = None if backends is None else list(backends)
    with _Db(db) as con:
        raw = con.execute(
            f"""
            SELECT {_RUN_COLUMNS},
                   m.params->>'operation' AS operation,
                   m.params->>'backend' AS series_backend,
                   TRY_CAST(m.params->>'size' AS BIGINT) AS size,
                   m.metric, m.unit, m.value, m.attributes
            FROM measurements m
            JOIN current_runs r USING (run_id)
            WHERE m.format LIKE 'bench_all/%'
              AND m.metric = ?
              AND m.valid AND m.value IS NOT NULL
              AND list_contains(?::VARCHAR[], r.tier)
              AND {_MATCH_MACHINE}
              AND (?::VARCHAR[] IS NULL OR list_contains(?::VARCHAR[], m.params->>'operation'))
              AND (?::VARCHAR[] IS NULL OR list_contains(?::VARCHAR[], m.params->>'backend'))
              AND TRY_CAST(m.params->>'size' AS BIGINT) IS NOT NULL
            """,
            [metric, list(tiers), machine, machine, ops, ops, bks, bks],
        ).df()
    point = ["run_id", "operation", "series_backend", "size"]
    # Rows of one point may differ only in what describes a sample. A row that
    # differs in another column (a thread count, say) is another point, which
    # this frame has no column for: refuse it rather than take a median across.
    raw["_other"] = [
        json.dumps({k: v for k, v in json.loads(a or "{}").items() if k not in _SAMPLE_COLUMNS},
                   sort_keys=True)
        for a in raw["attributes"]
    ]  # fmt: skip
    mixed = raw.groupby(point, dropna=False)["_other"].nunique()
    if (mixed > 1).any():
        run, op, backend, size = mixed[mixed > 1].index[0]
        rows = raw[(raw["run_id"] == run) & (raw["operation"] == op)]
        cols = sorted({k for o in rows["_other"] for k in json.loads(o)})
        raise ValueError(
            f"rows of {op} ({backend}) at size {size} in run {run} differ in "
            f"{', '.join(cols)}: rate_vs_size reduces repeated samples of one point, "
            "and these are different points"
        )
    first = raw.drop(columns=["value", "attributes", "_other"]).drop_duplicates(point)
    agg = (
        raw.groupby(point, dropna=False)["value"].agg(value="median", samples="count").reset_index()
    )
    df = first.merge(agg, on=point).sort_values(
        ["machine", "operation", "series_backend", "size", "run_id"], na_position="first"
    )
    # The CSV's backend column names the series; the run's backend is the file's.
    df = df.drop(columns=["backend"]).rename(columns={"series_backend": "backend"})
    return df.reset_index(drop=True)


# --- across machines ------------------------------------------------------------

#: The result columns :func:`cross_machine` can compare, and their units.
METRICS = {
    "gflops": "GFLOP/s",
    "roofline_pct": "% of roofline",
    "gbytes_per_s": "GB/s",
    "median_s": "s",
}

# What a machine's runs must share to be one measurement of it: everything but
# the session. The thread count is chosen per machine before this.
_SESSION_KEY = (
    "machine_fingerprint",
    "label",
    "class",
    "backend",
    "build_config",
    "state_class",
    "git_commit",
    "threads",
)


def _threads_arg(threads: int | str) -> int | str:
    if threads == "max":
        return threads
    try:
        n = int(threads)
    except (TypeError, ValueError):
        raise ValueError(f"threads must be 'max' or a positive integer, not {threads!r}") from None
    if n < 1:
        raise ValueError(f"threads must be 'max' or a positive integer, not {threads!r}")
    return n


def _sessions(
    con,
    *,
    label: str | None,
    threads: int | str,
    machines: Sequence[str] | None,
    tiers: Sequence[str],
):
    """Each machine's comparable runs: its latest commit's sessions.

    For each physical machine, the thread count is ``threads``, or with
    ``"max"`` the most the machine has a run of. The reference is the latest
    such run; the machine's runs are those that share its configuration and
    commit (:data:`_SESSION_KEY`), so they differ only in the session. MTL5's
    method takes two sessions per commit, so most machines have two.
    """
    threads = _threads_arg(threads)
    runs = _named_code(_runs(con, None, label=label, tiers=tiers))
    if machines is not None:
        wanted = set(machines)
        hit = runs[["machine", "machine_dir", "machine_fingerprint"]].isin(wanted).any(axis=1)
        missing = wanted - set(runs.loc[hit, ["machine", "machine_dir", "machine_fingerprint"]]
                               .stack().dropna())  # fmt: skip
        if missing:
            raise LookupError(
                f"no {'/'.join(tiers)} mtl5-bench run of {', '.join(sorted(missing))}; "
                f"machines: {', '.join(_machines(con, tiers)) or 'none'}"
            )
        runs = runs[hit]
    if threads == "max":
        most = runs.groupby("machine_fingerprint")["threads"].transform("max")
        runs = runs[runs["threads"] == most]
    else:
        runs = runs[runs["threads"] == threads]
        if machines is not None:
            # A named machine with runs at other thread counts only would
            # otherwise drop out of the comparison without a word.
            names = runs[["machine", "machine_dir", "machine_fingerprint"]].stack().dropna()
            absent = set(machines) - set(names)
            if absent:
                raise LookupError(
                    f"no {threads}-thread run of {', '.join(sorted(absent))} with a "
                    "recorded commit and a clean tree"
                )
    if runs.empty:
        raise LookupError(
            f"no {'/'.join(tiers)} mtl5-bench run with label={label!r}, threads={threads} "
            "and a recorded commit and a clean tree"
        )
    # As strings, so a missing value matches a missing value.
    key = runs[list(_SESSION_KEY)].astype(str)
    keep = []
    for _, group in runs.groupby("machine_fingerprint", sort=True):
        ref = group.sort_values(
            ["run_timestamp", "run_id"], ascending=[False, True], na_position="last"
        ).index[0]
        keep.extend(group.index[(key.loc[group.index] == key.loc[ref]).all(axis=1)])
    out = runs.loc[keep]
    return out.sort_values(["machine", "run_timestamp", "run_id"]).reset_index(drop=True)


def machines(
    db,
    *,
    label: str | None = "native",
    threads: int | str = "max",
    machines: Sequence[str] | None = None,
    tiers: Sequence[str] = ("complete",),
):
    """Every machine's description and probes, one row per run.

    A machine contributes its comparable runs (the latest commit's sessions,
    at ``threads``, or its most threads with ``"max"``). Each row has the
    machine's CPU and build, the run's probes as columns (FMA peaks,
    bandwidths, the last-level cache) and ``ridge_f64``, the f64 FMA peak
    over the bandwidth: the arithmetic intensity, in FLOP/byte, where a
    kernel stops being bandwidth-bound. ``machines`` keeps the named ones.
    """
    with _Db(db) as con:
        runs = _sessions(con, label=label, threads=threads, machines=machines, tiers=tiers)
        hw = con.execute(
            "SELECT fingerprint AS machine_fingerprint, cpu_vendor, cpu_brand, cpu_arch, "
            "cpu_logical_cores FROM machines"
        ).df()
        sysinfo = {
            rid: dict(m or {})
            for rid, m in con.execute(
                "SELECT run_id, sysinfo FROM runs WHERE list_contains(?::VARCHAR[], run_id)",
                [list(runs["run_id"])],
            ).fetchall()
        }
        probes = _probes(con, list(runs["run_id"]))
    df = runs.merge(hw, on="machine_fingerprint", how="left")
    for key in _CARD_SYSINFO:
        df[key] = [sysinfo[r].get(key) for r in df["run_id"]]
    for name in sorted(probes.columns):
        if name != "threads":
            df[name] = [probes.at[r, name] for r in df["run_id"]]
    peak = df.get("fma_peak_f64_gflops")
    bw = df.get("bandwidth_gbytes_per_s")
    if peak is not None and bw is not None:
        df["ridge_f64"] = [
            p / b if _positive(p) and _positive(b) else None for p, b in zip(peak, bw)
        ]
    return df


def _positive(v) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def cross_machine(
    db,
    entries: Sequence[str],
    *,
    metric: str = "gflops",
    label: str | None = "native",
    threads: int | str = "max",
    machines: Sequence[str] | None = None,
    tiers: Sequence[str] = ("complete",),
):
    """Entries compared across machines, one row per (run, entry).

    Each machine contributes its comparable runs, as in :func:`machines`.
    ``entries`` keeps the entries whose id is, or starts with, one of the
    given prefixes. ``value`` is the result's ``metric``
    (:data:`METRICS`); results whose check did not pass are left out, and an
    entry a machine did not run is simply absent for it. ``cache_resident``
    and ``suspect_model`` carry MTL5's own flags: a result above its roofline
    is usually one of them.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; use one of {', '.join(METRICS)}")
    with _Db(db) as con:
        runs = _sessions(con, label=label, threads=threads, machines=machines, tiers=tiers)
        res = con.execute(
            f"""
            SELECT run_id, entry_id, area, op, dtype, shape, {metric} AS value,
                   cache_resident, suspect_model
            FROM results
            WHERE list_contains(?::VARCHAR[], run_id) AND check_ok IS TRUE
              AND {metric} IS NOT NULL
            """,
            [list(runs["run_id"])],
        ).df()
    res = res[res["entry_id"].map(lambda e: _matches(e, entries)).astype(bool)]
    df = res.merge(runs, on="run_id")
    df["metric"] = metric
    df["unit"] = METRICS[metric]
    return df.sort_values(["entry_id", "machine", "run_timestamp", "run_id"]).reset_index(drop=True)


# --- trends ---------------------------------------------------------------------


def trend(
    db,
    machine: str,
    entries: Sequence[str],
    *,
    metric: str = "gflops",
    label: str | None = "native",
    threads: int = 1,
):
    """An entry's history on one machine, one row per (run, entry).

    Read from the ``trend_results`` view only: current runs in the complete
    tier whose check passed, so every row is a comparable measurement. Each
    row carries its ``series_key`` and the run fields that make it
    (``backend``, ``build_config``, ``state_class``, ``class``,
    ``definition_version``), so :mod:`mtl5.bench.trends` can tell a series
    break from a change. ``entries`` are ids or prefixes of whole segments;
    the area may be left out (``"gemm/f64"``).
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; use one of {', '.join(METRICS)}")
    threads = _threads_arg(threads)
    if threads == "max":
        raise ValueError("a trend holds the thread count fixed: give a number")
    with _Db(db) as con:
        df = con.execute(
            f"""
            SELECT t.run_id, coalesce(t.machine, r.machine_dir) AS machine, r.machine_dir,
                   t.machine_fingerprint, t.label, t.threads, t.backend, t.build_config,
                   t.state_class, t.class, t.git_commit, r.git_dirty, t.run_timestamp, r.tier,
                   t.entry_id, t.definition_version, t.series_key, t.area, t.op, t.dtype,
                   t.shape, t.{metric} AS value
            FROM trend_results t
            JOIN current_runs r USING (run_id)
            WHERE {_MATCH_MACHINE.replace("r.machine,", "t.machine, r.machine_dir,")}
              AND (? IS NULL OR t.label = ?)
              AND t.threads = ?
              AND t.{metric} > 0
            ORDER BY t.entry_id, t.run_timestamp, t.run_id
            """,
            [machine, machine, label, label, threads],
        ).df()
        if df.empty:
            known = _machines(con, ("complete",))
    if df.empty:
        raise LookupError(
            f"no complete run of {machine!r} with label={label!r} and threads={threads}; "
            f"machines: {', '.join(known) or 'none'}"
        )
    df = df[df["entry_id"].map(lambda e: _matches(e, entries)).astype(bool)]
    df["metric"] = metric
    df["unit"] = METRICS[metric]
    return df.reset_index(drop=True)
