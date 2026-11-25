"""Ingest MTL5 benchmark data into the benchmark database.

``ingest(root, db)`` walks ``root`` (normally ``<mtl5>/benchmarks/data``) and
gives every file found there a row in the ``files`` table, so every file is
accounted for: ingested, ingested as incomplete, rejected with a reason, or
set aside with a reason (a report, a figure, a CSV header no format
matches). It reads ``mtl5-bench/1`` JSON, the output of ``mtl5_bench``, and
the legacy formats in :mod:`mtl5.bench.formats` (MTL5's CSVs and
``bench_blas.py`` JSON), each with its ``.sysinfo`` sidecar.

The rules that decide a run's provenance tier:

* **rejected:** the JSON does not parse, has another schema, lacks a field the
  schema requires, or contradicts its sidecar (label, machine, class, SIMD
  backend or benchmark name). Nothing from it reaches the result tables.
* **incomplete:** ingested, but excluded from trend queries. The sidecar is
  missing, lacks a key in :data:`~mtl5.bench.sysinfo.REQUIRED_KEYS`, repeats a
  key with conflicting values, or disagrees with the JSON on the thread
  count; the build was dirty, so its commit does not identify the code; or the
  run directory's name is not a UTC timestamp, so the run has no time.
* **complete** (file status ``ingested``): none of the above.

Re-ingesting an unchanged tree appends nothing. The database is a cache:
``rebuild=True`` drops it and ingests from scratch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import formats as fm
from . import sysinfo as si
from .schema import DDL, SCHEMA_VERSION

__all__ = [
    "INGEST_STATUSES",
    "FileOutcome",
    "IngestReport",
    "connect",
    "ingest",
]

#: The status a file row can have, and what it means.
INGEST_STATUSES = {
    "ingested": "an mtl5-bench/1 run with complete provenance",
    "incomplete": "an mtl5-bench/1 run ingested without complete provenance; excluded from trends",
    "rejected": "a data file that could not be ingested",
    "duplicate": "the same content as a run already ingested from another path",
    "paired": "a .sysinfo sidecar, read with its data file",
    "deferred": "a CSV whose header no format in mtl5.bench.formats matches",
    "ignored": "not benchmark data (a report, figure, manifest or model-only table)",
    "removed": "a file a previous ingest saw that is no longer there",
}

_SCHEMA = "mtl5-bench/1"
_APPLICATION = "mtl5.bench"  # the meta marker that says this package created a database
_SIDECAR = ".sysinfo"
_REPORT_SUFFIXES = {".md", ".txt", ".rst", ".log"}
_FIGURE_SUFFIXES = {".png", ".svg", ".pdf", ".jpg", ".jpeg"}
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_TIMESTAMP_RE = re.compile(r"[0-9]{8}T[0-9]{6}Z")
# The fields the result columns are read from, and the JSON types accepted.
_RESULT_FIELDS = ("area", "op", "dtype", "shape")


@dataclass
class FileOutcome:
    """What the ingest did with one file."""

    path: str
    kind: str
    status: str
    reasons: list[str] = field(default_factory=list)
    run_id: str | None = None
    new: bool = True  # False when the file was unchanged since the last ingest


@dataclass
class IngestReport:
    """The outcome of :func:`ingest`, one :class:`FileOutcome` per file."""

    root: str
    outcomes: list[FileOutcome]

    def counts(self) -> dict[str, int]:
        """The number of files per status."""
        out: dict[str, int] = {}
        for o in self.outcomes:
            out[o.status] = out.get(o.status, 0) + 1
        return dict(sorted(out.items()))

    @property
    def new_rows(self) -> int:
        """How many file rows this ingest appended."""
        return sum(o.new for o in self.outcomes)

    def problems(self) -> list[FileOutcome]:
        """The rejected and incomplete files, with their reasons."""
        return [o for o in self.outcomes if o.status in ("rejected", "incomplete")]

    def __str__(self) -> str:
        lines = [f"ingest of {self.root}: {len(self.outcomes)} files, {self.new_rows} new rows"]
        lines += [f"  {status:<11} {n}" for status, n in self.counts().items()]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("mtl5.bench needs duckdb and pandas: pip install 'mtl5[bench]'") from exc
    return duckdb


def _is_bench_database(duckdb, path: Path) -> bool:
    """True when path is a DuckDB file this package created.

    The header check comes first so that a file that is not DuckDB is never
    handed to DuckDB; then a read-only open looks for the ``application``
    marker every version of this schema writes into ``meta``. Another
    application's DuckDB file, even one with a ``meta`` table, lacks it.
    """
    if not path.is_file():
        return False
    with path.open("rb") as f:
        if f.read(12)[8:12] != b"DUCK":
            return False
    try:
        probe = duckdb.connect(str(path), read_only=True)
    except duckdb.Error:
        return False
    try:
        row = probe.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'meta'"
        ).fetchone()
        if not row[0]:
            return False
        marker = probe.execute("SELECT value FROM meta WHERE key = 'application'").fetchone()
        return marker is not None and marker[0] == _APPLICATION
    except duckdb.Error:
        return False
    finally:
        probe.close()


def connect(db: str | os.PathLike = ":memory:", *, rebuild: bool = False):
    """Open (or create) a benchmark database and return the DuckDB connection.

    ``rebuild=True`` deletes an existing database file first, and refuses to
    delete one this package did not create. A database written by another
    schema version raises, since it is a cache and is rebuilt rather than
    migrated; the version is read before the schema is applied, because the
    views would not bind against another version's tables.
    """
    duckdb = _duckdb()
    path = str(db)
    if rebuild and path != ":memory:":
        target, wal = Path(path), Path(path + ".wal")
        if target.exists():
            if not _is_bench_database(duckdb, target):
                raise ValueError(
                    f"{path} is not an mtl5 benchmark database; --rebuild deletes only one"
                )
        elif wal.exists():
            raise ValueError(f"{wal} has no database beside it; remove it by hand")
        for p in (target, wal):
            if p.exists():
                p.unlink()
    con = duckdb.connect(path)
    try:
        _initialize(con, path)
    except BaseException:
        con.close()
        raise
    return con


def _initialize(con, path: str) -> None:
    tables = {
        name
        for (name,) in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = current_database()"
        ).fetchall()
    }
    if tables and "meta" not in tables:
        # Someone else's database: installing the schema would mark it as
        # ours, and a later --rebuild would then delete it.
        raise RuntimeError(f"{path} is a DuckDB database this package did not create")
    if tables:
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
        if meta.get("application") != _APPLICATION:
            raise RuntimeError(f"{path} is a DuckDB database this package did not create")
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            raise RuntimeError(
                f"{path} has benchmark schema version {meta.get('schema_version')}, this "
                f"mtl5 writes {SCHEMA_VERSION}; rerun the ingest with --rebuild "
                f"(mtl5-bench ingest ROOT --db {path} --rebuild)"
            )
        con.execute(DDL)
        return
    # A new database: the schema and the marker commit together, so an
    # interrupted first open leaves nothing a retry would call foreign.
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(DDL)
        con.execute(
            "INSERT INTO meta VALUES ('application', ?), ('schema_version', ?)",
            [_APPLICATION, str(SCHEMA_VERSION)],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


# --------------------------------------------------------------------------
# Parsing one mtl5-bench/1 run
# --------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
        except OverflowError:  # an int too large for a double
            return None
        return f if math.isfinite(f) else None
    return None


_INT64 = (-(2**63), 2**63 - 1)  # the BIGINT columns' range


def _int(v: Any) -> int | None:
    """An integer that fits a BIGINT column, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        try:
            v = int(v.strip())
        except ValueError:
            return None
    if isinstance(v, int) and _INT64[0] <= v <= _INT64[1]:
        return v
    return None


def _str(v: Any) -> str | None:
    return v if isinstance(v, str) else None


def _bool(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _run_timestamp(rel: str) -> datetime | None:
    parts = rel.split("/")
    # fullmatch first: strptime alone accepts 20261111T19041Z as 19:04:01.
    if len(parts) < 2 or not _TIMESTAMP_RE.fullmatch(parts[-2]):
        return None
    try:
        # Stored as naive UTC (TIMESTAMP), which round-trips without pytz.
        return datetime.strptime(parts[-2], _TIMESTAMP_FORMAT)
    except ValueError:
        return None


class _Rejected(Exception):
    pass


def _validate_json(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise _Rejected("the JSON is not an object")
    if doc.get("schema") != _SCHEMA:
        raise _Rejected(f"schema is {doc.get('schema')!r}, not {_SCHEMA!r}")
    for key in ("label", "machine", "class", "simd_backend"):
        if not isinstance(doc.get(key), str):
            raise _Rejected(f"missing or non-string field {key!r}")
    probes = doc.get("probes")
    if not isinstance(probes, dict):
        raise _Rejected("missing field 'probes'")
    if _int(probes.get("threads")) is None:
        raise _Rejected("probes.threads is missing or not an integer")
    results = doc.get("results")
    if not isinstance(results, list):
        raise _Rejected("missing field 'results'")
    seen: set[str] = set()
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            raise _Rejected(f"results[{i}] is not an object")
        for key in _RESULT_FIELDS:
            if not isinstance(r.get(key), str):
                raise _Rejected(f"results[{i}] has no string {key!r}")
        if _num(r.get("median_s")) is None:
            raise _Rejected(f"results[{i}] has no finite median_s")
        for key in ("median_s", "min_s"):
            value = _num(r.get(key))
            if value is not None and value < 0:
                raise _Rejected(f"results[{i}].{key} is negative")
        eid = _entry_id(r)
        if eid in seen:
            raise _Rejected(f"results[{i}] repeats entry {eid!r}")
        seen.add(eid)
    skipped = doc.get("skipped", [])
    if not isinstance(skipped, list) or not all(isinstance(s, dict) for s in skipped):
        raise _Rejected("'skipped' is not a list of objects")
    for i, s in enumerate(skipped):
        for key in ("area", "op", "dtype", "reason"):
            if s.get(key) is not None and not isinstance(s[key], str):
                raise _Rejected(f"skipped[{i}].{key} is not a string")
    _check_text(doc)


def _check_text(doc: Any) -> None:
    """Reject a JSON document holding text that cannot be encoded."""
    try:
        # A lone surrogate (\ud800) decodes from JSON but cannot be encoded,
        # and every key and row is encoded on its way to a hash or DuckDB.
        json.dumps(doc, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        raise _Rejected("the JSON contains text that is not valid Unicode") from None


def _entry_id(r: dict) -> str:
    return "/".join(r[k] for k in _RESULT_FIELDS)


@dataclass
class _Run:
    run_id: str
    tier: str
    reasons: list[str]
    run_row: dict
    machine_row: dict
    probes: list[dict]
    results: list[dict]
    skipped: list[dict]
    entries: list[dict]
    measurements: list[dict] = field(default_factory=list)


def _dirty(fields: dict[str, str]) -> bool | None:
    """Whether the build or the tree it was built from was dirty.

    ``git_dirty`` is the build's own record and ``tree_git_dirty`` the
    preflight's look at the checkout. Either one set means the commit does
    not identify the code, so the stored ``git_dirty`` is both. None when
    the sidecar records neither.
    """
    keys = [k for k in ("git_dirty", "tree_git_dirty") if k in fields]
    if not keys:
        return None
    return any(fields[k] != "0" for k in keys)


def _sidecar_gaps(fields: dict[str, str], required) -> list[str]:
    """The reasons a sidecar's required keys are missing or empty."""
    reasons = []
    missing = [k for k in required if k not in fields]
    if missing:
        reasons.append("sidecar lacks " + ", ".join(missing))
    empty = [k for k in required if k in fields and not fields[k] and k not in si.EMPTY_ALLOWED]
    if empty:
        reasons.append("sidecar has empty " + ", ".join(empty))
    return reasons


def _parse_run(rel: str, run_id: str, doc: Any, sidecar_text: str | None) -> _Run:
    """Validate and flatten one run. Raises _Rejected."""
    _validate_json(doc)
    reasons: list[str] = []

    if sidecar_text is None:
        fields: dict[str, str] = {}
        reasons.append("no .sysinfo sidecar")
    else:
        fields, problems = si.parse_sysinfo(sidecar_text)
        reasons += problems + _sidecar_gaps(fields, si.REQUIRED_KEYS)
        if "benchmark" in fields and fields["benchmark"] != "mtl5_bench":
            raise _Rejected(f"sidecar says benchmark={fields['benchmark']}, not mtl5_bench")
        for key in ("label", "machine", "class", "simd_backend"):
            if key in fields and fields[key] != doc[key]:
                raise _Rejected(f"sidecar says {key}={fields[key]!r}, the JSON {doc[key]!r}")

    threads = _int(doc["probes"]["threads"])
    if "pool_size" in fields:
        pool = _int(fields["pool_size"])
        if pool is None:
            reasons.append(f"sidecar pool_size={fields['pool_size']!r} is not an integer")
        elif pool != threads:
            reasons.append(f"sidecar pool_size={pool} but probes.threads={threads}")
    side_vendor = fields.get("blas_vendor") or ""
    # Stripped like every sidecar value, so whitespace is no vendor.
    probe_vendor = (_str(doc["probes"].get("blas_vendor")) or "").strip()
    if side_vendor and probe_vendor and side_vendor != probe_vendor:
        raise _Rejected(f"sidecar says blas_vendor={side_vendor!r}, the probes {probe_vendor!r}")
    vendor = side_vendor or probe_vendor
    if doc["label"] == "blas" and not vendor:
        # Without the vendor, OpenBLAS and MKL runs would share one series.
        reasons.append("a blas run with no blas_vendor in the sidecar or the probes")
    if _dirty(fields):
        reasons.append("dirty build: the commit does not identify the code")
    timestamp = _run_timestamp(rel)
    if timestamp is None:
        reasons.append("the run directory name is not a UTC timestamp (YYYYMMDDTHHMMSSZ)")

    tier = "incomplete" if reasons else "complete"

    fingerprint = si.machine_fingerprint(fields)
    backend = si.backend(doc["label"], {"blas_vendor": vendor})
    build = si.build_config(fields)
    state = si.state_class(fields)
    config_key = si.short_hash(
        "\n".join(
            [
                si.machine_identity(fields),
                state,
                backend,
                build,
                f"threads {threads}",
                f"class {doc['class']}",
            ]
        )
    )

    parts = rel.split("/")
    run_row = {
        "run_id": run_id,
        "path": rel,
        "schema": _SCHEMA,
        "tier": tier,
        "reasons": reasons,
        "run_timestamp": timestamp,
        "time_source": "directory" if timestamp is not None else None,
        "machine": doc["machine"],
        "machine_dir": parts[0] if len(parts) > 1 else None,
        "machine_fingerprint": fingerprint,
        "label": doc["label"],
        "class": doc["class"],
        "threads": threads,
        "backend": backend,
        "build_config": build,
        "state_class": state,
        "config_key": config_key,
        "git_commit": fields.get("git_commit"),
        "git_dirty": _dirty(fields),
        "os_version": fields.get("os_version"),
        "blas_vendor": vendor or None,
        "thermal_before_c": _float_str(fields.get("thermal_before_c")),
        "thermal_after_c": _float_str(fields.get("thermal_after_c")),
        "loadavg_1m": _float_str(fields.get("loadavg_1m")),
        "sysinfo": fields,
    }
    machine_row = {
        "fingerprint": fingerprint,
        "identity": si.machine_identity(fields),
        "cpu_vendor": fields.get("cpu_vendor"),
        "cpu_brand": fields.get("cpu_brand"),
        "cpu_arch": fields.get("cpu_arch"),
        "cpu_logical_cores": _int(fields.get("cpu_logical_cores")),
        "host": fields.get("preflight_host", "").lower().split(".")[0] or None,
    }

    probes = []
    for name, value in doc["probes"].items():
        num = _num(value)
        probes.append(
            {
                "run_id": run_id,
                "name": name,
                "value": num,
                "text": None if num is not None else (None if value is None else str(value)),
            }
        )

    results, entries = [], []
    for r in doc["results"]:
        eid = _entry_id(r)
        version = _int(r.get("version")) or 0
        check = r.get("check") if isinstance(r.get("check"), dict) else {}
        results.append(
            {
                "run_id": run_id,
                "entry_id": eid,
                "definition_version": version,
                "series_key": si.short_hash(f"{config_key}\n{eid}\nv{version}"),
                "area": r["area"],
                "op": r["op"],
                "dtype": r["dtype"],
                "shape": r["shape"],
                "median_s": _num(r.get("median_s")),
                "min_s": _num(r.get("min_s")),
                "samples": _int(r.get("samples")),
                "flops": _num(r.get("flops")),
                "bytes": _num(r.get("bytes")),
                "gflops": _num(r.get("gflops")),
                "gbytes_per_s": _num(r.get("gbytes_per_s")),
                "roofline_gflops": _num(r.get("roofline_gflops")),
                "roofline_pct": _num(r.get("roofline_pct")),
                "cache_resident": _bool(r.get("cache_resident")),
                "suspect_model": _bool(r.get("suspect_model")),
                "speedup": _num(r.get("speedup")),
                "efficiency": _num(r.get("efficiency")),
                "check_ok": _bool(check.get("ok")),
                "check_error": _num(check.get("error")),
                "check_tolerance": _num(check.get("tolerance")),
                "check_metric": check.get("metric")
                if isinstance(check.get("metric"), str)
                else None,
            }
        )
        entries.append(
            {
                "entry_id": eid,
                "definition_version": version,
                "area": r["area"],
                "op": r["op"],
                "dtype": r["dtype"],
                "shape": r["shape"],
            }
        )

    skipped = [
        {
            "run_id": run_id,
            "area": s.get("area"),
            "op": s.get("op"),
            "dtype": s.get("dtype"),
            "reason": s.get("reason"),
        }
        for s in doc.get("skipped", [])
    ]
    return _Run(run_id, tier, reasons, run_row, machine_row, probes, results, skipped, entries)


def _parse_table_run(
    rel: str,
    run_id: str,
    fmt: fm.Format,
    table: fm.ParsedTable,
    sidecar_text: str | None,
    *,
    git_time: datetime | None,
    exempt_listed: bool,
    git_problem: str | None = None,
) -> _Run:
    """Validate and flatten one legacy-format file (a CSV or bench_blas JSON).

    The provenance rules are mtl5_bench's, with the format's exemptions,
    aliases and columns applied to the sidecar first. A run directory's
    timestamp gives the run time; failing that, the commit that added the
    file does (``time_source`` says which). Raises _Rejected.
    """
    reasons: list[str] = []
    if sidecar_text is None:
        fields: dict[str, str] = {}
        if exempt_listed:
            reasons.append("no .sysinfo sidecar (grandfathered in .sidecar-exempt)")
        else:
            reasons.append("no .sysinfo sidecar")
    else:
        fields, problems = si.parse_sysinfo(sidecar_text)
        reasons += problems
        for alias, key in fmt.aliases.items():
            if alias in fields and key not in fields:
                fields[key] = fields[alias]
    for key, column in fmt.from_columns.items():
        if column not in table.header:
            continue  # an older variant without the column: the sidecar stands
        values = {row[column] for row in table.rows}
        if len(values) > 1:
            raise _Rejected(f"column {column!r} has {len(values)} different values")
        if values:
            value = values.pop()
            if key in fields and fields[key] != value:
                raise _Rejected(f"sidecar says {key}={fields[key]!r}, column {column!r} {value!r}")
            fields[key] = value
    if sidecar_text is not None:
        required = [k for k in si.REQUIRED_KEYS if k not in fmt.exempt]
        reasons += _sidecar_gaps(fields, required)
    if "pool_size" in fields and _int(fields["pool_size"]) is None:
        reasons.append(f"sidecar pool_size={fields['pool_size']!r} is not an integer")
    if fields.get("label") == "blas" and not fields.get("blas_vendor"):
        # As for mtl5_bench: without the vendor, OpenBLAS and MKL share a series.
        reasons.append("a blas run with no blas_vendor in the sidecar")
    if _dirty(fields):
        reasons.append("dirty build: the commit does not identify the code")

    timestamp, time_source = _run_timestamp(rel), "directory"
    if timestamp is None:
        timestamp, time_source = git_time, "git"
    if timestamp is None:
        time_source = None
        reasons.append(
            "no run time: the directory name is not a UTC timestamp, "
            f"and {git_problem or 'git has no commit adding the file'}"
        )
    if not table.rows:
        reasons.append("the table has no data rows")

    tier = "incomplete" if reasons else "complete"
    label = fields.get("label") or rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    vendor = fields.get("blas_vendor", "")
    fingerprint = si.machine_fingerprint(fields)
    backend = si.backend(label, {"blas_vendor": vendor})
    build = si.build_config(fields)
    state = si.state_class(fields)
    pool = fields.get("pool_size", "?")
    config_key = si.short_hash(
        "\n".join(
            [
                si.machine_identity(fields),
                state,
                backend,
                build,
                f"pool {pool}",
                f"format {fmt.kind}",
            ]
        )
    )

    parts = rel.split("/")
    run_row = {
        "run_id": run_id,
        "path": rel,
        "schema": fmt.kind,
        "tier": tier,
        "reasons": reasons,
        "run_timestamp": timestamp,
        "time_source": time_source,
        "machine": None,
        "machine_dir": parts[0] if len(parts) > 1 else None,
        "machine_fingerprint": fingerprint,
        "label": label,
        "class": None,
        "threads": _int(fields.get("pool_size")),
        "backend": backend,
        "build_config": build,
        "state_class": state,
        "config_key": config_key,
        "git_commit": fields.get("git_commit"),
        "git_dirty": _dirty(fields),
        "os_version": fields.get("os_version"),
        "blas_vendor": vendor or None,
        "thermal_before_c": _float_str(fields.get("thermal_before_c")),
        "thermal_after_c": _float_str(fields.get("thermal_after_c")),
        "loadavg_1m": _float_str(fields.get("loadavg_1m")),
        "sysinfo": fields,
    }
    machine_row = {
        "fingerprint": fingerprint,
        "identity": si.machine_identity(fields),
        "cpu_vendor": fields.get("cpu_vendor"),
        "cpu_brand": fields.get("cpu_brand"),
        "cpu_arch": fields.get("cpu_arch"),
        "cpu_logical_cores": _int(fields.get("cpu_logical_cores")),
        "host": fields.get("preflight_host", "").lower().split(".")[0] or None,
    }

    params_present = [p for p in fmt.params if p in table.header]
    metrics_present = [m for m in fmt.metrics if m in table.header]
    skip = set(params_present) | set(metrics_present) | ({fmt.repeat} if fmt.repeat else set())
    seen: dict[str, int] = {}
    measurements = []
    stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    for i, row in enumerate(table.rows):
        params = {p: row[p] for p in params_present}
        if fmt.file_in_entry:
            params = {"file": stem, **params}
        entry_id = f"{fmt.name}:" + ",".join(f"{k}={v}" for k, v in params.items())
        sample = seen.get(entry_id, 0)
        seen[entry_id] = sample + 1
        attributes = {k: v for k, v in row.items() if k not in skip}
        for metric in metrics_present:
            gate = fmt.valid_if.get(metric)
            valid = gate is None or row.get(gate[0]) == gate[1]
            measurements.append(
                {
                    "run_id": run_id,
                    "row_index": i,
                    "metric": metric,
                    "format": fmt.kind,
                    "entry_id": entry_id,
                    "series_key": si.short_hash(f"{config_key}\n{entry_id}\n{metric}"),
                    "params": json.dumps(params, sort_keys=True),
                    "repeat": row.get(fmt.repeat) if fmt.repeat else None,
                    "sample": sample,
                    "unit": fmt.metrics[metric],
                    "value": fm.number(row[metric]),
                    "valid": valid,
                    "attributes": json.dumps(attributes, sort_keys=True),
                }
            )
    return _Run(run_id, tier, reasons, run_row, machine_row, [], [], [], [], measurements)


def _bench_blas_table(doc: dict) -> fm.ParsedTable:
    """bench_blas.py's results as a table of strings. Raises _Rejected."""
    results = doc.get("results")
    if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
        raise _Rejected("bench_blas JSON: 'results' is not a list of objects")
    # Seeded with the metrics, so a document whose every result failed (and
    # so carries no metric keys) still records each result as invalid.
    header: list[str] = list(fm.BENCH_BLAS.metrics)
    rows = []
    for i, r in enumerate(results):
        for key in fm.BENCH_BLAS.signature:
            if key not in r:
                raise _Rejected(f"bench_blas JSON: results[{i}] has no {key!r}")
        row = {}
        for k, v in r.items():
            if k not in header:
                header.append(k)
            if isinstance(v, (dict, list)):
                raise _Rejected(f"bench_blas JSON: results[{i}].{k} is not a scalar")
            row[k] = "" if v is None else str(v).lower() if isinstance(v, bool) else str(v)
        rows.append(row)
    # A failed result omits the metrics a passing one has: every row gets
    # every column, so a missing value is an empty field, as in a CSV.
    for row in rows:
        for key in header:
            row.setdefault(key, "")
    return fm.ParsedTable(header, rows, [])


def _float_str(v: str | None) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------
# Walking the tree
# --------------------------------------------------------------------------


def _raise(exc: OSError) -> None:
    raise exc


def _walk(root: Path, exclude: set[Path]) -> list[str]:
    """Every file under root, as sorted '/'-separated relative paths.

    Directories whose name starts with '.' (``.git``) and ``__pycache__`` are
    not walked; dot-files are. ``exclude`` holds resolved paths to leave out:
    the database itself, when it lives under the root. A directory that
    cannot be read raises instead of being skipped, since a skipped directory
    would look like deleted files.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=_raise):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        for name in sorted(filenames):
            full = Path(dirpath) / name
            try:
                # stat() raises on a permission error, where is_file() would
                # (on Python 3.14+) say False and the file would look removed.
                mode = full.stat().st_mode
            except FileNotFoundError:  # a dangling symlink, or removed mid-scan
                continue
            if stat.S_ISREG(mode) and full.resolve() not in exclude:
                out.append(full.relative_to(root).as_posix())
    return sorted(out)


def _database_files(con) -> set[Path]:
    """The files of the databases attached to con, and their WALs."""
    out: set[Path] = set()
    for (path,) in con.execute(
        "SELECT path FROM duckdb_databases() WHERE coalesce(path, '') <> ''"
    ).fetchall():
        p = Path(path).resolve()
        out |= {p, Path(str(p) + ".wal")}
    return out


def _classify(rel: str, all_paths: set[str]) -> tuple[str, str, list[str]] | None:
    """(kind, status, reasons) for a file that is not a JSON data file, else None."""
    name = rel.rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()
    if name.endswith(_SIDECAR):
        data = rel[: -len(_SIDECAR)]
        if data in all_paths:
            return "sidecar", "paired", []
        return "sidecar", "rejected", [f"sidecar without its data file {data.rsplit('/', 1)[-1]}"]
    if suffix in (".json", ".csv"):
        return None
    if suffix in _REPORT_SUFFIXES:
        return "report", "ignored", []
    if suffix in _FIGURE_SUFFIXES:
        return "figure", "ignored", []
    if name.startswith("."):
        return "manifest", "ignored", []
    return "other", "ignored", ["not a recognised benchmark file"]


# Variables that point git at a repository other than the one -C finds.
_GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
)


def _git_env() -> dict[str, str]:
    """The environment without repository-location variables, so that
    ``git -C root`` discovers root's own repository."""
    return {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_VARS}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        timeout=120,
        env=_git_env(),
    ).stdout.decode("utf-8", "surrogateescape")


def _git_add_times(root: Path) -> tuple[dict[str, datetime], str | None]:
    """When each file under root was first committed, as naive UTC.

    Returns the times, keyed by paths relative to root as the walk produces
    them, and why there are none when git cannot give them: git missing, root
    not in a work tree, a failure or timeout, or a shallow clone, whose
    grafted root commit would date every file to the same moment.
    """
    try:
        if _git(root, "rev-parse", "--is-shallow-repository").strip() == "true":
            return {}, "the clone is shallow, so git cannot date the file"
    except FileNotFoundError:
        return {}, "git is not installed"
    except subprocess.TimeoutExpired:
        return {}, "git timed out"
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip().splitlines()
        return {}, f"git failed: {detail[0] if detail else 'exit ' + str(exc.returncode)}"
    except OSError as exc:
        return {}, f"git could not run: {exc}"
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--diff-filter=A",
                # A rename is not an add of the new name unless renames are off.
                "--no-renames",
                "--relative",
                # -z: names are NUL-terminated and never quoted, and each
                # commit is "\0<time>\0\n" followed by its names.
                "-z",
                "--format=%x00%ct",
                "--name-only",
                "--",
                ".",
            ],
            capture_output=True,
            check=True,
            timeout=120,
            env=_git_env(),
        ).stdout.decode("utf-8", "surrogateescape")
    except subprocess.TimeoutExpired:
        return {}, "git timed out"
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"git log failed: {exc}"
    times: dict[str, datetime] = {}
    when = None
    after_time = False
    previous = ""
    for token in out.split("\x00"):
        if previous == "" and token.isdigit():
            # A name is never empty, so an empty token separates records and
            # the token after it is a commit time.
            try:
                when = datetime.fromtimestamp(int(token), timezone.utc).replace(tzinfo=None)
            except (OverflowError, OSError, ValueError):
                when = None
            after_time = True
        elif token:
            name = token[1:] if after_time and token.startswith("\n") else token
            after_time = False
            if when is not None:
                # git log runs newest first, so the last assignment is the first add.
                times[name] = when
        previous = token
    return times, None


def _sidecar_exempt(root: Path) -> set[str]:
    """The paths MTL5's .sidecar-exempt grandfathers, relative to root.

    The manifest lists paths from the repository root
    (``benchmarks/data/...``); the part after ``benchmarks/data/`` is the
    path under an ingest root of benchmarks/data.
    """
    manifest = root / ".sidecar-exempt"
    if not manifest.is_file():
        return set()
    out = set()
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        marker = "benchmarks/data/"
        out.add(line.split(marker, 1)[1] if marker in line else line)
    return out


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _path_provenance(rel: str) -> tuple[bytes, bytes, bytes]:
    parts = rel.split("/")
    machine_dir = parts[0] if len(parts) > 1 else ""
    stamp = parts[-2] if _run_timestamp(rel) is not None else ""
    # A CSV's name can be what it measured (klu_scoreboard, superlu_scoreboard),
    # so the same bytes under two names are two runs.
    name = parts[-1] if rel.lower().endswith(".csv") else ""
    return machine_dir.encode(), stamp.encode(), name.encode()


def _sha256(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(len(c).to_bytes(8, "little"))
        h.update(c)
    return h.hexdigest()


def ingest(
    root: str | os.PathLike,
    db: Any = ":memory:",
    *,
    rebuild: bool = False,
    time_from_git: bool = True,
) -> tuple[Any, IngestReport]:
    """Ingest every file under ``root`` into the database ``db``.

    ``db`` is a path, ``":memory:"``, or a connection from :func:`connect`.
    Returns ``(connection, report)``. One database holds one ingest root: its
    paths are relative to it, and a file missing from the root is recorded as
    removed, so ingesting a different root into the same database raises.

    A legacy-format file (a CSV, bench_blas.py JSON) outside a timestamp
    directory takes its run time from the commit that added it, when ``root``
    is in a git work tree and ``time_from_git`` is true; without one it is
    incomplete. An ``mtl5-bench/1`` run always needs its timestamp directory:
    mtl5_bench writes one, so a run outside it has been moved.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    owned = not hasattr(db, "execute")
    if not owned and rebuild:
        raise ValueError("rebuild=True needs the database path, not an open connection")
    con = connect(db, rebuild=rebuild) if owned else db
    try:
        return con, _ingest(root_path, con, time_from_git=time_from_git)
    except BaseException:
        if owned:
            con.close()
        raise


def _ingest(root_path: Path, con, *, time_from_git: bool = True) -> IngestReport:
    row = con.execute("SELECT value FROM meta WHERE key = 'root'").fetchone()
    if row is not None and row[0] != str(root_path):
        raise ValueError(
            f"this database was ingested from {row[0]}, not {root_path}; "
            "use another database or rebuild it"
        )
    # The generation this scan starts from. _write commits only if it is
    # still current, so two ingests on one database cannot interleave.
    generation = _generation(con)

    current = {
        path: (content_hash, kind, status, reasons, run_id)
        for path, content_hash, kind, status, reasons, run_id in con.execute(
            "SELECT path, content_hash, kind, status, reasons, run_id FROM current_files"
        ).fetchall()
    }
    known_runs = {
        run_id: (path, tier, list(reasons or []))
        for run_id, path, tier, reasons in con.execute(
            "SELECT run_id, path, tier, reasons FROM runs"
        ).fetchall()
    }

    paths = _walk(root_path, _database_files(con))
    path_set = set(paths)
    if time_from_git:
        git_times, git_problem = _git_add_times(root_path)
    else:
        git_times, git_problem = {}, "git run times are switched off"
    exempt_list = _sidecar_exempt(root_path)
    outcomes: list[FileOutcome] = []
    file_rows: list[dict] = []
    runs: list[_Run] = []

    def record(o: FileOutcome, content_hash: str) -> None:
        prev = current.get(o.path)
        unchanged = (
            prev is not None
            and prev[0] == content_hash
            and prev[2] != "removed"
            # A sidecar's status depends on its data file, not its bytes.
            and o.status in ("?", prev[2])
        )
        if unchanged:
            # Unchanged since the last ingest: report what is recorded.
            o.kind, o.status, o.reasons, o.run_id = prev[1], prev[2], list(prev[3] or []), prev[4]
            o.new = False
        else:
            file_rows.append(
                {
                    "path": o.path,
                    "content_hash": content_hash,
                    "kind": o.kind,
                    "status": o.status,
                    "reasons": o.reasons,
                    "run_id": o.run_id,
                }
            )
        outcomes.append(o)

    for rel in paths:
        full = root_path / rel
        if not _is_utf8(rel):
            # Python carries the undecodable bytes as surrogates, which
            # neither a hash input nor a VARCHAR accepts. The file is
            # accounted for under a key no walked path can have (walked paths
            # never start with '/'), lossless because it is the bytes in hex;
            # the reason shows the escaped name.
            raw = os.fsencode(rel)
            shown = raw.decode("utf-8", "backslashreplace")
            record(
                FileOutcome(
                    f"/non-utf8/{raw.hex()}",
                    "other",
                    "rejected",
                    [f"the file name is not UTF-8: {shown}"],
                ),
                _sha256(raw, full.read_bytes()),
            )
            continue
        classified = _classify(rel, path_set)
        if classified is not None:
            kind, status, reasons = classified
            record(FileOutcome(rel, kind, status, reasons), _sha256(full.read_bytes()))
            continue

        data = full.read_bytes()
        sidecar_path = root_path / (rel + _SIDECAR)
        sidecar = sidecar_path.read_bytes() if (rel + _SIDECAR) in path_set else None
        # Without a timestamp directory, the commit that added the file is the
        # run time, so committing it later must re-evaluate the run.
        git_time = None if _run_timestamp(rel) is not None else git_times.get(rel)
        stamp = git_time.isoformat().encode() if git_time is not None else b""
        # The marker byte tells an absent sidecar from an empty one.
        content_hash = _sha256(data, b"\x00" if sidecar is None else b"\x01" + sidecar, stamp)
        # The run is the content plus the provenance its path carries (the
        # machine directory and the timestamp directory), so a run moved to a
        # path that changes its tier is a new run, not a duplicate.
        run_id = _sha256(content_hash.encode(), *_path_provenance(rel))[:32]

        prev = current.get(rel)
        if prev is not None and prev[0] == content_hash and prev[2] != "removed":
            record(FileOutcome(rel, "?", "?"), content_hash)
            continue

        def finish(kind: str, parse, reject_kind: str | None = None) -> None:
            """Record a data file as a run: a duplicate, a restore, or parsed."""
            if run_id in known_runs:
                first_path, tier, reasons = known_runs[run_id]
                if first_path == rel:
                    # Removed and restored unchanged: the run is already there.
                    status = "ingested" if tier == "complete" else "incomplete"
                    o = FileOutcome(rel, kind, status, reasons)
                else:
                    o = FileOutcome(rel, kind, "duplicate", [f"same content as {first_path}"])
                o.run_id = run_id
                record(o, content_hash)
                return
            try:
                sidecar_text = None if sidecar is None else sidecar.decode("utf-8")
            except UnicodeDecodeError:
                o = FileOutcome(rel, reject_kind or kind, "rejected", ["sidecar is not UTF-8"])
                record(o, content_hash)
                return
            try:
                run = parse(sidecar_text)
            except _Rejected as exc:
                record(FileOutcome(rel, reject_kind or kind, "rejected", [str(exc)]), content_hash)
                return
            status = "ingested" if run.tier == "complete" else "incomplete"
            o = FileOutcome(rel, kind, status, list(run.reasons))
            o.run_id = run_id
            record(o, content_hash)
            runs.append(run)
            known_runs[run_id] = (rel, run.tier, list(run.reasons))

        def legacy(fmt: fm.Format, table: fm.ParsedTable) -> None:
            finish(
                fmt.kind,
                lambda text: _parse_table_run(
                    rel,
                    run_id,
                    fmt,
                    table,
                    text,
                    git_time=git_time,
                    exempt_listed=rel in exempt_list,
                    git_problem=git_problem,
                ),
            )

        if rel.lower().endswith(".csv"):
            try:
                table = fm.parse_csv(data.decode("utf-8-sig"))
            except UnicodeDecodeError:
                record(FileOutcome(rel, "csv", "rejected", ["the CSV is not UTF-8"]), content_hash)
                continue
            except (ValueError, csv.Error) as exc:
                record(FileOutcome(rel, "csv", "rejected", [f"invalid CSV: {exc}"]), content_hash)
                continue
            fmt = fm.match_csv(table.header)
            if fmt is None:
                shown = ",".join(table.header)[:200]
                record(
                    FileOutcome(rel, "csv", "deferred", [f"no format matches the header {shown}"]),
                    content_hash,
                )
            elif not fmt.metrics:
                record(
                    FileOutcome(rel, fmt.kind, "ignored", ["model output: no measured columns"]),
                    content_hash,
                )
            else:
                legacy(fmt, table)
            continue

        try:
            doc = json.loads(data)
        except (ValueError, RecursionError) as exc:
            # ValueError covers JSONDecodeError, UnicodeDecodeError and the
            # integer digit limit; RecursionError a pathologically nested file.
            record(FileOutcome(rel, "json", "rejected", [f"invalid JSON: {exc}"]), content_hash)
            continue
        if not isinstance(doc, dict):
            record(
                FileOutcome(rel, "json", "rejected", ["the JSON is not an object"]), content_hash
            )
            continue
        if "schema" not in doc:
            if "meta" in doc and "probes" not in doc:
                # bench_blas.py --json writes {"meta": ..., "results": ...}.
                try:
                    _check_text(doc)
                    table = _bench_blas_table(doc)
                except _Rejected as exc:
                    record(
                        FileOutcome(rel, fm.BENCH_BLAS.kind, "rejected", [str(exc)]), content_hash
                    )
                    continue
                legacy(fm.BENCH_BLAS, table)
            else:
                record(
                    FileOutcome(rel, "json", "rejected", ["missing field 'schema'"]), content_hash
                )
            continue

        # A rejection's kind is fixed: the schema value may be the invalid text.
        finish(_SCHEMA, lambda text: _parse_run(rel, run_id, doc, text), reject_kind="json")

    # Compare recorded names, which escape a non-UTF-8 name the same way twice.
    seen = {o.path for o in outcomes}
    for path, (content_hash, kind, status, _reasons, _run) in sorted(current.items()):
        if path not in seen and status != "removed":
            o = FileOutcome(path, kind, "removed", ["no longer under the ingest root"])
            file_rows.append(
                {
                    "path": path,
                    "content_hash": content_hash,
                    "kind": kind,
                    "status": "removed",
                    "reasons": o.reasons,
                    "run_id": None,
                }
            )
            outcomes.append(o)

    _write(con, file_rows, runs, root=str(root_path), bind_root=row is None, generation=generation)
    return IngestReport(str(root_path), outcomes)


def _insert_columns(con, table: str, rows: list[dict], *, ignore: bool = False) -> None:
    """Insert rows (dicts with the same keys) in one statement.

    Each column goes in as one typed list parameter and the lists are
    UNNESTed side by side. That keeps a 16,000-row insert to one statement
    without a DataFrame in between: DuckDB 1.1 to 1.4 cannot read pandas 3's
    string columns, and pandas is then needed only to read results out.
    """
    if not rows:
        return
    types = {name: typ for name, typ, *_ in con.execute(f"DESCRIBE {table}").fetchall()}
    cols = list(rows[0])
    select = ", ".join(f"UNNEST(?::{types[c]}[])" for c in cols)
    verb = "INSERT OR IGNORE" if ignore else "INSERT"
    con.execute(
        f"{verb} INTO {table} ({', '.join(cols)}) SELECT {select}",
        [[row[c] for row in rows] for c in cols],
    )


def _generation(con) -> int:
    row = con.execute("SELECT value FROM meta WHERE key = 'generation'").fetchone()
    return int(row[0]) if row else 0


def _write(
    con,
    file_rows: list[dict],
    runs: list[_Run],
    *,
    root: str,
    bind_root: bool,
    generation: int,
) -> None:
    if not file_rows and not bind_root:
        # Nothing to write, but a report of the tree is a claim about the
        # database too: refuse it if another ingest changed the database.
        if _generation(con) != generation:
            raise RuntimeError("the database changed during this ingest; run it again")
        return
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.execute("BEGIN TRANSACTION")
    try:
        # Claim the next generation. A concurrent ingest that committed since
        # this scan read the database makes the claim fail; one committing at
        # the same time conflicts on the same meta row, and DuckDB aborts one.
        if _generation(con) != generation:
            raise RuntimeError("the database changed during this ingest; run it again")
        con.execute("INSERT OR REPLACE INTO meta VALUES ('generation', ?)", [str(generation + 1)])
        if bind_root:
            # Bound with the first successful write, so a failed first ingest
            # leaves the database free to take another root.
            con.execute("INSERT INTO meta VALUES ('root', ?)", [root])
        for m in {r.machine_row["fingerprint"]: r.machine_row for r in runs}.values():
            con.execute(
                "INSERT OR IGNORE INTO machines VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    m["fingerprint"],
                    m["identity"],
                    m["cpu_vendor"],
                    m["cpu_brand"],
                    m["cpu_arch"],
                    m["cpu_logical_cores"],
                    m["host"],
                ],
            )
        run_cols = [c for c in runs[0].run_row if c != "sysinfo"] if runs else []
        for r in runs:
            row = r.run_row
            placeholders = ", ".join("?" for _ in run_cols)
            con.execute(
                f"INSERT INTO runs ({', '.join(run_cols)}, sysinfo) "
                f"VALUES ({placeholders}, MAP(?::VARCHAR[], ?::VARCHAR[]))",
                [row[c] for c in run_cols]
                + [list(row["sysinfo"].keys()), list(row["sysinfo"].values())],
            )
        _insert_columns(con, "probes", [p for r in runs for p in r.probes])
        _insert_columns(con, "results", [x for r in runs for x in r.results])
        _insert_columns(con, "skipped", [s for r in runs for s in r.skipped])
        _insert_columns(con, "measurements", [m for r in runs for m in r.measurements])
        entries = {(e["entry_id"], e["definition_version"]): e for r in runs for e in r.entries}
        _insert_columns(con, "entry_definitions", list(entries.values()), ignore=True)
        for f in file_rows:
            con.execute(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, nextval('ingest_seq'), ?)",
                [
                    f["path"],
                    f["content_hash"],
                    f["kind"],
                    f["status"],
                    f["reasons"],
                    f["run_id"],
                    now,
                ],
            )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
