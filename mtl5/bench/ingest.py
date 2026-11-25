"""Ingest MTL5 benchmark data into the benchmark database.

``ingest(root, db)`` walks ``root`` (normally ``<mtl5>/benchmarks/data``) and
gives every file found there a row in the ``files`` table, so every file is
accounted for: ingested, ingested as incomplete, rejected with a reason, or
set aside with a reason (a report, a figure, a format a later milestone
reads). This milestone reads ``mtl5-bench/1`` JSON, the output of
``mtl5_bench``, with its ``.sysinfo`` sidecar.

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

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    "deferred": "a data format a later milestone ingests (CSV: P2)",
    "ignored": "not benchmark data (a report, figure or manifest)",
    "removed": "a file a previous ingest saw that is no longer there",
}

_SCHEMA = "mtl5-bench/1"
_SIDECAR = ".sysinfo"
_REPORT_SUFFIXES = {".md", ".txt", ".rst", ".log"}
_FIGURE_SUFFIXES = {".png", ".svg", ".pdf", ".jpg", ".jpeg"}
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
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


def connect(db: str | os.PathLike = ":memory:", *, rebuild: bool = False):
    """Open (or create) a benchmark database and return the DuckDB connection.

    ``rebuild=True`` deletes an existing database file first. A database
    written by another schema version raises, since it is a cache and is
    rebuilt rather than migrated.
    """
    duckdb = _duckdb()
    path = str(db)
    if rebuild and path != ":memory:":
        for p in (Path(path), Path(path + ".wal")):
            if p.exists():
                p.unlink()
    con = duckdb.connect(path)
    con.execute(DDL)
    row = con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        con.execute("INSERT INTO meta VALUES ('schema_version', ?)", [str(SCHEMA_VERSION)])
    elif row[0] != str(SCHEMA_VERSION):
        con.close()
        raise RuntimeError(
            f"{path} has benchmark schema version {row[0]}, this mtl5 writes "
            f"{SCHEMA_VERSION}; rebuild it (mtl5-bench ingest --rebuild)"
        )
    return con


# --------------------------------------------------------------------------
# Parsing one mtl5-bench/1 run
# --------------------------------------------------------------------------


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return None


def _int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _bool(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _run_timestamp(rel: str) -> datetime | None:
    parts = rel.split("/")
    if len(parts) < 2:
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
        eid = _entry_id(r)
        if eid in seen:
            raise _Rejected(f"results[{i}] repeats entry {eid!r}")
        seen.add(eid)
    skipped = doc.get("skipped", [])
    if not isinstance(skipped, list) or not all(isinstance(s, dict) for s in skipped):
        raise _Rejected("'skipped' is not a list of objects")


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


def _parse_run(rel: str, run_id: str, doc: Any, sidecar_text: str | None) -> _Run:
    """Validate and flatten one run. Raises _Rejected."""
    _validate_json(doc)
    reasons: list[str] = []

    if sidecar_text is None:
        fields: dict[str, str] = {}
        reasons.append("no .sysinfo sidecar")
    else:
        fields, problems = si.parse_sysinfo(sidecar_text)
        reasons += problems
        missing = [k for k in si.REQUIRED_KEYS if k not in fields]
        if missing:
            reasons.append("sidecar lacks " + ", ".join(missing))
        if "benchmark" in fields and fields["benchmark"] != "mtl5_bench":
            raise _Rejected(f"sidecar says benchmark={fields['benchmark']}, not mtl5_bench")
        for key in ("label", "machine", "class", "simd_backend"):
            if key in fields and fields[key] != doc[key]:
                raise _Rejected(f"sidecar says {key}={fields[key]!r}, the JSON {doc[key]!r}")

    threads = _int(doc["probes"]["threads"])
    pool = _int(fields.get("pool_size"))
    if pool is not None and pool != threads:
        reasons.append(f"sidecar pool_size={pool} but probes.threads={threads}")
    if fields.get("git_dirty", "0") != "0" or fields.get("tree_git_dirty", "0") != "0":
        reasons.append("dirty build: the commit does not identify the code")
    timestamp = _run_timestamp(rel)
    if timestamp is None:
        reasons.append("the run directory name is not a UTC timestamp (YYYYMMDDTHHMMSSZ)")

    tier = "incomplete" if reasons else "complete"

    fingerprint = si.machine_fingerprint(fields)
    backend = si.backend(doc["label"], fields)
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
        "git_dirty": None if "git_dirty" not in fields else fields["git_dirty"] != "0",
        "os_version": fields.get("os_version"),
        "blas_vendor": fields.get("blas_vendor") or None,
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


def _walk(root: Path) -> list[str]:
    """Every file under root, as sorted '/'-separated relative paths.

    Directories whose name starts with '.' (``.git``) and ``__pycache__`` are
    not walked; dot-files are.
    """
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
        for name in sorted(filenames):
            full = Path(dirpath) / name
            if full.is_file():
                out.append(full.relative_to(root).as_posix())
    return sorted(out)


def _classify(rel: str, all_paths: set[str]) -> tuple[str, str, list[str]] | None:
    """(kind, status, reasons) for a file that is not a JSON data file, else None."""
    name = rel.rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()
    if name.endswith(_SIDECAR):
        data = rel[: -len(_SIDECAR)]
        if data in all_paths:
            return "sidecar", "paired", []
        return "sidecar", "rejected", [f"sidecar without its data file {data.rsplit('/', 1)[-1]}"]
    if suffix == ".json":
        return None
    if suffix == ".csv":
        return "csv", "deferred", ["CSV ingest is milestone P2 of #121"]
    if suffix in _REPORT_SUFFIXES:
        return "report", "ignored", []
    if suffix in _FIGURE_SUFFIXES:
        return "figure", "ignored", []
    if name.startswith("."):
        return "manifest", "ignored", []
    return "other", "ignored", ["not a recognised benchmark file"]


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
) -> tuple[Any, IngestReport]:
    """Ingest every file under ``root`` into the database ``db``.

    ``db`` is a path, ``":memory:"``, or a connection from :func:`connect`.
    Returns ``(connection, report)``. One database holds one ingest root: its
    paths are relative to it, and a file missing from the root is recorded as
    removed, so ingesting a different root into the same database raises.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    con = connect(db, rebuild=rebuild) if not hasattr(db, "execute") else db

    row = con.execute("SELECT value FROM meta WHERE key = 'root'").fetchone()
    if row is None:
        con.execute("INSERT INTO meta VALUES ('root', ?)", [str(root_path)])
    elif row[0] != str(root_path):
        raise ValueError(
            f"this database was ingested from {row[0]}, not {root_path}; "
            "use another database or rebuild it"
        )

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

    paths = _walk(root_path)
    path_set = set(paths)
    outcomes: list[FileOutcome] = []
    file_rows: list[dict] = []
    runs: list[_Run] = []

    def record(o: FileOutcome, content_hash: str) -> None:
        prev = current.get(o.path)
        if prev is not None and prev[0] == content_hash and prev[2] != "removed":
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
        classified = _classify(rel, path_set)
        if classified is not None:
            kind, status, reasons = classified
            record(FileOutcome(rel, kind, status, reasons), _sha256(full.read_bytes()))
            continue

        data = full.read_bytes()
        sidecar_path = root_path / (rel + _SIDECAR)
        sidecar = sidecar_path.read_bytes() if (rel + _SIDECAR) in path_set else None
        content_hash = _sha256(data, sidecar if sidecar is not None else b"")
        run_id = content_hash[:32]

        prev = current.get(rel)
        if prev is not None and prev[0] == content_hash and prev[2] != "removed":
            record(FileOutcome(rel, "?", "?"), content_hash)
            continue

        try:
            doc = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            record(FileOutcome(rel, "json", "rejected", [f"invalid JSON: {exc}"]), content_hash)
            continue
        if not isinstance(doc, dict) or "schema" not in doc:
            record(
                FileOutcome(
                    rel, "json", "deferred", ["JSON without a schema field (bench_blas.py: P2)"]
                ),
                content_hash,
            )
            continue

        if run_id in known_runs:
            first_path, tier, reasons = known_runs[run_id]
            if first_path == rel:
                # Removed and restored unchanged: the run is already there.
                status = "ingested" if tier == "complete" else "incomplete"
                o = FileOutcome(rel, _SCHEMA, status, reasons)
            else:
                o = FileOutcome(rel, _SCHEMA, "duplicate", [f"same content as {first_path}"])
            o.run_id = run_id
            record(o, content_hash)
            continue

        try:
            sidecar_text = None if sidecar is None else sidecar.decode("utf-8")
        except UnicodeDecodeError:
            record(
                FileOutcome(rel, str(doc.get("schema")), "rejected", ["sidecar is not UTF-8"]),
                content_hash,
            )
            continue
        try:
            run = _parse_run(rel, run_id, doc, sidecar_text)
        except _Rejected as exc:
            record(FileOutcome(rel, str(doc.get("schema")), "rejected", [str(exc)]), content_hash)
            continue

        status = "ingested" if run.tier == "complete" else "incomplete"
        o = FileOutcome(rel, _SCHEMA, status, list(run.reasons))
        o.run_id = run_id
        record(o, content_hash)
        runs.append(run)
        known_runs[run_id] = (rel, run.tier, list(run.reasons))

    for path, (content_hash, kind, status, _reasons, _run) in sorted(current.items()):
        if path not in path_set and status != "removed":
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

    _write(con, file_rows, runs)
    return con, IngestReport(str(root_path), outcomes)


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


def _write(con, file_rows: list[dict], runs: list[_Run]) -> None:
    if not file_rows:
        return
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.execute("BEGIN TRANSACTION")
    try:
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
