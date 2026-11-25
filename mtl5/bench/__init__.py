"""Benchmark data: ingestion, provenance and (in later milestones) analysis.

This subpackage reads the benchmark results MTL5 commits under
``benchmarks/data`` into a DuckDB database, with the provenance that decides
whether two measurements are comparable (#121). It needs the ``bench`` extra::

    pip install 'mtl5[bench]'
    mtl5-bench ingest <mtl5 checkout>/benchmarks/data --db bench.duckdb

or from Python::

    from mtl5 import bench
    con, report = bench.ingest("<mtl5>/benchmarks/data", "bench.duckdb")
    print(report)
    df = con.sql("SELECT * FROM trend_results").df()

A **series** is one measurement repeated over time. Every result row carries a
``series_key``, the hash of its comparability key: the physical machine, its
state class, the backend, the build configuration without the commit, the
thread count, the problem class, the entry and the entry's definition version.
Two rows with the same series key are the same measurement at different
commits or times; a change in any key field starts a new series. The readable
parts of the key are stored on each run (``machine_fingerprint``,
``state_class``, ``backend``, ``build_config``, ``threads``), so a series break
can say which one changed.

Legacy formats (MTL5's CSVs, ``bench_blas.py`` JSON) go to the long
``measurements`` table and the ``trend_measurements`` view instead, one row
per file row and metric; :mod:`mtl5.bench.formats` describes each dialect.
A legacy file outside a timestamp directory takes its run time from the git
commit that added it.

The submodules ``sysinfo``, ``schema``, ``formats`` and ``ingest`` hold the
details. Only
the standard library is imported here; DuckDB and pandas are imported when a
function needs them.
"""

from .ingest import INGEST_STATUSES, FileOutcome, IngestReport, connect, ingest
from .sysinfo import REQUIRED_KEYS

__all__ = [
    "INGEST_STATUSES",
    "REQUIRED_KEYS",
    "FileOutcome",
    "IngestReport",
    "connect",
    "ingest",
]
