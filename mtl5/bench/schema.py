"""The benchmark database schema.

The database is a cache. The committed data files are the source of truth,
and ``mtl5-bench ingest --rebuild`` recreates the database from them.

Tables
------
``files``
    One row per file seen under the ingest root, for every file, so the
    ingest accounts for all of them. A file is identified by its path and the
    SHA-256 of its content (plus its sidecar's, for a data file). Rows are
    only ever appended: a file that changes on disk is a new row, and the
    ``current_files`` view picks the newest row per path.
``machines``
    One row per physical machine, keyed by its fingerprint.
``runs``
    One row per distinct data file content (a run of one benchmark binary);
    the same content committed at a second path is the same run. Each has its
    provenance tier and the parts of the comparability key that belong to the
    run.
``probes``
    The run's machine probes (FMA peak, bandwidths), in long form, so a probe
    added upstream needs no schema change.
``results``
    One row per benchmark entry per run, with its series key.
``skipped``
    The entries a run skipped, and why.
``entry_definitions``
    Each entry and its definition version. MTL5's JSON carries no version yet,
    so every entry is version 0 (unversioned) until it does.

Views
-----
``current_runs``
    The runs that some current file row still points at.
``trend_results``
    The results trend queries may use: current runs in the ``complete`` tier
    whose correctness check passed.
"""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION", "DDL"]

#: Bumped when a table changes shape. A database with another version is
#: rebuilt rather than migrated: it is a cache of the files.
SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR
);

CREATE SEQUENCE IF NOT EXISTS ingest_seq START 1;

CREATE TABLE IF NOT EXISTS files (
    path          VARCHAR NOT NULL,   -- relative to the ingest root, '/'-separated
    content_hash  VARCHAR NOT NULL,   -- sha256 of the file (and its sidecar)
    kind          VARCHAR NOT NULL,   -- mtl5-bench/1, sidecar, csv, report, ...
    status        VARCHAR NOT NULL,   -- see INGEST_STATUSES in ingest.py
    reasons       VARCHAR[],
    run_id        VARCHAR,            -- the run this data file is (ingested, incomplete, duplicate)
    ingest_seq    BIGINT NOT NULL,
    ingested_at   TIMESTAMP NOT NULL  -- UTC
);

CREATE TABLE IF NOT EXISTS machines (
    fingerprint        VARCHAR PRIMARY KEY,
    identity           VARCHAR NOT NULL,
    cpu_vendor         VARCHAR,
    cpu_brand          VARCHAR,
    cpu_arch           VARCHAR,
    cpu_logical_cores  INTEGER,
    host               VARCHAR
);

CREATE TABLE IF NOT EXISTS runs (
    run_id               VARCHAR PRIMARY KEY,  -- the content hash of the data file and sidecar
    path                 VARCHAR NOT NULL,
    schema               VARCHAR NOT NULL,
    tier                 VARCHAR NOT NULL,     -- complete or incomplete
    reasons              VARCHAR[],
    run_timestamp        TIMESTAMP,            -- UTC, from the run directory name
    machine              VARCHAR,              -- the label in the JSON
    machine_dir          VARCHAR,              -- the data directory it was committed under
    machine_fingerprint  VARCHAR,
    label                VARCHAR,
    class                VARCHAR,
    threads              INTEGER,
    backend              VARCHAR,
    build_config         VARCHAR,
    state_class          VARCHAR,
    config_key           VARCHAR,              -- hash of everything but the entry and the commit
    git_commit           VARCHAR,
    git_dirty            BOOLEAN,
    os_version           VARCHAR,
    blas_vendor          VARCHAR,
    thermal_before_c     DOUBLE,
    thermal_after_c      DOUBLE,
    loadavg_1m           DOUBLE,
    sysinfo              MAP(VARCHAR, VARCHAR)
);

CREATE TABLE IF NOT EXISTS probes (
    run_id  VARCHAR NOT NULL,
    name    VARCHAR NOT NULL,
    value   DOUBLE,
    text    VARCHAR,
    PRIMARY KEY (run_id, name)
);

CREATE TABLE IF NOT EXISTS entry_definitions (
    entry_id            VARCHAR NOT NULL,
    definition_version  INTEGER NOT NULL,
    area                VARCHAR,
    op                  VARCHAR,
    dtype               VARCHAR,
    shape               VARCHAR,
    PRIMARY KEY (entry_id, definition_version)
);

CREATE TABLE IF NOT EXISTS results (
    run_id              VARCHAR NOT NULL,
    entry_id            VARCHAR NOT NULL,   -- area/op/dtype/shape
    definition_version  INTEGER NOT NULL,
    series_key          VARCHAR NOT NULL,
    area                VARCHAR,
    op                  VARCHAR,
    dtype               VARCHAR,
    shape               VARCHAR,
    median_s            DOUBLE,
    min_s               DOUBLE,
    samples             INTEGER,
    flops               DOUBLE,
    bytes               DOUBLE,
    gflops              DOUBLE,
    gbytes_per_s        DOUBLE,
    roofline_gflops     DOUBLE,
    roofline_pct        DOUBLE,
    cache_resident      BOOLEAN,
    suspect_model       BOOLEAN,
    speedup             DOUBLE,
    efficiency          DOUBLE,
    check_ok            BOOLEAN,
    check_error         DOUBLE,
    check_tolerance     DOUBLE,
    check_metric        VARCHAR,
    PRIMARY KEY (run_id, entry_id)
);

CREATE TABLE IF NOT EXISTS skipped (
    run_id  VARCHAR NOT NULL,
    area    VARCHAR,
    op      VARCHAR,
    dtype   VARCHAR,
    reason  VARCHAR
);

CREATE OR REPLACE VIEW current_files AS
    SELECT * FROM files
    QUALIFY row_number() OVER (PARTITION BY path ORDER BY ingest_seq DESC) = 1;

CREATE OR REPLACE VIEW current_runs AS
    SELECT * FROM runs
    WHERE run_id IN (SELECT run_id FROM current_files WHERE run_id IS NOT NULL);

CREATE OR REPLACE VIEW trend_results AS
    SELECT res.*, r.run_timestamp, r.git_commit, r.machine_fingerprint,
           r.machine, r.label, r.threads, r.backend, r.build_config,
           r.state_class, r.class
    FROM results res
    JOIN current_runs r USING (run_id)
    WHERE r.tier = 'complete' AND res.check_ok IS NOT FALSE;
"""
