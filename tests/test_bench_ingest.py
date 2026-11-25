"""Benchmark ingestion and provenance (mtl5.bench, #121 milestone P1).

The fixtures under tests/data/bench are three unmodified mtl5_bench runs
copied from stillwater-sc/mtl5 benchmarks/data: two native-t1 runs on the
i7-12700K a day apart (one series), and one on the Jetson Orin. The
negative cases edit copies of them in a temporary directory.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from mtl5 import bench  # noqa: E402
from mtl5.bench import sysinfo as si  # noqa: E402
from mtl5.bench.cli import main as cli_main  # noqa: E402

FIXTURES = Path(__file__).parent / "data" / "bench"
I7_A = "i7-12700k/mtl5_bench/20261111T190041Z/native-t1.json"
I7_B = "i7-12700k/mtl5_bench/20261112T004852Z/native-t1.json"
JETSON = "jetson-orin/mtl5_bench/20261113T131507Z/native-t1.json"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(FIXTURES, root)
    return root


def _load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text())


def _dump(root: Path, rel: str, doc: dict) -> None:
    (root / rel).write_text(json.dumps(doc))


def _edit_sidecar(root: Path, rel: str, **changes: str | None) -> None:
    """Set (or, with None, delete) sidecar keys."""
    path = root / (rel + ".sysinfo")
    fields, _ = si.parse_sysinfo(path.read_text())
    for k, v in changes.items():
        if v is None:
            fields.pop(k, None)
        else:
            fields[k] = v
    path.write_text("".join(f"{k}={v}\n" for k, v in fields.items()))


def _entries(root: Path, rel: str) -> set[tuple[str, ...]]:
    return {
        tuple(r[k] for k in ("area", "op", "dtype", "shape")) for r in _load(root, rel)["results"]
    }


def _status(report: bench.IngestReport, rel: str) -> bench.FileOutcome:
    (o,) = [o for o in report.outcomes if o.path == rel]
    return o


def _scalar(con, sql: str, params=None):
    return con.execute(sql, params or []).fetchone()[0]


# --------------------------------------------------------------------------
# The committed fixtures
# --------------------------------------------------------------------------


class TestFixtures:
    def test_every_file_is_accounted_for(self, tree):
        con, report = bench.ingest(tree)
        on_disk = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file())
        assert sorted(o.path for o in report.outcomes) == on_disk
        assert report.counts() == {"ingested": 3, "paired": 3}
        assert _scalar(con, "SELECT count(*) FROM files") == 6

    def test_results_match_the_json(self, tree):
        con, _ = bench.ingest(tree)
        for rel in (I7_A, I7_B, JETSON):
            doc = _load(tree, rel)
            run_id = _scalar(con, "SELECT run_id FROM runs WHERE path = ?", [rel])
            assert _scalar(con, "SELECT count(*) FROM results WHERE run_id = ?", [run_id]) == len(
                doc["results"]
            )
            assert _scalar(con, "SELECT count(*) FROM probes WHERE run_id = ?", [run_id]) == len(
                doc["probes"]
            )
            first = doc["results"][0]
            gflops = _scalar(
                con,
                "SELECT gflops FROM results WHERE run_id = ? AND entry_id = ?",
                [run_id, "/".join(first[k] for k in ("area", "op", "dtype", "shape"))],
            )
            assert gflops == pytest.approx(first["gflops"])

    def test_one_machine_one_day_apart_is_one_series(self, tree):
        con, _ = bench.ingest(tree)
        keys = dict(con.execute("SELECT path, config_key FROM runs").fetchall())
        assert keys[I7_A] == keys[I7_B]
        assert keys[JETSON] != keys[I7_A]
        # Every entry the two i7 runs share is one series with two points.
        n = _scalar(
            con,
            """SELECT count(*) FROM (
                 SELECT series_key FROM trend_results WHERE machine = 'i7-12700k'
                 GROUP BY series_key HAVING count(*) = 2)""",
        )
        shared = _entries(tree, I7_A) & _entries(tree, I7_B)
        assert n == len(shared) > 0

    def test_machines_are_fingerprinted_by_hardware(self, tree):
        con, _ = bench.ingest(tree)
        rows = con.execute("SELECT cpu_brand, host FROM machines ORDER BY host").fetchall()
        assert rows == [
            ("ARMv8 Processor rev 1 (v8l)", "branes-jetson"),
            ("12th Gen Intel(R) Core(TM) i7-12700K", "sw-21"),
        ]

    def test_run_timestamp_comes_from_the_directory(self, tree):
        con, _ = bench.ingest(tree)
        ts = _scalar(con, "SELECT run_timestamp FROM runs WHERE path = ?", [I7_A])
        assert ts.isoformat() == "2026-11-11T19:00:41"

    def test_reingest_appends_nothing(self, tree):
        con, first = bench.ingest(tree)
        before = [
            _scalar(con, f"SELECT count(*) FROM {t}")
            for t in ("files", "runs", "results", "probes")
        ]
        con, second = bench.ingest(tree, con)
        after = [
            _scalar(con, f"SELECT count(*) FROM {t}")
            for t in ("files", "runs", "results", "probes")
        ]
        assert second.new_rows == 0 and first.new_rows == 6
        assert after == before
        assert second.counts() == first.counts()

    def test_database_file_round_trip_and_rebuild(self, tree, tmp_path):
        db = tmp_path / "bench.duckdb"
        con, _ = bench.ingest(tree, db)
        con.close()
        con, report = bench.ingest(tree, db)
        assert report.new_rows == 0
        con.close()
        con, report = bench.ingest(tree, db, rebuild=True)
        assert report.new_rows == 6
        con.close()


# --------------------------------------------------------------------------
# Provenance tiers
# --------------------------------------------------------------------------


def _trend_paths(con) -> set[str]:
    return {
        p
        for (p,) in con.execute(
            "SELECT DISTINCT r.path FROM trend_results t JOIN runs r USING (run_id)"
        ).fetchall()
    }


class TestIncomplete:
    def test_no_sidecar(self, tree):
        (tree / (I7_A + ".sysinfo")).unlink()
        con, report = bench.ingest(tree)
        o = _status(report, I7_A)
        assert o.status == "incomplete" and o.reasons == ["no .sysinfo sidecar"]
        assert I7_A not in _trend_paths(con)
        assert _scalar(con, "SELECT tier FROM runs WHERE path = ?", [I7_A]) == "incomplete"

    def test_sidecar_missing_a_required_key(self, tree):
        _edit_sidecar(tree, I7_A, compiler_version=None, governor=None)
        _, report = bench.ingest(tree)
        o = _status(report, I7_A)
        assert o.status == "incomplete"
        assert o.reasons == ["sidecar lacks compiler_version, governor"]

    def test_dirty_build(self, tree):
        _edit_sidecar(tree, I7_A, git_dirty="1")
        con, report = bench.ingest(tree)
        assert _status(report, I7_A).reasons == [
            "dirty build: the commit does not identify the code"
        ]
        assert _scalar(con, "SELECT git_dirty FROM runs WHERE path = ?", [I7_A]) is True

    def test_conflicting_duplicate_sidecar_key(self, tree):
        path = tree / (I7_A + ".sysinfo")
        path.write_text(path.read_text() + "turbo=disabled\n")
        _, report = bench.ingest(tree)
        # The conflicted key is dropped, so it is also missing.
        assert _status(report, I7_A).reasons == [
            "sidecar key 'turbo' has conflicting values",
            "sidecar lacks turbo",
        ]

    def test_conflicting_identity_key_is_order_independent(self, tree):
        path = tree / (I7_A + ".sysinfo")
        text = path.read_text()
        verdicts = []
        for extra in ("machine=other\n", None):
            if extra is None:
                # The contradicting value first, the matching one last.
                body = "machine=other\n" + text
            else:
                body = text + extra
            path.write_text(body)
            _, report = bench.ingest(tree)
            o = _status(report, I7_A)
            verdicts.append((o.status, o.reasons))
        assert verdicts[0] == verdicts[1]
        assert verdicts[0][0] == "incomplete"

    def test_sidecar_without_machine_state(self, tree):
        # Older sidecars (18 of MTL5's 134 runs) predate these keys.
        _edit_sidecar(tree, I7_A, clock_policy=None, bench_cpus=None)
        con, report = bench.ingest(tree)
        assert _status(report, I7_A).reasons == ["sidecar lacks clock_policy, bench_cpus"]
        assert I7_A not in _trend_paths(con)

    def test_empty_commit_and_host(self, tree):
        _edit_sidecar(tree, I7_A, git_commit="", preflight_host=" ", cxx_flags="")
        _, report = bench.ingest(tree)
        assert _status(report, I7_A).reasons == ["sidecar has empty preflight_host, git_commit"]

    def test_thread_count_disagreement(self, tree):
        _edit_sidecar(tree, I7_A, pool_size="4")
        _, report = bench.ingest(tree)
        assert _status(report, I7_A).reasons == ["sidecar pool_size=4 but probes.threads=1"]

    def test_pool_size_not_an_integer(self, tree):
        _edit_sidecar(tree, I7_A, pool_size="unknown")
        _, report = bench.ingest(tree)
        assert _status(report, I7_A).reasons == ["sidecar pool_size='unknown' is not an integer"]

    def test_timestamp_directory_with_a_digit_missing(self, tree):
        dest = tree / "i7-12700k" / "mtl5_bench" / "20261112T00485Z"
        (tree / I7_B).parent.rename(dest)
        _, report = bench.ingest(tree)
        o = _status(report, "i7-12700k/mtl5_bench/20261112T00485Z/native-t1.json")
        assert o.status == "incomplete" and "not a UTC timestamp" in o.reasons[0]

    def test_run_directory_without_a_timestamp(self, tree):
        dest = tree / "i7-12700k" / "mtl5_bench" / "latest"
        dest.mkdir()
        for suffix in ("", ".sysinfo"):
            shutil.move(tree / (I7_B + suffix), dest / ("native-t1.json" + suffix))
        con, report = bench.ingest(tree)
        o = _status(report, "i7-12700k/mtl5_bench/latest/native-t1.json")
        assert o.status == "incomplete"
        assert "not a UTC timestamp" in o.reasons[0]


class TestRejected:
    def _rejected(self, tree, rel=I7_A):
        con, report = bench.ingest(tree)
        o = _status(report, rel)
        assert o.status == "rejected"
        assert _scalar(con, "SELECT count(*) FROM runs WHERE path = ?", [rel]) == 0
        return o.reasons[0]

    def test_invalid_json(self, tree):
        (tree / I7_A).write_text("{not json")
        assert self._rejected(tree).startswith("invalid JSON")

    def test_other_schema(self, tree):
        doc = _load(tree, I7_A)
        doc["schema"] = "mtl5-bench/2"
        _dump(tree, I7_A, doc)
        assert self._rejected(tree) == "schema is 'mtl5-bench/2', not 'mtl5-bench/1'"

    def test_sidecar_contradicts_json(self, tree):
        _edit_sidecar(tree, I7_A, machine="ryzen-7-2700x")
        assert self._rejected(tree) == "sidecar says machine='ryzen-7-2700x', the JSON 'i7-12700k'"

    def test_sidecar_of_another_benchmark(self, tree):
        _edit_sidecar(tree, I7_A, benchmark="bench_all")
        assert "not mtl5_bench" in self._rejected(tree)

    def test_missing_result_field(self, tree):
        doc = _load(tree, I7_A)
        del doc["results"][3]["median_s"]
        _dump(tree, I7_A, doc)
        assert self._rejected(tree) == "results[3] has no finite median_s"

    def test_repeated_entry(self, tree):
        doc = _load(tree, I7_A)
        doc["results"].append(dict(doc["results"][0]))
        _dump(tree, I7_A, doc)
        assert "repeats entry" in self._rejected(tree)

    def test_probe_threads_missing(self, tree):
        doc = _load(tree, I7_A)
        del doc["probes"]["threads"]
        _dump(tree, I7_A, doc)
        assert self._rejected(tree) == "probes.threads is missing or not an integer"

    def test_huge_integer_median(self, tree):
        # 400 digits overflows a double; 5000 also exceeds Python 3.11+'s
        # integer-string limit inside json.loads. Either way, one rejection.
        for digits in (400, 5000):
            text = (tree / I7_A).read_text()
            first = _load(tree, I7_A)["results"][0]["median_s"]
            (tree / I7_A).write_text(
                text.replace(f'"median_s": {first}', '"median_s": 1' + "0" * digits, 1)
            )
            reason = self._rejected(tree)
            assert reason == "results[0] has no finite median_s" or reason.startswith(
                "invalid JSON"
            )
            shutil.copy(FIXTURES / I7_A, tree / I7_A)

    def test_out_of_range_optional_integer_is_dropped(self, tree):
        doc = _load(tree, I7_A)
        doc["results"][0]["samples"] = 2**70
        _dump(tree, I7_A, doc)
        con, report = bench.ingest(tree)
        assert _status(report, I7_A).status == "ingested"
        assert _scalar(con, "SELECT count(*) FROM results WHERE samples IS NULL") == 1

    def test_non_object_json(self, tree):
        (tree / I7_A).write_text("[]")
        assert self._rejected(tree) == "the JSON is not an object"

    def test_lone_surrogate_in_the_schema(self, tree):
        text = (tree / I7_A).read_text().replace('"mtl5-bench/1"', '"\\ud800"', 1)
        (tree / I7_A).write_text(text)
        con, report = bench.ingest(tree)
        o = _status(report, I7_A)
        assert (o.kind, o.status) == ("json", "rejected")
        assert _scalar(con, "SELECT count(*) FROM runs") == 2

    def test_lone_surrogate(self, tree):
        (tree / (I7_A + ".sysinfo")).unlink()
        doc = _load(tree, I7_A)
        text = json.dumps(doc).replace('"class": "medium"', '"class": "\\ud800"', 1)
        (tree / I7_A).write_text(text)
        assert self._rejected(tree) == "the JSON contains text that is not valid Unicode"

    def test_skipped_reason_not_a_string(self, tree):
        doc = _load(tree, I7_A)
        doc["skipped"] = [{"area": "direct", "op": "x", "dtype": "f64", "reason": 3}]
        _dump(tree, I7_A, doc)
        assert self._rejected(tree) == "skipped[0].reason is not a string"

    def test_orphan_sidecar(self, tree):
        (tree / I7_A).unlink()
        _, report = bench.ingest(tree)
        o = _status(report, I7_A + ".sysinfo")
        assert o.status == "rejected" and "without its data file" in o.reasons[0]


class TestOtherFiles:
    def test_csv_report_figure_and_manifest(self, tree):
        (tree / "blas_sweep_mkl.csv").write_text("n,gflops\n")
        (tree / "blas_sweep_mkl.csv.sysinfo").write_text("machine=x\n")
        (tree / "i7-12700k" / "mtl5_bench" / "20261111T190041Z" / "summary.md").write_text("# s\n")
        (tree / "gemm.png").write_bytes(b"\x89PNG")
        (tree / ".sidecar-exempt").write_text("# none\n")
        (tree / "bench_blas.json").write_text('{"results": []}')
        _, report = bench.ingest(tree)
        got = {o.path: (o.kind, o.status) for o in report.outcomes}
        assert got["blas_sweep_mkl.csv"] == ("csv", "deferred")
        assert got["blas_sweep_mkl.csv.sysinfo"] == ("sidecar", "paired")
        assert got["i7-12700k/mtl5_bench/20261111T190041Z/summary.md"] == ("report", "ignored")
        assert got["gemm.png"] == ("figure", "ignored")
        assert got[".sidecar-exempt"] == ("manifest", "ignored")
        assert got["bench_blas.json"] == ("json", "deferred")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX byte file names")
    def test_a_non_utf8_file_name_is_accounted_for(self, tree):
        name = os.fsdecode(b"run-\xff.json")
        try:
            (tree / name).write_text("{}")
        except (OSError, UnicodeEncodeError):
            pytest.skip("this file system refuses non-UTF-8 names")
        con, report = bench.ingest(tree)
        (o,) = [o for o in report.outcomes if o.path.startswith("run-")]
        assert (o.path, o.status) == ("run-\\xff.json", "rejected")
        assert report.counts()["ingested"] == 3
        con, report = bench.ingest(tree, con)
        assert report.new_rows == 0

    def test_hidden_directories_are_not_walked(self, tree):
        (tree / ".git").mkdir()
        (tree / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        _, report = bench.ingest(tree)
        assert not any(o.path.startswith(".git/") for o in report.outcomes)


# --------------------------------------------------------------------------
# Append-only history
# --------------------------------------------------------------------------


class TestHistory:
    def test_changed_file_supersedes_its_run(self, tree):
        con, _ = bench.ingest(tree)
        old = _scalar(con, "SELECT run_id FROM runs WHERE path = ?", [I7_A])
        doc = _load(tree, I7_A)
        doc["results"][0]["gflops"] = 1.0
        _dump(tree, I7_A, doc)
        con, report = bench.ingest(tree, con)
        assert report.new_rows == 1
        assert _scalar(con, "SELECT count(*) FROM files WHERE path = ?", [I7_A]) == 2
        current = {r for (r,) in con.execute("SELECT run_id FROM current_runs").fetchall()}
        assert old not in current
        assert _scalar(con, "SELECT count(*) FROM runs WHERE path = ?", [I7_A]) == 2

    def test_removed_and_restored(self, tree, tmp_path):
        con, _ = bench.ingest(tree)
        keep = tmp_path / "keep"
        keep.mkdir()
        for suffix in ("", ".sysinfo"):
            shutil.move(tree / (I7_A + suffix), keep / ("x" + suffix))
        con, report = bench.ingest(tree, con)
        assert _status(report, I7_A).status == "removed"
        assert _scalar(con, "SELECT count(*) FROM current_runs") == 2
        for suffix in ("", ".sysinfo"):
            shutil.move(keep / ("x" + suffix), tree / (I7_A + suffix))
        con, report = bench.ingest(tree, con)
        assert _status(report, I7_A).status == "ingested"
        assert _scalar(con, "SELECT count(*) FROM current_runs") == 3
        assert _scalar(con, "SELECT count(*) FROM runs") == 3

    def test_same_content_at_two_paths_is_one_run(self, tree):
        # Same machine directory and timestamp directory, so the same
        # provenance; z-copy sorts after mtl5_bench, so the copy is seen second.
        copy = "i7-12700k/z-copy/20261111T190041Z/native-t1.json"
        (tree / copy).parent.mkdir(parents=True)
        for suffix in ("", ".sysinfo"):
            shutil.copy(tree / (I7_A + suffix), tree / (copy + suffix))
        con, report = bench.ingest(tree)
        o = _status(report, copy)
        assert o.status == "duplicate" and o.reasons == [f"same content as {I7_A}"]
        assert _scalar(con, "SELECT count(*) FROM runs") == 3

    def test_moving_a_run_out_of_its_timestamp_directory(self, tree):
        con, _ = bench.ingest(tree)
        moved = "i7-12700k/mtl5_bench/scratch/native-t1.json"
        (tree / moved).parent.mkdir()
        for suffix in ("", ".sysinfo"):
            shutil.move(tree / (I7_A + suffix), tree / (moved + suffix))
        con, report = bench.ingest(tree, con)
        assert _status(report, I7_A).status == "removed"
        o = _status(report, moved)
        assert o.status == "incomplete" and "not a UTC timestamp" in o.reasons[0]
        assert _trend_paths(con) == {I7_B, JETSON}

    def test_sidecar_status_follows_its_data_file(self, tree, tmp_path):
        con, _ = bench.ingest(tree)
        side = I7_A + ".sysinfo"
        shutil.move(tree / I7_A, tmp_path / "x.json")
        con, report = bench.ingest(tree, con)
        assert _status(report, side).status == "rejected"
        shutil.move(tmp_path / "x.json", tree / I7_A)
        con, report = bench.ingest(tree, con)
        assert _status(report, side).status == "paired"
        assert _status(report, side).new

    def test_adding_an_empty_sidecar_revalidates(self, tree):
        side = tree / (I7_A + ".sysinfo")
        side.unlink()
        con, report = bench.ingest(tree)
        assert _status(report, I7_A).reasons == ["no .sysinfo sidecar"]
        side.write_text("")
        con, report = bench.ingest(tree, con)
        o = _status(report, I7_A)
        assert o.new and o.reasons[0].startswith("sidecar lacks benchmark, label")

    def test_database_under_the_root_is_not_an_input(self, tree):
        con, report = bench.ingest(tree, tree / "bench.duckdb")
        assert not any(o.path.startswith("bench.duckdb") for o in report.outcomes)
        con, report = bench.ingest(tree, con)
        assert report.new_rows == 0
        con.close()

    def test_rebuild_refuses_to_delete_a_non_database(self, tree):
        target = tree / I7_A
        with pytest.raises(ValueError, match="not an mtl5 benchmark database"):
            bench.ingest(tree, target, rebuild=True)
        assert target.read_bytes() == (FIXTURES / I7_A).read_bytes()

    def test_rebuild_refuses_to_delete_another_duckdb_database(self, tree, tmp_path):
        other = tmp_path / "other.duckdb"
        c = duckdb.connect(str(other))
        c.execute("CREATE TABLE important (x INTEGER)")
        c.close()
        with pytest.raises(ValueError, match="not an mtl5 benchmark database"):
            bench.connect(other, rebuild=True)
        c = duckdb.connect(str(other))
        assert c.execute("SELECT count(*) FROM important").fetchone() == (0,)
        c.close()

    def test_rebuild_refuses_a_lookalike_meta_table(self, tmp_path):
        other = tmp_path / "other.duckdb"
        c = duckdb.connect(str(other))
        c.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
        c.execute("INSERT INTO meta VALUES ('schema_version', '1')")
        c.close()
        with pytest.raises(ValueError, match="not an mtl5 benchmark database"):
            bench.connect(other, rebuild=True)
        assert other.exists()

    def test_another_applications_database_is_not_opened_for_writing(self, tmp_path):
        other = tmp_path / "other.duckdb"
        c = duckdb.connect(str(other))
        c.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
        c.close()
        with pytest.raises(RuntimeError, match="did not create"):
            bench.connect(other)
        # The failed connect closed its connection: a read-only open works.
        duckdb.connect(str(other), read_only=True).close()

    def test_old_schema_reports_the_version_not_a_binder_error(self, tmp_path):
        db = tmp_path / "old.duckdb"
        c = duckdb.connect(str(db))
        c.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        c.execute("INSERT INTO meta VALUES ('application', 'mtl5.bench'), ('schema_version', '0')")
        # A table without the columns the current views select.
        c.execute("CREATE TABLE runs (run_id VARCHAR)")
        c.close()
        with pytest.raises(RuntimeError, match="schema version 0"):
            bench.connect(db)
        bench.connect(db, rebuild=True).close()

    def test_rebuild_leaves_a_stray_wal_alone(self, tmp_path):
        wal = tmp_path / "b.duckdb.wal"
        wal.write_bytes(b"not ours")
        with pytest.raises(ValueError, match="no database beside it"):
            bench.connect(tmp_path / "b.duckdb", rebuild=True)
        assert wal.read_bytes() == b"not ours"

    @pytest.mark.skipif(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="needs POSIX permissions that bind the caller",
    )
    def test_a_failed_first_ingest_does_not_bind_the_root(self, tree, tmp_path):
        con = bench.connect()
        locked = tree / "jetson-orin"
        locked.chmod(0)
        try:
            with pytest.raises(PermissionError):
                bench.ingest(tree, con)
        finally:
            locked.chmod(0o755)
        other = tmp_path / "other"
        shutil.copytree(FIXTURES / "i7-12700k", other / "i7-12700k")
        _, report = bench.ingest(other, con)
        assert report.counts() == {"ingested": 2, "paired": 2}

    def test_an_empty_root_binds_on_success(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        con, report = bench.ingest(empty)
        assert report.outcomes == []
        with pytest.raises(ValueError, match="ingested from"):
            bench.ingest(tmp_path, con)

    def test_concurrent_ingest_is_refused(self, tree, tmp_path, monkeypatch):
        ing = sys.modules["mtl5.bench.ingest"]
        db = tmp_path / "b.duckdb"
        con = bench.connect(db)
        real_walk = ing._walk
        nested = []

        def walk_and_race(root, exclude):
            if not nested:
                nested.append(True)
                # Another ingest on the same database commits mid-scan.
                bench.ingest(tree, con.cursor())
            return real_walk(root, exclude)

        monkeypatch.setattr(ing, "_walk", walk_and_race)
        with pytest.raises(RuntimeError, match="changed during this ingest"):
            bench.ingest(tree, con)
        monkeypatch.setattr(ing, "_walk", real_walk)
        # The nested ingest's rows are intact, and a rerun appends nothing.
        _, report = bench.ingest(tree, con)
        assert report.new_rows == 0
        con.close()

    def test_rebuild_needs_a_path(self, tree):
        con, _ = bench.ingest(tree)
        with pytest.raises(ValueError, match="database path"):
            bench.ingest(tree, con, rebuild=True)

    def test_an_owned_connection_is_closed_on_failure(self, tree, tmp_path, monkeypatch):
        # The package re-exports ingest(), which shadows the submodule name.
        ing = sys.modules["mtl5.bench.ingest"]
        closed = []
        real = ing.connect

        def tracking(*a, **k):
            con = real(*a, **k)
            real_close = con.close

            class Wrapper:
                def __getattr__(self, name):
                    return getattr(con, name)

                def close(self):
                    closed.append(True)
                    real_close()

            return Wrapper()

        db = tmp_path / "b.duckdb"
        bench.ingest(tree, db)[0].close()
        monkeypatch.setattr(ing, "connect", tracking)
        with pytest.raises(ValueError, match="ingested from"):
            ing.ingest(tmp_path, db)
        assert closed == [True]

    @pytest.mark.skipif(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="needs POSIX permissions that bind the caller",
    )
    def test_unreadable_directory_aborts_before_recording_removals(self, tree):
        con, _ = bench.ingest(tree)
        locked = tree / "jetson-orin"
        locked.chmod(0)
        try:
            with pytest.raises(PermissionError):
                bench.ingest(tree, con)
        finally:
            locked.chmod(0o755)
        assert _scalar(con, "SELECT count(*) FROM files WHERE status = 'removed'") == 0

    def test_one_database_one_root(self, tree, tmp_path):
        con, _ = bench.ingest(tree)
        other = tmp_path / "other"
        other.mkdir()
        with pytest.raises(ValueError, match="ingested from"):
            bench.ingest(other, con)

    def test_schema_version_mismatch(self, tree, tmp_path):
        db = tmp_path / "old.duckdb"
        con = bench.connect(db)
        con.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
        con.close()
        with pytest.raises(RuntimeError, match="rebuild"):
            bench.connect(db)
        bench.connect(db, rebuild=True).close()


# --------------------------------------------------------------------------
# The comparability key
# --------------------------------------------------------------------------


def _config_keys(tree) -> dict[str, str]:
    con, _ = bench.ingest(tree)
    return dict(con.execute("SELECT path, config_key FROM runs").fetchall())


class TestComparabilityKey:
    def test_commit_is_the_trend_axis_not_the_key(self, tree):
        _edit_sidecar(tree, I7_B, git_commit="0123456789ab")
        keys = _config_keys(tree)
        assert keys[I7_A] == keys[I7_B]

    @pytest.mark.parametrize(
        "change",
        [
            {"compiler_version": "15.3.0"},  # build
            {"cxx_flags": "-O2"},  # build
            {"governor": "powersave"},  # machine state
            {"turbo": "disabled"},  # machine state
            {"blas_vendor": "OpenBLAS"},  # backend
            {"preflight_host": "sw-22"},  # another box with the same CPU
            {"clock_policy": "nvpmodel 15W (mode 0)"},  # machine state
        ],
    )
    def test_a_key_field_starts_a_new_series(self, tree, change):
        _edit_sidecar(tree, I7_B, **change)
        keys = _config_keys(tree)
        assert keys[I7_A] != keys[I7_B]

    def test_thermals_and_load_are_measurements_not_key(self, tree):
        _edit_sidecar(tree, I7_B, thermal_before_c="60.0", loadavg_1m="3.5")
        keys = _config_keys(tree)
        assert keys[I7_A] == keys[I7_B]

    def test_outer_flag_whitespace_is_not_a_new_build(self, tree):
        _edit_sidecar(tree, I7_B, cxx_flags="  -O3 -DNDEBUG   ")
        keys = _config_keys(tree)
        assert keys[I7_A] == keys[I7_B]

    def test_inner_flag_whitespace_is_a_new_build(self):
        # -DMSG='"a  b"' and -DMSG='"a b"' define different strings.
        a = si.build_config({"cxx_flags": "-DMSG='\"a  b\"'"})
        b = si.build_config({"cxx_flags": "-DMSG='\"a b\"'"})
        assert a != b

    def test_host_domain_is_not_a_new_machine(self):
        a = {"cpu_brand": "Xeon", "preflight_host": "sw-13"}
        b = {"cpu_brand": "Xeon", "preflight_host": "SW-13.stillwater-sc.com"}
        assert si.machine_fingerprint(a) == si.machine_fingerprint(b)

    def test_definition_version_splits_a_series(self, tree):
        doc = _load(tree, I7_B)
        for r in doc["results"]:
            r["version"] = 2
        _dump(tree, I7_B, doc)
        con, _ = bench.ingest(tree)
        n = _scalar(
            con,
            """SELECT count(*) FROM (
                 SELECT series_key FROM trend_results WHERE machine = 'i7-12700k'
                 GROUP BY series_key HAVING count(*) = 2)""",
        )
        assert n == 0
        assert _scalar(con, "SELECT max(definition_version) FROM entry_definitions") == 2

    @pytest.mark.parametrize("check", ["missing", "failed"])
    def test_result_without_a_passing_check_is_excluded_from_trends(self, tree, check):
        doc = _load(tree, I7_A)
        first = doc["results"][0]
        if check == "missing":
            del first["check"]
        else:
            first["check"]["ok"] = False
        _dump(tree, I7_A, doc)
        con, _ = bench.ingest(tree)
        run_id = _scalar(con, "SELECT run_id FROM runs WHERE path = ?", [I7_A])
        eid = "/".join(first[k] for k in ("area", "op", "dtype", "shape"))
        key = [run_id, eid]
        where = "WHERE run_id = ? AND entry_id = ?"
        stored = _scalar(con, f"SELECT check_ok FROM results {where}", key)
        assert stored is (None if check == "missing" else False)
        assert _scalar(con, f"SELECT count(*) FROM trend_results {where}", key) == 0
        total = _scalar(con, "SELECT count(*) FROM results")
        assert _scalar(con, "SELECT count(*) FROM trend_results") == total - 1


# --------------------------------------------------------------------------
# Command line, and an optional full MTL5 checkout
# --------------------------------------------------------------------------


def test_cli_ingest(tree, tmp_path, capsys):
    db = tmp_path / "cli.duckdb"
    (tree / (I7_A + ".sysinfo")).unlink()
    assert cli_main(["ingest", str(tree), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "5 files, 5 new rows" in out
    assert f"incomplete {I7_A}: no .sysinfo sidecar" in out
    assert "trend-eligible: 2 runs" in out


@pytest.mark.skipif(
    not os.environ.get("MTL5_BENCH_DATA"),
    reason="set MTL5_BENCH_DATA to an MTL5 checkout's benchmarks/data",
)
def test_full_mtl5_checkout():
    root = Path(os.environ["MTL5_BENCH_DATA"])
    con, report = bench.ingest(root)
    on_disk = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(root).parts[:-1])
        and "__pycache__" not in p.parts
    }
    assert {o.path for o in report.outcomes} == on_disk
    assert not [o for o in report.outcomes if o.status == "rejected"]
    assert _scalar(con, "SELECT count(*) FROM runs WHERE tier = 'complete'") > 0
