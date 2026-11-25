"""Legacy benchmark formats: MTL5's CSVs and bench_blas.py JSON (#121 milestone P2).

The fixtures under tests/data/bench_legacy are unmodified files copied from
stillwater-sc/mtl5 benchmarks/data, one per CSV family worth exercising:

- an interleave_rows CSV with a full sidecar, the one family that reaches
  the complete tier;
- a blocking_ab CSV, whose harness records no clock policy or CPU set;
- a column_sweep CSV, in the shell harnesses' sidecar dialect;
- an n_c model sweep, which is predictions only;
- a BLAS sweep with an older, thinner sidecar;
- the KLU scoreboard, which has no sidecar and is grandfathered in
  .sidecar-exempt.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from mtl5 import bench  # noqa: E402
from mtl5.bench import formats as fm  # noqa: E402
from mtl5.bench import sysinfo as si  # noqa: E402

FIXTURES = Path(__file__).parent / "data" / "bench_legacy"
P1 = Path(__file__).parent / "data" / "bench"
INTERLEAVE = "arduino-ventuno-q/interleave_rows/20261111T185721Z/interleave-g++.csv"
BLOCKING = "xeon-e5-2420-ccap2/blocking_ab_default.csv"
COLUMN = "i7-12700k/column_sweep/20261117T133905Z/g++-native.csv"
MODEL = "i7-12700k/nc_model_sweep_double.csv"
SWEEP = "blas_sweep_blis.csv"
KLU = "klu_scoreboard.csv"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(FIXTURES, root)
    return root


def _ingest(root, db=None, **kw):
    return bench.ingest(root, db if db is not None else ":memory:", time_from_git=False, **kw)


def _status(report, rel):
    (o,) = [o for o in report.outcomes if o.path == rel]
    return o


def _scalar(con, sql, params=None):
    return con.execute(sql, params or []).fetchone()[0]


def _run_id(con, rel):
    return _scalar(con, "SELECT run_id FROM runs WHERE path = ?", [rel])


def _complete_sidecar() -> str:
    """The interleave fixture's sidecar: every key a legacy run needs."""
    return (FIXTURES / (INTERLEAVE + ".sysinfo")).read_text()


# --------------------------------------------------------------------------
# The format table
# --------------------------------------------------------------------------


class TestFormatTable:
    @pytest.mark.parametrize("fmt", fm.FORMATS + (fm.BENCH_BLAS,), ids=lambda f: f.name)
    def test_columns_have_one_role(self, fmt):
        params, metrics = set(fmt.params), set(fmt.metrics)
        assert not params & metrics
        assert fmt.repeat is None or fmt.repeat not in params | metrics
        for metric, (column, _) in fmt.valid_if.items():
            assert metric in fmt.metrics and column not in metrics

    def test_names_are_unique(self):
        names = [f.name for f in fm.FORMATS + (fm.BENCH_BLAS,)]
        assert len(names) == len(set(names))

    def test_most_specific_first(self):
        # A timing table has every column the sweep's signature asks for
        # except packedB_bytes, and the sweep must not swallow a timing table.
        timing = ["model", "dtype", "m", "n", "k", "threads", "nc", "best_seconds", "gflops"]
        assert fm.match_csv(timing).name == "nc_model_timing"
        assert fm.match_csv(timing + ["packedB_bytes"]).name == "nc_model_timing"

    def test_comment_lines_are_kept_apart(self):
        t = fm.parse_csv("# one\n# two\n\nkernel,type\ngemv,f64\n")
        assert t.comments == ["one", "two"]
        assert t.header == ["kernel", "type"] and t.rows == [{"kernel": "gemv", "type": "f64"}]

    @pytest.mark.parametrize(
        "text, message",
        [
            ("# only comments\n", "no header row"),
            ("a,a\n1,2\n", "repeats a column"),
            ("a,b\n1,2,3\n", "line 2 has 3 fields"),
        ],
    )
    def test_malformed_tables(self, text, message):
        with pytest.raises(ValueError, match=message):
            fm.parse_csv(text)

    @pytest.mark.parametrize("text", ["", "nan", "inf", "n/a", "-"])
    def test_non_numbers(self, text):
        assert fm.number(text) is None


# --------------------------------------------------------------------------
# The fixture tree
# --------------------------------------------------------------------------


class TestFixtures:
    def test_every_file_is_accounted_for(self, tree):
        _, report = _ingest(tree)
        on_disk = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file())
        assert sorted(o.path for o in report.outcomes) == on_disk
        got = {o.path: (o.kind, o.status) for o in report.outcomes}
        assert got[INTERLEAVE] == ("interleave_rows/1", "ingested")
        assert got[BLOCKING] == ("blocking_ab/1", "incomplete")
        assert got[COLUMN] == ("column_sweep/1", "incomplete")
        assert got[MODEL] == ("nc_model_sweep/1", "ignored")
        assert got[SWEEP] == ("bench_all/1", "incomplete")
        assert got[KLU] == ("solver_scoreboard/1", "incomplete")

    def test_the_reasons(self, tree):
        _, report = _ingest(tree)
        assert _status(report, BLOCKING).reasons == [
            "sidecar lacks clock_policy, bench_cpus",
            "no run time: the directory name is not a UTC timestamp, "
            "and git has no commit adding the file",
        ]
        assert _status(report, MODEL).reasons == ["model output: no measured columns"]
        assert _status(report, KLU).reasons[0] == (
            "no .sysinfo sidecar (grandfathered in .sidecar-exempt)"
        )

    def test_harness_aliases_fill_the_sidecar(self, tree):
        _, report = _ingest(tree)
        lacks = _status(report, COLUMN).reasons[0]
        # host, cpu and commit arrive under their own names in this dialect.
        for key in ("preflight_host", "cpu_brand", "git_commit"):
            assert key not in lacks
        assert "governor" in lacks

    def test_interleave_measurements(self, tree):
        con, _ = _ingest(tree)
        table = fm.parse_csv((tree / INTERLEAVE).read_text())
        rid = _run_id(con, INTERLEAVE)
        assert _scalar(con, "SELECT count(*) FROM measurements WHERE run_id = ?", [rid]) == (
            2 * len(table.rows)
        )
        assert _scalar(con, "SELECT time_source FROM runs WHERE run_id = ?", [rid]) == "directory"
        # compiler_version came from its column; the sidecar has none.
        side = si.parse_sysinfo((tree / (INTERLEAVE + ".sysinfo")).read_text())[0]
        assert "compiler_version" not in side
        assert _status(_ingest(tree)[1], INTERLEAVE).reasons == []

    def test_repeats_are_points_of_one_series(self, tree):
        con, _ = _ingest(tree)
        rows = con.execute(
            "SELECT series_key, count(*), count(DISTINCT repeat), min(sample), max(sample) "
            "FROM trend_measurements WHERE format = 'interleave_rows/1' AND metric = 'gbps' "
            "GROUP BY series_key"
        ).fetchall()
        runs = {r["run"] for r in fm.parse_csv((tree / INTERLEAVE).read_text()).rows}
        assert rows and all(n == len(runs) == reps for _, n, reps, _, _ in rows)
        assert all((lo, hi) == (0, len(runs) - 1) for *_, lo, hi in rows)

    def test_params_and_attributes(self, tree):
        con, _ = _ingest(tree)
        params, attrs, value = con.execute(
            "SELECT params, attributes, value FROM measurements "
            "WHERE format = 'blocking_ab/1' AND metric = 'gflops' AND row_index = 0"
        ).fetchone()
        first = fm.parse_csv((tree / BLOCKING).read_text()).rows[0]
        assert json.loads(params) == {
            k: first[k] for k in ("arm", "dtype", "m", "n", "k", "threads")
        }
        assert json.loads(attrs)["checksum"] == first["checksum"]
        assert value == pytest.approx(float(first["gflops"]))

    def test_only_complete_runs_trend(self, tree):
        con, _ = _ingest(tree)
        formats = {
            f for (f,) in con.execute("SELECT DISTINCT format FROM trend_measurements").fetchall()
        }
        assert formats == {"interleave_rows/1"}

    def test_reingest_appends_nothing(self, tree):
        con, first = _ingest(tree)
        n = _scalar(con, "SELECT count(*) FROM measurements")
        con, second = _ingest(tree, con)
        assert second.new_rows == 0 and _scalar(con, "SELECT count(*) FROM measurements") == n


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


class TestEdges:
    def test_unknown_header_is_deferred_with_the_header(self, tree):
        (tree / "new_harness.csv").write_text("alpha,beta\n1,2\n")
        _, report = _ingest(tree)
        o = _status(report, "new_harness.csv")
        assert (o.kind, o.status) == ("csv", "deferred")
        assert o.reasons == ["no format matches the header alpha,beta"]

    @pytest.mark.parametrize(
        "content, reason",
        [
            (b"kernel,type\n\xff\n", "the CSV is not UTF-8"),
            (b"a,b\n1,2,3\n", "invalid CSV: line 2 has 3 fields, the header 2"),
        ],
    )
    def test_unreadable_csv(self, tree, content, reason):
        (tree / "bad.csv").write_bytes(content)
        _, report = _ingest(tree)
        o = _status(report, "bad.csv")
        assert (o.status, o.reasons) == ("rejected", [reason])

    def test_header_only(self, tree):
        rel = "arduino-ventuno-q/interleave_rows/20261111T185721Z/empty.csv"
        header = (tree / INTERLEAVE).read_text().splitlines()[0]
        (tree / rel).write_text(header + "\n")
        (tree / (rel + ".sysinfo")).write_text(_complete_sidecar())
        _, report = _ingest(tree)
        assert "the table has no data rows" in _status(report, rel).reasons

    def test_a_column_supplying_a_sidecar_key_must_agree(self, tree):
        text = (tree / INTERLEAVE).read_text().splitlines()
        cols = text[0].split(",")
        i = cols.index("compiler_version")
        row = text[1].split(",")
        row[i] = "other"
        (tree / INTERLEAVE).write_text("\n".join([text[0], ",".join(row)] + text[2:]) + "\n")
        _, report = _ingest(tree)
        o = _status(report, INTERLEAVE)
        assert (
            o.status == "rejected" and "'compiler_version' has 2 different values" in o.reasons[0]
        )

    def test_a_column_contradicting_the_sidecar(self, tree):
        side = tree / (INTERLEAVE + ".sysinfo")
        side.write_text(side.read_text() + "compiler_version=gcc 99\n")
        _, report = _ingest(tree)
        o = _status(report, INTERLEAVE)
        assert o.status == "rejected"
        assert o.reasons[0].startswith("sidecar says compiler_version='gcc 99', column")

    def test_status_gates_a_scoreboard_value(self, tree):
        rel = "i7-12700k/superlu/20261117T202641Z/x.csv"
        (tree / rel).parent.mkdir(parents=True)
        (tree / rel).write_text(
            "matrix,n,nnz,native_status,native_s,ext_status,ext_s\na,10,20,ok,0.5,singular,0.25\n"
        )
        con, _ = _ingest(tree)
        got = dict(
            con.execute(
                "SELECT metric, valid FROM measurements m JOIN runs r USING (run_id) "
                "WHERE r.path = ?",
                [rel],
            ).fetchall()
        )
        assert got == {"native_s": True, "ext_s": False}

    def test_absent_format_columns_are_skipped(self, tree):
        # The older scoreboard has no snode_* columns.
        con, _ = _ingest(tree)
        metrics = {
            m
            for (m,) in con.execute(
                "SELECT DISTINCT metric FROM measurements WHERE format = 'solver_scoreboard/1'"
            ).fetchall()
        }
        assert metrics and not any(m.startswith("snode") for m in metrics)

    def test_duplicate_rows_without_a_repeat_column_are_samples(self, tree):
        # run_blocking_ab.sh writes one row per round with no round column, so
        # the same parameters recur; each occurrence is the next sample.
        table = fm.parse_csv((tree / BLOCKING).read_text())
        keys = [tuple(r[k] for k in fm.FORMATS[2].params) for r in table.rows]
        assert len(set(keys)) < len(keys)
        con, _ = _ingest(tree)
        got = con.execute(
            "SELECT row_index, sample FROM measurements "
            "WHERE format = 'blocking_ab/1' AND metric = 'gflops' ORDER BY row_index"
        ).fetchall()
        assert [s for _, s in got] == [keys[:i].count(k) for i, k in enumerate(keys)]

    def test_a_non_number_is_a_null_value(self, tree):
        text = (tree / INTERLEAVE).read_text().splitlines()
        cols = text[0].split(",")
        row = text[1].split(",")
        row[cols.index("gbps")] = "nan"
        (tree / INTERLEAVE).write_text("\n".join([text[0], ",".join(row)] + text[2:]) + "\n")
        con, _ = _ingest(tree)
        value = _scalar(
            con,
            "SELECT value FROM measurements WHERE format = 'interleave_rows/1' "
            "AND row_index = 0 AND metric = 'gbps'",
        )
        assert value is None
        assert (
            _scalar(
                con,
                "SELECT count(*) FROM trend_measurements WHERE row_index = 0 AND metric = 'gbps'",
            )
            == 0
        )


# --------------------------------------------------------------------------
# bench_blas.py JSON
# --------------------------------------------------------------------------


def _bench_blas_doc():
    return {
        "meta": {"mtl5_version": "5.25.0"},
        "results": [
            {
                "kernel": "gemm",
                "dtype": "f64",
                "n": 256,
                "status": "ok",
                "accumulator": "default",
                "seconds_per_op": 0.002,
                "gflops": 16.7,
                "baseline_overhead_dominated": False,
            },
            {
                "kernel": "gemm",
                "dtype": "posit32",
                "n": 256,
                "status": "too_slow",
                "accumulator": "default",
                "seconds_per_op": None,
                "gflops": None,
                "baseline_overhead_dominated": False,
            },
        ],
    }


class TestBenchBlas:
    REL = "i7-12700k/bench_blas/20261124T120000Z/bench_blas.json"

    def _write(self, tree, doc, sidecar=True):
        (tree / self.REL).parent.mkdir(parents=True)
        (tree / self.REL).write_text(json.dumps(doc))
        if sidecar:
            text = (P1 / "i7-12700k/mtl5_bench/20261111T190041Z/native-t1.json.sysinfo").read_text()
            (tree / (self.REL + ".sysinfo")).write_text(text)

    def test_ingested_with_status_gating(self, tree):
        self._write(tree, _bench_blas_doc())
        con, report = _ingest(tree)
        assert _status(report, self.REL).kind == "bench_blas/1"
        assert _status(report, self.REL).status == "ingested"
        rows = con.execute(
            "SELECT json_extract_string(params, '$.dtype'), metric, valid, value "
            "FROM measurements m JOIN runs r USING (run_id) WHERE r.path = ? ORDER BY 1, 2",
            [self.REL],
        ).fetchall()
        assert rows == [
            ("f64", "gflops", True, pytest.approx(16.7)),
            ("f64", "seconds_per_op", True, pytest.approx(0.002)),
            ("posit32", "gflops", False, None),
            ("posit32", "seconds_per_op", False, None),
        ]

    def test_a_failed_result_without_metric_keys(self, tree):
        doc = _bench_blas_doc()
        for key in ("seconds_per_op", "gflops"):
            del doc["results"][1][key]
        self._write(tree, doc)
        con, report = _ingest(tree)
        assert _status(report, self.REL).status == "ingested"
        rid = _run_id(con, self.REL)
        n = "SELECT count(*) FROM measurements WHERE run_id = ? AND NOT valid"
        assert _scalar(con, n, [rid]) == 2

    def test_without_a_sidecar(self, tree):
        self._write(tree, _bench_blas_doc(), sidecar=False)
        _, report = _ingest(tree)
        assert _status(report, self.REL).reasons == ["no .sysinfo sidecar"]

    @pytest.mark.parametrize(
        "mutate, reason",
        [
            (lambda d: d.update(results={}), "'results' is not a list of objects"),
            (lambda d: d["results"][0].pop("status"), "results[0] has no 'status'"),
            (lambda d: d["results"][0].update(n=[1]), "results[0].n is not a scalar"),
        ],
    )
    def test_rejections(self, tree, mutate, reason):
        doc = _bench_blas_doc()
        mutate(doc)
        self._write(tree, doc)
        _, report = _ingest(tree)
        o = _status(report, self.REL)
        assert (o.kind, o.status) == ("bench_blas/1", "rejected")
        assert o.reasons == [f"bench_blas JSON: {reason}"]


# --------------------------------------------------------------------------
# Run time from git
# --------------------------------------------------------------------------


def _git(root: Path, *args: str, date: str = "2026-11-20T10:00:00Z") -> None:
    env = dict(os.environ, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env=env,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
class TestGitTime:
    REL = "arduino-ventuno-q/interleave-root.csv"

    def _tree(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (root / "arduino-ventuno-q").mkdir()
        shutil.copy(FIXTURES / INTERLEAVE, root / self.REL)
        (root / (self.REL + ".sysinfo")).write_text(_complete_sidecar())
        _git(root, "init", "-q")
        return root

    def test_the_adding_commit_is_the_run_time(self, tmp_path):
        root = self._tree(tmp_path)
        _git(root, "add", ".")
        _git(root, "commit", "-q", "-m", "add")
        con, report = bench.ingest(root)
        assert _status(report, self.REL).status == "ingested"
        ts, source = con.execute(
            "SELECT run_timestamp, time_source FROM runs WHERE path = ?", [self.REL]
        ).fetchone()
        assert (ts.isoformat(), source) == ("2026-11-20T10:00:00", "git")

    def test_committing_later_reevaluates(self, tmp_path):
        root = self._tree(tmp_path)
        con, report = bench.ingest(root)
        assert _status(report, self.REL).status == "incomplete"
        _git(root, "add", ".")
        _git(root, "commit", "-q", "-m", "add")
        con, report = bench.ingest(root, con)
        o = _status(report, self.REL)
        assert o.new and o.status == "ingested"

    def test_a_renamed_file_takes_the_commit_that_moved_it(self, tmp_path):
        root = self._tree(tmp_path)
        _git(root, "add", ".")
        _git(root, "commit", "-q", "-m", "add")
        moved = "arduino-ventuno-q/moved.csv"
        _git(root, "mv", self.REL, moved)
        _git(root, "mv", self.REL + ".sysinfo", moved + ".sysinfo")
        _git(root, "commit", "-q", "-m", "move", date="2026-11-21T10:00:00Z")
        con, report = bench.ingest(root)
        assert _status(report, moved).status == "ingested"
        ts = _scalar(con, "SELECT run_timestamp FROM runs WHERE path = ?", [moved])
        assert ts.isoformat() == "2026-11-21T10:00:00"

    def test_switched_off(self, tmp_path):
        root = self._tree(tmp_path)
        _git(root, "add", ".")
        _git(root, "commit", "-q", "-m", "add")
        _, report = bench.ingest(root, time_from_git=False)
        assert _status(report, self.REL).status == "incomplete"

    def test_outside_a_work_tree(self, tmp_path):
        root = self._tree(tmp_path)
        shutil.rmtree(root / ".git")
        _, report = bench.ingest(root)
        assert "no run time" in _status(report, self.REL).reasons[0]


@pytest.mark.skipif(
    not os.environ.get("MTL5_BENCH_DATA"),
    reason="set MTL5_BENCH_DATA to an MTL5 checkout's benchmarks/data",
)
def test_every_mtl5_csv_has_a_format():
    root = Path(os.environ["MTL5_BENCH_DATA"])
    _, report = bench.ingest(root)
    csvs = [o for o in report.outcomes if o.path.endswith(".csv")]
    assert csvs
    assert not [o for o in csvs if o.status in ("deferred", "rejected")]
