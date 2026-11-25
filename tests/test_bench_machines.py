"""Tests for the cross-machine analyses, tables and figures (#121 P4).

The fixture (``tests/data/bench_machines``) is MTL5's committed data from
three machines at one commit (``bfa1c28``):

- the i7-12700k: two 8-thread sessions, and a 1-thread run;
- the Jetson Orin: two 6-thread sessions;
- the Arduino Ventuno Q: one 4-thread session.

As for the figures of P3, the tests compare data, not pixels. The analyses
reproduce pinned frames, the tables reproduce pinned Markdown, the figures'
artists hold the frames' numbers, and a redraw from a saved frame gives the
same files.

To re-pin after an intended change:
``MTL5_BENCH_REPIN=1 pytest tests/test_bench_machines.py``.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from mtl5.bench import analyze, ingest, plot, style, tables  # noqa: E402
from mtl5.bench.cli import main  # noqa: E402

DATA = Path(__file__).parent / "data" / "bench_machines"
PINNED = Path(__file__).parent / "data" / "bench_machines_expected"
ENTRIES = ["blas1/dot/f64", "blas3/gemm/f64", "sparse/spmv/f64"]
I7_OLD = "i7-12700k/mtl5_bench/20261111T190041Z/native-t8.json"
JETSON = "jetson-orin/mtl5_bench/20261111T193624Z/native-t6.json"

FRAMES = {
    "machines": lambda db: analyze.machines(db),
    "compare": lambda db: analyze.cross_machine(db, ENTRIES),
}


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """Run each test in its own directory: the CLI writes to the current one."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("machines") / "bench.duckdb"
    con, _ = ingest(DATA, path, time_from_git=False)
    con.close()
    return str(path)


def _roundtrip(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return plot.read_frame(buf)


def _pinned(kind: str):
    return plot.read_frame(PINNED / f"{kind}.csv")


def _copy(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(DATA, root)
    return root


def _edit_sysinfo(path: Path, **fields: str) -> None:
    lines = path.read_text().splitlines()
    out = [f"{k}={fields[k]}" if (k := line.split("=", 1)[0]) in fields else line for line in lines]
    path.write_text("\n".join(out) + "\n")


def _repin(name: str, text: str) -> None:
    if os.environ.get("MTL5_BENCH_REPIN"):
        PINNED.mkdir(exist_ok=True)
        (PINNED / name).write_text(text)


# --- pinned frames and tables -----------------------------------------------------


@pytest.mark.parametrize("kind", sorted(FRAMES))
def test_analysis_reproduces_the_pinned_frame(db, kind):
    frame = _roundtrip(FRAMES[kind](db))
    _repin(f"{kind}.csv", frame.to_csv(index=False))
    pd.testing.assert_frame_equal(frame, _pinned(kind))


def test_machine_table_reproduces_the_pinned_markdown():
    df = _pinned("machines")
    text = tables.to_markdown(tables.machine_table(df), footer=plot.provenance(df))
    _repin("machines.md", text)
    assert text == (PINNED / "machines.md").read_text()


def test_compare_table_reproduces_the_pinned_markdown():
    df = _pinned("compare")
    text = tables.to_markdown(tables.compare_table(df, spread=True), footer=plot.provenance(df))
    _repin("compare.md", text)
    assert text == (PINNED / "compare.md").read_text()


# --- which runs a machine contributes ---------------------------------------------


def test_each_machine_contributes_its_sessions_at_its_most_threads(db):
    df = analyze.machines(db)
    got = df.groupby("machine").agg(runs=("run_id", "nunique"), threads=("threads", "max"))
    assert got.to_dict("index") == {
        "arduino-ventuno-q": {"runs": 1, "threads": 4},
        "i7-12700k": {"runs": 2, "threads": 8},
        "jetson-orin": {"runs": 2, "threads": 6},
    }


def test_a_fixed_thread_count_keeps_the_machines_that_ran_it(db):
    df = analyze.machines(db, threads=1)
    assert set(df["machine"]) == {"i7-12700k"} and set(df["threads"]) == {1}


def test_only_the_latest_commits_sessions_count(tmp_path):
    # The older i7 session at another commit is another build of the code: it
    # is not a second measurement of the latest one.
    root = _copy(tmp_path)
    _edit_sysinfo(
        root / f"{I7_OLD}.sysinfo", git_commit="0000000000aa", tree_git_commit="0000000000aa"
    )
    con, _ = ingest(root, time_from_git=False)
    df = analyze.machines(con, machines=["i7-12700k"])
    con.close()
    assert len(df) == 1 and set(df["git_commit"]) == {"bfa1c2822688"}


def test_dirty_sessions_do_not_count(tmp_path):
    root = _copy(tmp_path)
    _edit_sysinfo(root / f"{JETSON}.sysinfo", git_dirty="1")
    con, _ = ingest(root, time_from_git=False)
    df = analyze.machines(con, machines=["jetson-orin"], tiers=("complete", "incomplete"))
    con.close()
    # The dirty session was the latest; the clean, older one remains.
    assert len(df) == 1
    assert str(pd.Timestamp(df.at[0, "run_timestamp"])) == "2026-11-11 19:06:10"


def test_machines_can_be_selected_by_name(db):
    df = analyze.machines(db, machines=["jetson-orin", "arduino-ventuno-q"])
    assert set(df["machine"]) == {"jetson-orin", "arduino-ventuno-q"}


def test_an_unknown_machine_lists_the_known_ones(db):
    with pytest.raises(LookupError, match="no complete mtl5-bench run of nope; machines: "):
        analyze.machines(db, machines=["nope", "i7-12700k"])


@pytest.mark.parametrize("threads", [0, -1, "abc", "1.5"])
def test_threads_must_be_max_or_a_positive_integer(db, threads):
    with pytest.raises(ValueError, match="threads must be"):
        analyze.machines(db, threads=threads)


def test_a_named_machine_without_that_thread_count_is_an_error(db):
    # It was dropped from the comparison without a word.
    with pytest.raises(LookupError, match="no 1-thread run of jetson-orin"):
        analyze.machines(db, threads=1, machines=["i7-12700k", "jetson-orin"])


def test_no_machine_ran_the_thread_count(db):
    with pytest.raises(LookupError, match="threads=3"):
        analyze.machines(db, threads=3)


# --- what the frames hold -----------------------------------------------------------


def test_ridge_is_peak_over_bandwidth(db):
    df = analyze.machines(db)
    expected = df["fma_peak_f64_gflops"] / df["bandwidth_gbytes_per_s"]
    assert df["ridge_f64"].tolist() == pytest.approx(expected.tolist())


def test_compare_keeps_the_named_entries_and_flags(db):
    df = analyze.cross_machine(db, ["blas3/gemm"], metric="roofline_pct")
    assert {e.split("/")[1] for e in df["entry_id"]} == {"gemm"}
    assert {"cache_resident", "suspect_model"} <= set(df.columns)
    assert set(df["unit"]) == {"% of roofline"}


def test_compare_refuses_an_unknown_metric(db):
    with pytest.raises(ValueError, match="unknown metric"):
        analyze.cross_machine(db, ENTRIES, metric="speedup")


# --- tables ------------------------------------------------------------------------------


def test_machine_table_is_the_median_of_each_machines_runs():
    df = _pinned("machines")
    t = tables.machine_table(df).set_index("Machine")
    i7 = df[df["machine"] == "i7-12700k"]
    assert t.at["i7-12700k", "Runs"] == 2
    assert t.at["i7-12700k", "f64 peak, GFLOP/s"] == pytest.approx(
        i7["fma_peak_f64_gflops"].median()
    )
    assert t.at["i7-12700k", "Commit"] == "bfa1c28"


def test_an_unknown_cache_size_is_blank_not_zero():
    # The probe reports 0 when it finds no cache size; 0 MiB would be a claim.
    df = _pinned("machines")
    df.loc[df["machine"] == "jetson-orin", "last_level_cache_bytes"] = 0
    t = tables.machine_table(df).set_index("Machine")
    assert pd.isna(t.at["jetson-orin", "LLC, MiB"])
    assert "| jetson-orin |" in tables.to_markdown(tables.machine_table(df))
    row = next(r for r in tables.to_markdown(t.reset_index()).splitlines() if "jetson" in r)
    assert "| 0 |" not in row


def test_compare_table_cells():
    df = _pinned("compare")
    plain = tables.compare_table(df).set_index("Entry")
    gemm = df[df["entry_id"].str.startswith("blas3/gemm/f64") & (df["machine"] == "i7-12700k")]
    assert plain.at["blas3/gemm/f64", "i7-12700k (8 thr)"] == pytest.approx(gemm["value"].median())
    spread = tables.compare_table(df, spread=True).set_index("Entry")
    lo, hi = gemm["value"].min(), gemm["value"].max()
    cell = spread.at["blas3/gemm/f64", "i7-12700k (8 thr)"]
    assert cell.endswith(f"({style.number(lo)}–{style.number(hi)})")
    # One run: no range to show.
    assert "(" not in spread.at["blas3/gemm/f64", "arduino-ventuno-q (4 thr)"]


def test_markdown_escapes_pipes_and_keeps_integers():
    t = pd.DataFrame({"name": ["a|b"], "n": [3], "x": [1201.4], "y": [0.000508]})
    text = tables.to_markdown(t, footer="f")
    assert "| a\\|b | 3 | 1201 | 0.000508 |" in text
    assert text.splitlines()[1] == "|---|---:|---:|---:|"
    assert text.endswith("\n*f*\n")


@pytest.mark.parametrize(
    ("x", "text"), [(1201.4, "1201"), (78.288, "78.3"), (0.000508, "0.000508"), (999.6, "1000")]
)
def test_numbers_never_use_an_exponent_when_large(x, text):
    assert style.number(x) == text


# --- figures ---------------------------------------------------------------------------------


def test_balance_points_are_each_machines_median():
    df = _pinned("machines")
    fig = plot.machine_balance(df)
    ax = fig.figure.axes[0]
    labels = plot._machine_labels(df)
    points = {c.get_label(): c for c in ax.containers}
    assert set(points) == set(labels.values())
    for m, label in labels.items():
        g = df[df["machine"] == m]
        x, y = points[label].lines[0].get_data()
        assert x[0] == pytest.approx(g["bandwidth_gbytes_per_s"].median())
        assert y[0] == pytest.approx(g["fma_peak_f64_gflops"].median())
    guides = [t.get_text() for t in ax.texts]
    assert guides and all(t.endswith("FLOP/B") for t in guides)


def test_balance_refuses_an_unknown_precision():
    with pytest.raises(ValueError, match="precision"):
        plot.machine_balance(_pinned("machines"), precision="f16")


def test_compare_markers_are_each_machines_median():
    df = _pinned("compare")
    fig = plot.cross_machine(df)
    ax = fig.figure.axes[0]
    labels = plot._machine_labels(df)
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    for m, label in labels.items():
        medians = df[df["machine"] == m].groupby("entry_id")["value"].median()
        drawn = list(lines[label].get_xdata())
        if f"_{label}, flagged" in lines:  # flagged results are drawn hollow
            drawn += list(lines[f"_{label}, flagged"].get_xdata())
        assert sorted(drawn) == sorted(medians.tolist())


def test_compare_draws_flagged_results_hollow():
    df = _pinned("compare")
    flag = df["entry_id"].str.startswith("sparse/spmv") & (df["machine"] == "i7-12700k")
    df["suspect_model"] = flag
    fig = plot.cross_machine(df)
    ax = fig.figure.axes[0]
    hollow = [ln for ln in ax.get_lines() if ln.get_markerfacecolor() == "white"]
    assert [ln.get_label() for ln in hollow] == ["_i7-12700k (8 threads), flagged"]
    assert len(hollow[0].get_xdata()) == 1
    assert "hollow: flagged by MTL5" in ax.get_legend().get_texts()[-1].get_text()


def test_compare_marks_100_percent_of_roofline(db):
    df = analyze.cross_machine(db, ENTRIES, metric="roofline_pct")
    ax = plot.cross_machine(df).figure.axes[0]
    assert any(list(ln.get_xdata()) == [100, 100] for ln in ax.get_lines())
    assert ax.get_xscale() == "linear"
    gflops = plot.cross_machine(_pinned("compare")).figure.axes[0]
    assert gflops.get_xscale() == "log"


def test_compare_shows_one_metric():
    df = _pinned("compare")
    other = df.copy()
    other["metric"] = "median_s"
    with pytest.raises(ValueError, match="one metric"):
        plot.cross_machine(pd.concat([df, other]))


@pytest.mark.parametrize("kind", ["balance", "compare"])
def test_a_saved_figure_is_redrawn_from_its_csv(tmp_path, kind):
    frame = _pinned("machines" if kind == "balance" else "compare")
    fig = plot.FIGURES[kind](frame)
    fig.save(tmp_path / kind)
    again = plot.from_csv(tmp_path / f"{kind}.csv", kind)
    again.save(tmp_path / "again", data=False)
    for fmt in ("pdf", "svg", "png"):
        assert (tmp_path / f"{kind}.{fmt}").read_bytes() == (tmp_path / f"again.{fmt}").read_bytes()


@pytest.mark.parametrize("kind", ["balance", "compare"])
def test_cross_machine_figures_have_the_provenance_footer(kind):
    frame = _pinned("machines" if kind == "balance" else "compare")
    fig = plot.FIGURES[kind](frame)
    assert plot.provenance(frame) in [t.get_text() for t in fig.figure.texts]


def test_compare_flags_need_no_pandas_2_1(monkeypatch):
    # DataFrame.map is pandas 2.1; the bench extra allows 2.0.
    df = _pinned("compare")

    def missing(*args, **kwargs):
        raise AttributeError("DataFrame.map is pandas 2.1")

    monkeypatch.setattr(pd.DataFrame, "map", missing, raising=False)
    fig = plot.cross_machine(df)
    hollow = [ln for ln in fig.figure.axes[0].get_lines() if ln.get_markerfacecolor() == "white"]
    assert hollow


def test_balance_leaves_out_runs_without_probes():
    # A missing or zero probe gave NaN or 0 axis limits, and log2 of them raised.
    df = _pinned("machines")
    df.loc[df["machine"] == "jetson-orin", "bandwidth_gbytes_per_s"] = None
    df.loc[df["machine"] == "arduino-ventuno-q", "fma_peak_f64_gflops"] = 0
    fig = plot.machine_balance(df)
    ax = fig.figure.axes[0]
    assert [c.get_label() for c in ax.containers] == ["i7-12700k (8 threads)"]
    assert "no probes: arduino-ventuno-q, jetson-orin" in ax.get_title(loc="left")
    assert len(fig.data) == len(df)  # the figure keeps the whole frame
    df["bandwidth_gbytes_per_s"] = None
    with pytest.raises(ValueError, match="nothing to plot"):
        plot.machine_balance(df)


def test_balance_does_not_mix_rows_that_share_an_index_label():
    df = _pinned("machines")
    expected = df.groupby("machine")["bandwidth_gbytes_per_s"].median()
    df.index = [0] * len(df)
    ax = plot.machine_balance(df).figure.axes[0]
    labels = plot._machine_labels(df)
    for m, label in labels.items():
        (c,) = [c for c in ax.containers if c.get_label() == label]
        assert c.lines[0].get_xdata()[0] == pytest.approx(expected[m])


def test_balance_redraws_at_its_saved_precision(tmp_path):
    fig = plot.machine_balance(_pinned("machines"), precision="f32")
    fig.save(tmp_path / "b", ["svg"])
    again = plot.from_csv(tmp_path / "b.csv", "balance")
    assert again.figure.axes[0].get_ylabel() == "f32 FMA peak (GFLOP/s)"
    assert again._render("svg") == fig._render("svg")
    # An explicit precision still wins over the saved one.
    f64 = plot.from_csv(tmp_path / "b.csv", "balance", precision="f64")
    assert f64.figure.axes[0].get_ylabel() == "f64 FMA peak (GFLOP/s)"


def test_compare_table_keeps_its_unit():
    assert tables.compare_table(_pinned("compare")).attrs["unit"] == "GFLOP/s"


# --- command line --------------------------------------------------------------------------------


def test_cli_table_machines_markdown(db, capsys):
    assert main(["table", "machines", "--db", db]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| Machine | CPU |")
    assert "| i7-12700k |" in out and out.rstrip().endswith("*")


def test_cli_table_compare_csv_to_a_file(db, tmp_path, capsys):
    out = tmp_path / "t.csv"
    args = ["table", "compare", "--db", db, "--entry", "blas3/gemm/f64", "--format", "csv"]
    assert main([*args, "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert list(table["Entry"]) == ["blas3/gemm/f64"]
    assert len(table.columns) == 4


@pytest.mark.parametrize("command", ["table", "figure"])
def test_cli_compare_with_no_matching_entries(db, capsys, command):
    # figure compare said "the frame has no metric" instead of the cause.
    assert main([command, "compare", "--db", db, "--entry", "nothing/here"]) == 1
    assert "no results match" in capsys.readouterr().err


def test_cli_writes_tables_as_utf8(db, tmp_path, monkeypatch):
    # The provenance line has "·". Written in the locale's encoding, it raised
    # on an ASCII locale and gave different bytes on others. The encoding is
    # checked as passed, since the interpreter's locale cannot be changed here.
    calls = []
    write = Path.write_text

    def recording(self, data, *args, **kwargs):
        calls.append(kwargs.get("encoding", args[0] if args else None))
        return write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", recording)
    out = tmp_path / "t.md"
    assert main(["table", "machines", "--db", db, "--out", str(out)]) == 0
    assert calls == ["utf-8"]
    assert "·" in out.read_bytes().decode("utf-8")


@pytest.mark.parametrize("kind", ["balance", "compare"])
def test_cli_cross_machine_figures_refuse_machine(db, kind):
    # --machine was ignored: every machine plotted, into KIND-MACHINE.*.
    with pytest.raises(SystemExit):
        main(["figure", kind, "--db", db, "--machine", "i7-12700k"])


def test_cli_figures_across_machines(db, tmp_path):
    for kind in ("balance", "compare"):
        args = ["figure", kind, "--db", db, "--out", str(tmp_path / kind), "--format", "svg"]
        assert main(args) == 0
        assert (tmp_path / f"{kind}.svg").exists() and (tmp_path / f"{kind}.csv").exists()


def test_cli_threads_max_is_for_cross_machine_kinds(db, capsys):
    args = ["figure", "card", "--db", db, "--machine", "i7-12700k", "--threads", "max"]
    assert main(args) == 1
    assert "--threads max applies" in capsys.readouterr().err


def test_cli_machines_filter(db, capsys):
    assert main(["table", "machines", "--db", db, "--machines", "jetson-orin"]) == 0
    out = capsys.readouterr().out
    assert "| jetson-orin |" in out and "i7-12700k |" not in out
