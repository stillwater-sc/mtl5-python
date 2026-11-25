"""Tests for mtl5.bench.style, mtl5.bench.analyze and mtl5.bench.plot (#121 P3).

The figure tests compare data, not pixels. Each figure is drawn from a pinned
DataFrame (``tests/data/bench_figures_expected``), and the tests check:

- the analysis of the fixture database reproduces the pinned frame;
- the figure's artists hold exactly the frame's numbers;
- the figure is reproducible: drawing the same frame twice, or again from the
  CSV the figure saved, gives the same SVG and PDF bytes.

The fixture (``tests/data/bench_figures``) is a 1-thread and an 8-thread
mtl5_bench run from one MTL5 session on the i7-12700k, and the BLIS
``bench_all`` sweep, all as MTL5 committed them.

To re-pin after an intended change to an analysis:
``MTL5_BENCH_REPIN=1 pytest tests/test_bench_figures.py``.
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

from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from mtl5.bench import analyze, ingest, plot, style  # noqa: E402
from mtl5.bench.cli import main  # noqa: E402
from mtl5.bench.sysinfo import state_class  # noqa: E402

DATA = Path(__file__).parent / "data" / "bench_figures"
PINNED = Path(__file__).parent / "data" / "bench_figures_expected"
RUN_DIR = "i7-12700k/mtl5_bench/20261112T004852Z"
INCOMPLETE = ("complete", "incomplete")

# The frames the figures are pinned to, and how the fixture database makes them.
FRAMES = {
    "card": lambda db: analyze.capability(db, "i7-12700k", threads=1),
    "roofline": lambda db: analyze.roofline(db, "i7-12700k", threads=1),
    "scaling": lambda db: analyze.thread_scaling(
        db, "i7-12700k", entries=["blas1/dot/f64", "blas3/gemm", "sparse/spmv"]
    ),
    "rate": lambda db: analyze.rate_vs_size(
        db, operations=["gemm", "gemv", "dot"], tiers=INCOMPLETE
    ),
}


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("figures") / "bench.duckdb"
    con, _ = ingest(DATA, path, time_from_git=False)
    con.close()
    return str(path)


def _roundtrip(df):
    """The frame as a saved CSV reads it back."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return plot.read_frame(buf)


def _pinned(kind: str):
    return plot.read_frame(PINNED / f"{kind}.csv")


def _copy_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    shutil.copytree(DATA, root)
    return root


def _edit_sysinfo(path: Path, **fields: str) -> None:
    lines = path.read_text().splitlines()
    out = [f"{k}={fields[k]}" if (k := line.split("=", 1)[0]) in fields else line for line in lines]
    path.write_text("\n".join(out) + "\n")


# --- the pinned frames ----------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(FRAMES))
def test_analysis_reproduces_the_pinned_frame(db, kind):
    frame = _roundtrip(FRAMES[kind](db))
    if os.environ.get("MTL5_BENCH_REPIN"):
        PINNED.mkdir(exist_ok=True)
        frame.to_csv(PINNED / f"{kind}.csv", index=False)
    pd.testing.assert_frame_equal(frame, _pinned(kind))


# --- figures hold the frame's numbers ---------------------------------------------


def _lines(fig) -> dict[str, Line2D]:
    ax = fig.figure.axes[0]
    return {line.get_label(): line for line in ax.get_lines()}


def test_scaling_lines_are_the_frames_speedups():
    df = _pinned("scaling")
    fig = plot.thread_scaling(df)
    lines = _lines(fig)
    labels = plot._entry_labels(sorted(df["entry_id"].unique()))
    assert set(lines) == set(labels.values()) | {"ideal"}
    for entry, label in labels.items():
        rows = df[df["entry_id"] == entry].sort_values("threads")
        assert list(lines[label].get_xdata()) == list(rows["threads"])
        assert list(lines[label].get_ydata()) == list(rows["speedup"])
    assert list(lines["ideal"].get_xdata()) == [1, 8]
    assert list(lines["ideal"].get_ydata()) == [1, 8]


def test_rate_lines_are_the_frames_values():
    df = _pinned("rate")
    fig = plot.rate_vs_size(df)
    lines = _lines(fig)
    assert set(lines) == {"dot (blis)", "gemm (blis)", "gemv (blis)"}
    for op in ("dot", "gemm", "gemv"):
        rows = df[df["operation"] == op].sort_values("size")
        assert list(lines[f"{op} (blis)"].get_xdata()) == list(rows["size"])
        assert list(lines[f"{op} (blis)"].get_ydata()) == list(rows["value"])
    assert fig.figure.axes[0].get_ylabel() == "GFLOP/s"


def test_roofline_points_are_the_frames_entries():
    df = _pinned("roofline")
    fig = plot.roofline(df)
    ax = fig.figure.axes[0]
    scatters = {c.get_label(): c for c in ax.collections if isinstance(c, PathCollection)}
    assert set(scatters) == set(df["area"])
    for area, coll in scatters.items():
        rows = df[df["area"] == area]
        got = sorted(map(tuple, coll.get_offsets().tolist()))
        assert got == sorted(zip(rows["intensity"], rows["gflops"]))
    # The ceilings flatten at the hardware peaks, and the slope is the bandwidth.
    row = df.iloc[0]
    tops = sorted(max(line.get_ydata()) for line in ax.get_lines())
    assert tops == pytest.approx(sorted([row["peak_f64_gflops"], row["peak_f32_gflops"]]))
    line = ax.get_lines()[0]
    x0, y0 = line.get_xdata()[0], line.get_ydata()[0]
    assert y0 == pytest.approx(row["bandwidth_gbytes_per_s"] * x0)


def test_roofline_labels_whole_segments_only():
    df = _pinned("roofline")
    fig = plot.roofline(df, label_entries=["blas3/gemm"])
    texts = [a.get_text() for a in fig.figure.axes[0].texts]
    assert "gemm f64" in texts and "gemm f32" in texts
    assert not any(t.startswith("gemmt") for t in texts)


def test_card_shows_the_frames_probes():
    df = _pinned("card")
    fig = plot.capability_card(df)
    texts = [t.get_text() for t in fig.figure.findobj(Text)]
    row = df.iloc[0]
    assert "12th Gen Intel(R) Core(TM) i7-12700K" in texts
    assert "i7-12700k · native · 1 thread" in texts
    peaks = next(t for t in texts if t.startswith(f"{row['fma_peak_f64_gflops']:.3g} f64"))
    assert f"{row['fma_peak_f32_gflops']:.3g} f32" in peaks
    assert f"triad {row['triad_gbytes_per_s']:.3g}" in " ".join(texts)


@pytest.mark.parametrize("kind", sorted(FRAMES))
def test_every_figure_has_the_provenance_footer(kind):
    df = _pinned(kind)
    fig = plot.FIGURES[kind](df)
    footer = plot.provenance(df)
    assert footer in [t.get_text() for t in fig.figure.texts]


# --- reproducibility ----------------------------------------------------------------


def _bytes(fig, fmt) -> bytes:
    return fig._render(fmt)


@pytest.mark.parametrize("kind", sorted(FRAMES))
def test_a_frame_draws_the_same_file_twice(kind):
    df = _pinned(kind)
    for fmt in ("svg", "pdf"):
        assert _bytes(plot.FIGURES[kind](df), fmt) == _bytes(plot.FIGURES[kind](df), fmt)


@pytest.mark.parametrize("kind", sorted(FRAMES))
def test_a_saved_figure_is_redrawn_from_its_csv(tmp_path, kind):
    fig = plot.FIGURES[kind](_pinned(kind))
    paths = fig.save(tmp_path / kind)
    assert [p.suffix for p in paths] == [".pdf", ".svg", ".png", ".csv"]
    again = plot.from_csv(tmp_path / f"{kind}.csv", kind)
    again.save(tmp_path / "again", data=False)
    for fmt in ("pdf", "svg", "png"):
        assert (tmp_path / f"{kind}.{fmt}").read_bytes() == (tmp_path / f"again.{fmt}").read_bytes()


def test_a_redraw_keeps_text_that_looks_like_a_number_or_missing(tmp_path):
    # pandas reads "NA" as missing and "0012345678" as 12345678 by default.
    df = _pinned("scaling")
    df["machine"] = "NA"
    df["git_commit"] = "0012345678"
    fig = plot.thread_scaling(df)
    fig.save(tmp_path / "s", ["svg"])
    again = plot.from_csv(tmp_path / "s.csv", "scaling")
    footer = "machine NA · commit 0012345 · 2 runs, 2026-11-12"
    assert footer in [t.get_text() for t in again.figure.texts]
    assert again._render("svg") == fig._render("svg")


def test_a_redraw_keeps_the_roofline_labels(tmp_path):
    fig = plot.roofline(_pinned("roofline"), label_entries=["blas3/gemm"])
    fig.save(tmp_path / "r", ["svg"])
    again = plot.from_csv(tmp_path / "r.csv", "roofline")
    assert again._render("svg") == fig._render("svg")
    # An explicit selection still wins over the saved one.
    bare = plot.from_csv(tmp_path / "r.csv", "roofline", label_entries=[])
    labels = [t.get_text() for t in bare.figure.axes[0].texts]
    assert "gemm f64" not in labels and "gemm f32" not in labels


@pytest.mark.parametrize("kind", sorted(FRAMES))
def test_a_figure_keeps_its_own_frame(kind):
    # The card and the scaling figure kept the caller's frame, so a change to
    # it after drawing went into the saved CSV but not into the figure.
    df = _pinned(kind)
    fig = plot.FIGURES[kind](df)
    df.loc[:, "git_commit"] = "changed"
    assert "changed" not in set(fig.data["git_commit"])


def test_the_write_order_does_not_move_the_layout(tmp_path):
    # Constrained layout left on reran at each draw from where the last left
    # the axes, so an SVG written after a PNG differed from one written first.
    df = _pinned("card")
    plot.capability_card(df).save(tmp_path / "a", ["png", "svg"], data=False)
    plot.capability_card(df).save(tmp_path / "b", ["svg", "png"], data=False)
    assert (tmp_path / "a.svg").read_bytes() == (tmp_path / "b.svg").read_bytes()


def test_saved_files_carry_no_timestamp(tmp_path):
    fig = plot.thread_scaling(_pinned("scaling"))
    fig.save(tmp_path / "s", ["svg", "pdf"], data=False)
    assert b"<dc:date>" not in (tmp_path / "s.svg").read_bytes()
    assert b"/CreationDate" not in (tmp_path / "s.pdf").read_bytes()


def test_figures_leave_pyplot_alone():
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    for kind in FRAMES:
        plot.FIGURES[kind](_pinned(kind))
    assert plt.get_fignums() == before


def test_a_figure_displays_itself_in_a_notebook():
    png = plot.thread_scaling(_pinned("scaling"))._repr_png_()
    assert png.startswith(b"\x89PNG")


# --- the provenance footer -------------------------------------------------------------


def _prov(**cols):
    base = {
        "run_id": ["a"],
        "machine": ["m"],
        "git_commit": ["0123456789ab"],
        "run_timestamp": ["2026-11-12 00:48:52"],
        "tier": ["complete"],
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_provenance_of_one_complete_run():
    assert plot.provenance(_prov()) == "machine m · commit 0123456 · 1 run, 2026-11-12"


def test_provenance_orders_commits_by_time_and_warns_on_incomplete():
    df = _prov(
        run_id=["b", "a", "b"],
        machine=["m", "n", "m"],
        git_commit=["bbbbbbbbbb", "aaaaaaaaaa", "bbbbbbbbbb"],
        run_timestamp=["2026-11-13 00:00:00", "2026-11-12 00:00:00", "2026-11-13 00:00:00"],
        tier=["incomplete", "complete", "incomplete"],
    )
    assert plot.provenance(df) == (
        "machines m, n · 2 commits aaaaaaa…bbbbbbb · 2 runs, 2026-11-12 to 2026-11-13 · "
        "INCOMPLETE PROVENANCE in 1 of 2 runs"
    )


def test_provenance_with_nothing_recorded():
    df = _prov(machine=[None], git_commit=[None], run_timestamp=[None], tier=["incomplete"])
    assert plot.provenance(df) == (
        "1 run with no machine · commit unrecorded · 1 run · INCOMPLETE PROVENANCE in 1 of 1 runs"
    )


def test_provenance_needs_its_columns():
    with pytest.raises(ValueError, match="provenance columns"):
        plot.provenance(pd.DataFrame({"run_id": ["a"]}))


# --- figure input checks ------------------------------------------------------------------


def test_too_many_series_is_refused():
    df = _pinned("rate")
    with pytest.raises(ValueError, match="more than a legible figure holds"):
        plot.rate_vs_size(df, max_series=2)


def test_an_empty_frame_is_refused():
    with pytest.raises(ValueError, match="nothing to plot"):
        plot.thread_scaling(_pinned("scaling").iloc[0:0])


def test_a_card_is_one_run():
    df = _pinned("card")
    with pytest.raises(ValueError, match="one run"):
        plot.capability_card(pd.concat([df, df]))


def test_a_roofline_is_one_run():
    df = _pinned("roofline")
    other = df.copy()
    other["run_id"] = "other"
    with pytest.raises(ValueError, match="one run"):
        plot.roofline(pd.concat([df, other]))


def test_from_csv_names_the_kinds(tmp_path):
    with pytest.raises(ValueError, match="unknown figure"):
        plot.from_csv(tmp_path / "x.csv", "pie")


# --- analysis -------------------------------------------------------------------------------


def test_scaling_speedup_is_against_the_one_thread_run(db):
    df = analyze.thread_scaling(db, "i7-12700k", entries=["blas3/gemm/f64"])
    t1 = df[df["threads"] == 1].iloc[0]
    t8 = df[df["threads"] == 8].iloc[0]
    assert t1["speedup"] == 1.0 and t1["efficiency"] == 1.0
    assert t8["speedup"] == pytest.approx(t1["median_s"] / t8["median_s"])
    assert t8["efficiency"] == pytest.approx(t8["speedup"] / 8)


def test_scaling_matches_runs_pinned_to_different_cpus(db):
    # The two runs differ in cpu_affinity and bench_cpus, as every 1- and
    # N-thread pair does: that is placement, not a different machine state.
    df = analyze.thread_scaling(db, "i7-12700k")
    assert sorted(df["threads"].unique()) == [1, 8]
    assert df["state_class"].nunique() == 2


def test_scaling_entries_match_whole_segments(db):
    df = analyze.thread_scaling(db, "i7-12700k", entries=["blas3/gemm"])
    assert {e.split("/")[1] for e in df["entry_id"]} == {"gemm"}


def test_scaling_needs_a_comparable_one_thread_run(tmp_path):
    root = _copy_fixture(tmp_path)
    _edit_sysinfo(root / RUN_DIR / "native-t1.json.sysinfo", git_commit="0000000000aa")
    con, _ = ingest(root, time_from_git=False)
    with pytest.raises(LookupError, match="same commit"):
        analyze.thread_scaling(con, "i7-12700k")
    con.close()


def test_scaling_reference_is_the_multithreaded_run(tmp_path):
    # Both runs share their directory's time. The reference was "the latest
    # run", which fell to the 1-thread run by id, and the 8-thread run, at
    # another commit, then dropped out without an error.
    root = _copy_fixture(tmp_path)
    _edit_sysinfo(root / RUN_DIR / "native-t8.json.sysinfo", git_commit="0000000000aa")
    con, _ = ingest(root, time_from_git=False)
    with pytest.raises(LookupError, match="same commit"):
        analyze.thread_scaling(con, "i7-12700k")
    con.close()


def test_scaling_needs_a_multithreaded_run(tmp_path):
    root = _copy_fixture(tmp_path)
    for f in ("native-t8.json", "native-t8.json.sysinfo"):
        (root / RUN_DIR / f).unlink()
    con, _ = ingest(root, time_from_git=False)
    with pytest.raises(LookupError, match="no multi-threaded"):
        analyze.thread_scaling(con, "i7-12700k")
    with pytest.raises(LookupError, match="machines: i7-12700k"):
        analyze.thread_scaling(con, "nope")
    con.close()


def test_scaling_does_not_pair_blas_vendors(tmp_path):
    # Every vendor's runs are labelled "blas": only the backend tells a
    # 1-thread OpenBLAS run from an 8-thread MKL one.
    root = _copy_fixture(tmp_path)
    _edit_sysinfo(root / RUN_DIR / "native-t1.json.sysinfo", blas_vendor="openblas")
    _edit_sysinfo(root / RUN_DIR / "native-t8.json.sysinfo", blas_vendor="mkl")
    con, _ = ingest(root, time_from_git=False)
    with pytest.raises(LookupError, match="1-thread run"):
        analyze.thread_scaling(con, "i7-12700k")
    con.close()


def test_scaling_does_not_cross_a_machine_state_change(tmp_path):
    root = _copy_fixture(tmp_path)
    _edit_sysinfo(root / RUN_DIR / "native-t1.json.sysinfo", turbo="disabled")
    con, _ = ingest(root, time_from_git=False)
    with pytest.raises(LookupError, match="1-thread run"):
        analyze.thread_scaling(con, "i7-12700k")
    con.close()


def test_roofline_intensity_and_filters(db):
    df = analyze.roofline(db, "i7-12700k", threads=1)
    assert (df["intensity"] == df["flops"] / df["bytes"]).all()
    assert (df["threads"] == 1).all() and df["run_id"].nunique() == 1
    only = analyze.roofline(db, "i7-12700k", threads=1, areas=["blas3"])
    assert set(only["area"]) == {"blas3"}


def test_roofline_leaves_out_failed_checks(tmp_path):
    import json

    root = _copy_fixture(tmp_path)
    path = root / RUN_DIR / "native-t1.json"
    doc = json.loads(path.read_text())
    gemm = next(r for r in doc["results"] if r["op"] == "gemm" and r["dtype"] == "f64")
    gemm["check"]["ok"] = False
    path.write_text(json.dumps(doc))
    con, _ = ingest(root, time_from_git=False)
    df = analyze.roofline(con, "i7-12700k", threads=1)
    assert not (df["entry_id"] == "blas3/gemm/f64/512x512x512").any()
    con.close()


def test_capability_reads_one_run_and_its_probes(db):
    df = analyze.capability(db, "i7-12700k", threads=8)
    assert len(df) == 1 and df.at[0, "threads"] == 8
    assert df.at[0, "hw_fma_isa"] == "AVX2+FMA"
    assert df.at[0, "cpu_logical_cores"] == 20


def test_the_latest_run_is_the_default(db):
    df = analyze.capability(db, "i7-12700k", threads=None)
    assert df.at[0, "run_id"] == analyze._runs(_con(db), "i7-12700k").at[0, "run_id"]


def _con(db):
    import duckdb

    return duckdb.connect(db, read_only=True)


def test_an_unknown_machine_lists_the_known_ones(db):
    with pytest.raises(LookupError, match="machines: i7-12700k"):
        analyze.capability(db, "nope")


def test_incomplete_runs_are_read_only_when_asked(db):
    assert analyze.rate_vs_size(db).empty
    df = analyze.rate_vs_size(db, tiers=INCOMPLETE)
    assert set(df["tier"]) == {"incomplete"}
    assert df["operation"].nunique() == 15
    assert (df["backend"] == "blis").all() and (df["samples"] == 1).all()


def test_rate_reduces_repeated_samples_to_their_median(tmp_path):
    root = _copy_fixture(tmp_path)
    csv = root / "blas_sweep_blis.csv"
    lines = csv.read_text().splitlines()
    row = next(line for line in lines if line.startswith("gemm,blis,65,"))
    fields = row.split(",")
    extra = []
    for gflops in ("1.0", "100.0"):
        f = list(fields)
        f[8] = gflops
        extra.append(",".join(f))
    csv.write_text("\n".join(lines + extra) + "\n")
    con, _ = ingest(root, time_from_git=False)
    df = analyze.rate_vs_size(con, operations=["gemm"], tiers=INCOMPLETE)
    point = df[df["size"] == 65].iloc[0]
    assert point["samples"] == 3
    assert point["value"] == pytest.approx(sorted([1.0, 100.0, float(fields[8])])[1])
    con.close()


def test_rate_keeps_two_runs_of_one_backend_apart(tmp_path):
    # Two runs of one backend were merged into one median per size: rates of
    # 10 and 30 plotted as 20, which neither run measured.
    root = _copy_fixture(tmp_path)
    src = root / "blas_sweep_blis.csv"
    other = root / "blas_sweep_blis_rebuilt.csv"
    lines = src.read_text().splitlines()
    rows = [lines[0]]
    for line in lines[1:]:
        f = line.split(",")
        f[8] = str(float(f[8]) * 3)
        rows.append(",".join(f))
    other.write_text("\n".join(rows) + "\n")
    (root / "blas_sweep_blis_rebuilt.csv.sysinfo").write_text(
        (root / "blas_sweep_blis.csv.sysinfo").read_text()
    )
    con, _ = ingest(root, time_from_git=False)
    df = analyze.rate_vs_size(con, operations=["gemm"], tiers=INCOMPLETE)
    con.close()
    fig = plot.rate_vs_size(df)
    lines = _lines(fig)
    assert len(lines) == 2
    for run in df["run_id"].unique():
        rows = df[df["run_id"] == run].sort_values("size")
        label = f"gemm (blis), run {run[:7]}"
        assert list(lines[label].get_ydata()) == list(rows["value"])


# --- style ------------------------------------------------------------------------------------


def test_palette_is_okabe_ito_without_yellow():
    assert "#F0E442" not in style.PALETTE and len(set(style.PALETTE)) == 7


def test_series_are_distinct_in_grayscale():
    styles = [style.series_style(i) for i in range(7)]
    assert len({s["marker"] for s in styles}) == 7
    assert len({str(s["linestyle"]) for s in styles}) == 7


def test_point_styles_are_distinct_past_the_palette():
    pairs = {tuple(plot._point_style(i).values()) for i in range(49)}
    assert len(pairs) == 49


def test_line_styles_never_differ_by_colour_alone():
    # Series i and i + 7 shared a marker and a dash, so in gray they were one.
    pairs = {
        (plot._line_style(i)["marker"], str(plot._line_style(i)["linestyle"])) for i in range(49)
    }
    assert len(pairs) == 49


def test_house_style_is_scoped():
    import matplotlib

    before = matplotlib.rcParams["font.size"]
    with style.house_style():
        assert matplotlib.rcParams["font.size"] == 8
        assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["font.size"] == before


def test_save_refuses_an_unknown_format(tmp_path):
    fig = plot.thread_scaling(_pinned("scaling"))
    with pytest.raises(ValueError, match="unsupported format"):
        style.save(fig.figure, tmp_path / "x", ["bmp"])


def test_state_class_without_placement():
    fields = {"governor": "performance", "cpu_affinity": "8", "bench_cpus": "0,2"}
    full = state_class(fields)
    bare = state_class(fields, placement=False)
    assert "bench_cpus" in full and "affinity" in full
    assert "bench_cpus" not in bare and "affinity" not in bare
    assert bare == state_class({**fields, "cpu_affinity": "1", "bench_cpus": "0"}, placement=False)


# --- command line ------------------------------------------------------------------------------


def test_cli_figure_writes_the_files(db, tmp_path, capsys):
    out = tmp_path / "card"
    assert main(["figure", "card", "--db", db, "--machine", "i7-12700k", "--out", str(out)]) == 0
    printed = capsys.readouterr().out.split()
    assert [Path(p).suffix for p in printed] == [".pdf", ".svg", ".png", ".csv"]
    assert all(Path(p).exists() for p in printed)


def test_cli_scaling_and_roofline(db, tmp_path):
    for kind in ("scaling", "roofline"):
        out = tmp_path / kind
        args = ["figure", kind, "--db", db, "--machine", "i7-12700k", "--out", str(out)]
        assert main([*args, "--format", "svg", "--no-data"]) == 0
        assert (tmp_path / f"{kind}.svg").exists()
        assert not (tmp_path / f"{kind}.csv").exists()


def test_cli_rate_points_at_include_incomplete(db, tmp_path, capsys):
    out = str(tmp_path / "rate")
    assert main(["figure", "rate", "--db", db, "--operation", "gemm", "--out", out]) == 1
    assert "--include-incomplete" in capsys.readouterr().err
    args = ["figure", "rate", "--db", db, "--operation", "gemm", "--out", out]
    assert main([*args, "--include-incomplete", "--format", "svg"]) == 0


def test_cli_needs_a_database(tmp_path, capsys):
    db = str(tmp_path / "none.duckdb")
    assert main(["figure", "card", "--db", db, "--machine", "m"]) == 2
    assert "ingest" in capsys.readouterr().err
    assert not Path(db).exists()


def test_cli_needs_a_machine(db):
    with pytest.raises(SystemExit):
        main(["figure", "card", "--db", db])
