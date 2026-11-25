"""Tests for trends: series, confidence intervals and change points (#121 P5).

The acceptance criteria of #121 for trends are tested here:

- a trend query returns only comparable runs;
- it shows confidence intervals from repeated runs;
- it marks series breaks when a key field changes;
- change-point detection flags a regression that exceeds run-to-run noise,
  and does not flag one within it (on synthetic series, with fixed seeds).

The database tests build a history from MTL5's committed i7-12700k runs
(``tests/data/bench_machines``): copies at further commits and dates, with
their rates scaled to make a known step.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from mtl5.bench import analyze, ingest, plot, trends  # noqa: E402
from mtl5.bench.analyze import _matches  # noqa: E402
from mtl5.bench.cli import main  # noqa: E402

DATA = Path(__file__).parent / "data" / "bench_machines"
I7 = "i7-12700k/mtl5_bench"
SESSIONS = ("20261111T190041Z", "20261112T004852Z")  # both at bfa1c28, 8 threads


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """Run each test in its own directory: the CLI writes to the current one."""
    monkeypatch.chdir(tmp_path)


# --- synthetic series ------------------------------------------------------------------


def synth(
    steps=None,
    *,
    noise=0.05,
    commits=10,
    reps=3,
    seed=0,
    metric="gflops",
    keys=None,
    build=None,
):
    """A synthetic history: ``steps`` maps a commit index to a factor on the level."""
    steps = steps or {}
    rng = np.random.default_rng(seed)
    rows, level = [], 100.0
    for c in range(commits):
        level *= steps.get(c, 1.0)
        for r in range(reps):
            rows.append(
                {
                    "run_id": f"r{c:02d}{r}",
                    "machine": "m",
                    "label": "native",
                    "threads": 1,
                    "backend": "native",
                    "build_config": build(c) if build else "b",
                    "state_class": "s",
                    "class": "medium",
                    "definition_version": 0,
                    "series_key": keys(c) if keys else "K",
                    "git_commit": f"{c:02d}" + "0" * 10,
                    "run_timestamp": pd.Timestamp("2026-10-01") + pd.Timedelta(days=c, minutes=r),
                    "tier": "complete",
                    "entry_id": "blas3/gemm/f64/n",
                    "value": level * math.exp(rng.normal(0, noise)),
                    "metric": metric,
                    "unit": "GFLOP/s" if metric == "gflops" else "s",
                }
            )
    return pd.DataFrame(rows)


def flagged(df, **kw) -> int:
    return len(trends.change_points(df, **kw))


# --- the acceptance criteria: change points against noise ------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_a_regression_beyond_the_noise_is_flagged(seed):
    cps = trends.change_points(synth({6: 0.80}, seed=seed))
    assert len(cps) == 1
    cp = cps.iloc[0]
    assert (cp["before"], cp["after"]) == ("05" + "0" * 10, "06" + "0" * 10)
    assert cp["kind"] == "regression" and cp["ratio"] == pytest.approx(0.8, rel=0.05)


@pytest.mark.parametrize("seed", range(5))
def test_a_step_within_the_noise_is_not_flagged(seed):
    assert flagged(synth({6: 0.98}, seed=seed)) == 0


def test_detection_rates_over_many_series():
    # 5% run-to-run noise, 10 commits of 3 runs each, 100 seeds per case.
    rate = {
        step: sum(flagged(synth({6: step} if step != 1 else {}, seed=s)) > 0 for s in range(100))
        for step in (1.0, 0.98, 0.80)
    }
    assert rate[1.0] <= 2  # false alarms on pure noise
    assert rate[0.98] <= 10  # a 2% step hides in 5% noise
    assert rate[0.80] == 100  # a 20% step never does


def test_direction_follows_the_metric():
    # A time that goes up is a regression; a rate that goes up is not.
    slower = trends.change_points(synth({6: 1.25}, metric="median_s"))
    faster = trends.change_points(synth({6: 1.25}, metric="gflops"))
    assert list(slower["kind"]) == ["regression"]
    assert list(faster["kind"]) == ["improvement"]


def test_several_change_points():
    cps = trends.change_points(synth({3: 0.8, 7: 1.3}, commits=11, seed=2))
    assert list(cps["after"].str[:2]) == ["03", "07"]
    assert list(cps["kind"]) == ["regression", "improvement"]


def _levels(levels, *, reps=2, noise=0.0, seed=0):
    """A series at the given level per commit, with ``reps`` runs each."""
    steps, prev = {}, 100.0
    for c, level in enumerate(levels):
        if level != prev:
            steps[c] = level / prev
        prev = level
    return synth(steps, commits=len(levels), reps=reps, noise=noise, seed=seed)


def test_a_reported_step_is_between_adjacent_stretches():
    # The 80 -> 120 step was found first, against the whole 100, 80 stretch,
    # and reported as +34% instead of the +50% the level lines show.
    cps = trends.change_points(_levels([100, 100, 80, 80, 120, 120], noise=0.001, seed=1))
    assert list(cps["ratio"]) == pytest.approx([0.8, 1.5], rel=0.01)
    assert list(cps["kind"]) == ["regression", "improvement"]


def test_identical_repeats_still_place_the_step_right():
    # With zero noise every t was infinite and the first split won, so one
    # step became two change points, the first at the wrong commit.
    cps = trends.change_points(_levels([100, 100, 80, 80]))
    assert len(cps) == 1
    assert cps.iloc[0]["after"][:2] == "02" and cps.iloc[0]["ratio"] == pytest.approx(0.8)
    assert cps.iloc[0]["t"] == -math.inf
    assert len(trends.change_points(_levels([100, 100, 100]))) == 0


def test_a_rerun_after_another_series_is_a_new_point():
    # Commit 00 ran in series K1, then 01 in K2, then 00 again in K1. The
    # rerun was merged into the first point, and the series never returned.
    df = synth(commits=2, reps=2, keys=lambda c: f"K{c + 1}", build=lambda c: f"b{c}")
    rerun = df[df["git_commit"].str.startswith("00")].copy()
    rerun["run_timestamp"] += pd.Timedelta(days=5)
    rerun["run_id"] += "x"
    pts = trends.points(pd.concat([df, rerun], ignore_index=True))
    assert list(pts["git_commit"].str[:2]) == ["00", "01", "00"]
    assert list(pts["segment"]) == [0, 1, 2] and list(pts["n"]) == [2, 2, 2]
    assert list(pts["break_fields"]) == ["", "build_config", "build_config"]


def test_min_effect_holds_back_a_precise_but_tiny_step():
    # 0.1% noise makes a 1% step overwhelming statistically, yet small.
    df = synth({6: 0.99}, noise=0.001)
    assert flagged(df) == 0
    assert flagged(df, min_effect=0.005) == 1


def test_detection_parameters_are_checked():
    with pytest.raises(ValueError, match="k must be positive"):
        trends.change_points(synth(), k=0)


# --- the acceptance criteria: series breaks --------------------------------------------


def test_a_series_break_is_marked_with_the_fields_that_changed():
    df = synth(
        keys=lambda c: "K1" if c < 5 else "K2", build=lambda c: "gcc15" if c < 5 else "gcc16"
    )
    pts = trends.points(df)
    assert list(pts["segment"]) == [0] * 5 + [1] * 5
    assert pts.loc[5, "break_fields"] == "build_config"
    assert (pts.drop(index=5)["break_fields"] == "").all()


def test_nothing_is_compared_across_a_series_break():
    # A 30% drop at the break is a different build, not a regression.
    df = synth(
        {5: 0.7}, keys=lambda c: "K1" if c < 5 else "K2", build=lambda c: "a" if c < 5 else "b"
    )
    assert flagged(df) == 0


def test_change_points_are_found_within_each_series():
    df = synth(
        {2: 0.8, 5: 0.5, 8: 0.8},
        commits=11,
        keys=lambda c: "K1" if c < 5 else "K2",
        build=lambda c: "a" if c < 5 else "b",
    )
    cps = trends.change_points(df)
    assert list(cps["segment"]) == [0, 1]
    assert list(cps["after"].str[:2]) == ["02", "08"]


# --- the acceptance criteria: confidence intervals -------------------------------------


def test_intervals_come_from_the_pooled_repeats():
    df = synth(noise=0.05, commits=4, reps=3, seed=4)
    logs = np.log(df["value"])
    within = df.assign(_l=logs).groupby("git_commit")["_l"]
    ss = sum(((g - g.mean()) ** 2).sum() for _, g in within)
    s = math.sqrt(ss / 8)  # 4 commits x (3 - 1)
    nz = trends.noise(df).iloc[0]
    assert (nz["source"], nz["dof"]) == ("repeats", 8)
    assert nz["s"] == pytest.approx(s)
    p = trends.points(df).iloc[0]
    mean = within.mean().iloc[0]
    half = trends.t975(8) * s / math.sqrt(3)
    assert p["value"] == pytest.approx(math.exp(mean))
    assert (p["ci_lo"], p["ci_hi"]) == pytest.approx((math.exp(mean - half), math.exp(mean + half)))


def test_intervals_cover_the_true_level():
    # About 95% of 95% intervals hold the level the noise was drawn around.
    hits = total = 0
    for seed in range(40):
        p = trends.points(synth(noise=0.05, commits=5, reps=3, seed=seed))
        hits += int(((p["ci_lo"] <= 100) & (p["ci_hi"] >= 100)).sum())
        total += len(p)
    assert 0.88 <= hits / total <= 0.995


def test_without_repeats_the_noise_comes_from_successive_points():
    nz = trends.noise(synth(reps=1, commits=8)).iloc[0]
    assert nz["source"] == "differences" and nz["dof"] == 6
    assert 0.02 < nz["s"] < 0.1


def test_one_pair_of_repeats_gives_a_wide_interval():
    df = synth(reps=2, commits=1)
    nz = trends.noise(df).iloc[0]
    assert (nz["source"], nz["dof"]) == ("repeats", 1)
    p = trends.points(df).iloc[0]
    # t = 12.706 with one degree of freedom, over sqrt(2) runs, either side.
    width = 2 * 12.706 * nz["s"] / math.sqrt(2)
    assert math.log(p["ci_hi"] / p["ci_lo"]) == pytest.approx(width)


def test_a_single_run_has_no_interval():
    df = synth(reps=1, commits=1)
    assert trends.noise(df).iloc[0]["source"] == "none"
    p = trends.points(df).iloc[0]
    assert math.isnan(p["ci_lo"]) and math.isnan(p["ci_hi"])
    assert flagged(df) == 0


def test_t975_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in [*range(1, 200), 250, 500, 1000, 10**6]:
        assert trends.t975(df) == pytest.approx(stats.t.ppf(0.975, df), abs=1e-3)
    with pytest.raises(ValueError):
        trends.t975(0)


def test_values_must_be_positive():
    df = synth(commits=2)
    df.loc[0, "value"] = 0
    with pytest.raises(ValueError, match="positive"):
        trends.points(df)


def test_a_trend_has_one_metric():
    df = synth(commits=4)
    df.loc[0, "metric"] = "median_s"
    with pytest.raises(ValueError, match="one metric"):
        trends.change_points(df)


# --- entries named without their area --------------------------------------------------


def test_entries_match_with_or_without_the_area():
    assert _matches("blas3/gemm/f64/512x512x512", ["gemm/f64"])
    assert _matches("blas3/gemm/f64/512x512x512", ["blas3/gemm"])
    assert not _matches("blas3/gemmt/f64/512x512x512", ["gemm"])
    assert _matches("integer/gemm/i8>i32/512^3", ["gemm"])


# --- the database: only comparable runs ------------------------------------------------


def _scale(doc: dict, factor: float) -> dict:
    for r in doc["results"]:
        for key in ("gflops", "gbytes_per_s"):
            if r.get(key) is not None:
                r[key] *= factor
        for key in ("median_s", "min_s"):
            if r.get(key) is not None:
                r[key] /= factor
    return doc


def _session(root: Path, src: str, stamp: str, commit: str, factor: float, **sidecar) -> None:
    out = root / I7 / stamp
    out.mkdir(parents=True)
    doc = json.loads((DATA / I7 / src / "native-t8.json").read_text())
    (out / "native-t8.json").write_text(json.dumps(_scale(doc, factor)))
    lines = []
    fields = {"git_commit": commit, "tree_git_commit": commit, **sidecar}
    for line in (DATA / I7 / src / "native-t8.json.sysinfo").read_text().splitlines():
        key = line.split("=", 1)[0]
        lines.append(f"{key}={fields[key]}" if key in fields else line)
    (out / "native-t8.json.sysinfo").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def history(tmp_path_factory) -> str:
    """i7 GEMM over five commits, two sessions each.

    The fourth commit is 25% slower, and the fifth is built with another
    compiler version (a series break). One more session is dirty, so it is
    incomplete and must not appear.
    """
    root = tmp_path_factory.mktemp("history") / "data"
    plan = [
        ("aaaaaaaaaaaa", 1.00, {}),
        ("bbbbbbbbbbbb", 1.00, {}),
        ("cccccccccccc", 1.00, {}),
        ("dddddddddddd", 0.75, {}),
        ("eeeeeeeeeeee", 0.75, {"compiler_version": "16.1.0"}),
    ]
    for i, (commit, factor, extra) in enumerate(plan):
        for j, src in enumerate(SESSIONS):
            stamp = f"202611{10 + i:02d}T0{j}0000Z"
            _session(root, src, stamp, commit, factor, **extra)
    _session(root, SESSIONS[0], "20261120T000000Z", "ffffffffffff", 2.0, git_dirty="1")
    path = tmp_path_factory.mktemp("db") / "bench.duckdb"
    con, _ = ingest(root, path, time_from_git=False)
    con.close()
    return str(path)


def test_a_trend_reads_only_complete_checked_runs(history):
    df = analyze.trend(history, "i7-12700k", ["gemm/f64"], threads=8)
    assert set(df["tier"]) == {"complete"}
    assert "ffffffffffff" not in set(df["git_commit"])  # the dirty session
    assert len(df) == 10 and df["entry_id"].nunique() == 1


def test_the_history_has_its_break_and_its_regression(history):
    df = analyze.trend(history, "i7-12700k", ["gemm/f64"], threads=8)
    pts = trends.points(df)
    assert list(pts["git_commit"].str[0]) == list("abcde")
    assert list(pts["segment"]) == [0, 0, 0, 0, 1]
    assert pts.iloc[4]["break_fields"] == "build_config"
    cps = trends.change_points(df)
    assert list(zip(cps["before"].str[0], cps["after"].str[0], cps["kind"])) == [
        ("c", "d", "regression")
    ]
    assert cps.iloc[0]["ratio"] == pytest.approx(0.75, rel=0.02)


def test_trend_arguments_are_checked(history):
    with pytest.raises(ValueError, match="unknown metric"):
        analyze.trend(history, "i7-12700k", ["gemm"], metric="speedup")
    with pytest.raises(ValueError, match="fixed"):
        analyze.trend(history, "i7-12700k", ["gemm"], threads="max")
    with pytest.raises(LookupError, match="machines: i7-12700k"):
        analyze.trend(history, "nope", ["gemm"])


# --- the figure ------------------------------------------------------------------------


def test_the_figure_marks_breaks_and_change_points(history):
    df = analyze.trend(history, "i7-12700k", ["gemm/f64"], threads=8)
    fig = plot.trend(df)
    texts = [t.get_text() for t in fig.figure.axes[0].texts]
    assert "series break: build_config" in texts
    assert any(t.endswith("% regression") and t.startswith("-25") for t in texts)
    assert plot.provenance(df) in [t.get_text() for t in fig.figure.texts]


def test_the_figure_shows_every_run_and_each_commits_interval():
    df = synth(commits=4, reps=3)
    ax = plot.trend(df).figure.axes[0]
    runs = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == len(df)]
    assert len(runs) == 1 and sorted(runs[0].get_ydata()) == sorted(df["value"])
    (bars,) = ax.containers
    pts = trends.points(df)
    assert list(bars.lines[0].get_ydata()) == pytest.approx(list(pts["value"]))


def test_a_trend_figure_redraws_from_its_csv_with_its_thresholds(tmp_path):
    df = synth({6: 0.9}, seed=1)
    fig = plot.trend(df, k=3.0, min_effect=0.05)
    fig.save(tmp_path / "t", ["svg"])
    again = plot.from_csv(tmp_path / "t.csv", "trend")
    assert again._render("svg") == fig._render("svg")


def test_too_many_entries_for_one_figure():
    frames = []
    for i in range(7):
        d = synth(commits=2)
        d["entry_id"] = f"a/op{i}/f64/n"
        frames.append(d)
    with pytest.raises(ValueError, match="more than a legible figure holds"):
        plot.trend(pd.concat(frames))


# --- command line ----------------------------------------------------------------------


def test_cli_trend_reports_and_draws(history, tmp_path, capsys):
    args = ["trend", "gemm/f64", "--machine", "i7-12700k", "--threads", "8", "--db", history]
    assert main([*args, "--out", str(tmp_path / "t"), "--format", "svg"]) == 0
    out = capsys.readouterr().out
    assert "| blas3/gemm/f64/512x512x512 | 5 | 10 | 2 |" in out
    assert "| cccccc" in out and "regression" in out
    assert (tmp_path / "t.svg").exists() and (tmp_path / "t.csv").exists()


def test_cli_trend_fails_on_a_regression(history, capsys):
    args = ["trend", "gemm/f64", "--machine", "i7-12700k", "--threads", "8", "--db", history]
    assert main([*args, "--no-figure", "--fail-on-regression"]) == 3
    assert main([*args, "--no-figure", "--fail-on-regression", "--min-effect", "0.5"]) == 0
    assert "no change points" in capsys.readouterr().out
    assert list(Path.cwd().iterdir()) == []  # --no-figure writes nothing


def test_cli_trend_with_no_matching_entry(history, capsys):
    args = ["trend", "nothing/here", "--machine", "i7-12700k", "--threads", "8", "--db", history]
    assert main(args) == 1
    assert "no complete results" in capsys.readouterr().err
