"""Trends: series, confidence intervals and change points.

The input is the frame of :func:`mtl5.bench.analyze.trend`: one row per
(run, entry) from the ``trend_results`` view, so every row is a complete,
checked run. This module needs only pandas and numpy, so its rules can be
tested on synthetic series.

Series
    A series is one ``series_key``: the comparability key of :mod:`mtl5.bench`
    (machine, machine state, backend, build without the commit, threads,
    class, entry and its definition version). Along one entry's history the
    key can change, for a compiler upgrade, say. That is a **series break**:
    the two sides are different measurements, and nothing is compared across
    it. :func:`points` names the key fields that changed.
Points
    A point is one commit of one series: every run (session) at that commit.
    Its value is the geometric mean of the runs, because timing noise is
    multiplicative.
Noise and confidence intervals
    Run-to-run noise is the standard deviation of ``log(value)`` between
    runs of one commit, pooled over the series' commits. With fewer than two
    degrees of freedom from repeats, it is estimated instead from the
    differences between successive points (robustly, by their median), and
    ``source`` says so; failing that, a single pair of repeats gives one
    degree of freedom, and a correspondingly wide interval. A point's 95%
    confidence interval is its mean log value plus or minus
    ``t * s / sqrt(n)``, with ``t`` the 97.5% Student quantile for the
    noise's degrees of freedom.
Change points
    Within each unbroken stretch of a series, binary segmentation looks for
    the split between commits with the largest t statistic of the step in
    mean log value, using the pooled noise. A step is a change point when
    ``|t| >= k`` (default 4: a step the noise makes only by rare chance, with
    room for having searched every split) **and** the step is at least
    ``min_effect`` (default 2%) in size. Each side is then searched again.
    A step in the bad direction (slower, or a lower rate) is a regression.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "BREAK_FIELDS",
    "HIGHER_IS_BETTER",
    "t975",
    "noise",
    "points",
    "change_points",
    "summary",
]

#: The run fields of the series key that can change along an entry's history.
BREAK_FIELDS = ("backend", "build_config", "state_class", "class", "definition_version")

#: Metrics where a larger value is better; for the others (times) smaller is.
HIGHER_IS_BETTER = frozenset({"gflops", "gbytes_per_s", "roofline_pct"})

# The 97.5% quantile of Student's t, for a two-sided 95% interval, by degrees
# of freedom. Between entries it is interpolated in 1/df, which is accurate to
# the third decimal; beyond 120 it approaches the normal quantile.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}  # fmt: skip
_Z975 = 1.959964


def t975(df: float) -> float:
    """The 97.5% quantile of Student's t with ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError(f"degrees of freedom must be at least 1, not {df}")
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    if df > keys[-1]:
        lo_df, lo_t, hi_inv, hi_t = keys[-1], _T975[keys[-1]], 0.0, _Z975
        lo_inv = 1 / lo_df
    else:
        hi = next(k for k in keys if k > df)
        lo = max(k for k in keys if k < df)
        lo_inv, lo_t, hi_inv, hi_t = 1 / lo, _T975[lo], 1 / hi, _T975[hi]
    w = (1 / df - hi_inv) / (lo_inv - hi_inv)
    return hi_t + w * (lo_t - hi_t)


def _logs(df):
    import numpy as np

    values = df["value"].astype(float)
    if (values <= 0).any():
        raise ValueError("trend values must be positive: noise is measured on a log scale")
    return np.log(values)


def _commit_order(df):
    """The frame's commits in time order, by their first run."""
    first = df.groupby("git_commit")["run_timestamp"].min().sort_values(kind="stable")
    return list(first.index)


def noise(df):
    """Run-to-run noise per (entry, series): ``s`` on the log scale, ``dof``, ``source``.

    ``source`` is ``"repeats"`` when the runs at one commit give the degrees
    of freedom (two or more, or one when nothing better exists),
    ``"differences"`` when successive points had to stand in, and ``"none"``
    when neither is possible.
    """
    import numpy as np
    import pandas as pd

    rows = []
    d = df.assign(_log=_logs(df))
    for (entry, key), g in d.groupby(["entry_id", "series_key"], sort=True):
        within = g.groupby("git_commit")["_log"]
        dof = int((within.count() - 1).clip(lower=0).sum())
        ss = float(within.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum())
        if dof >= 2:
            rows.append((entry, key, math.sqrt(ss / dof), dof, "repeats"))
            continue
        means = within.mean()[[c for c in _commit_order(g)]]
        diffs = np.diff(means.to_numpy())
        if len(diffs) >= 2:
            # Median absolute difference: robust to the few steps a trend has.
            # A difference of two noisy means has sqrt(2) times their noise.
            s = 1.4826 * float(np.median(np.abs(diffs))) / math.sqrt(2)
            rows.append((entry, key, s, len(diffs) - 1, "differences"))
        elif dof == 1:
            # One commit's pair of sessions: an estimate, with a wide interval.
            rows.append((entry, key, math.sqrt(ss), 1, "repeats"))
        else:
            rows.append((entry, key, float("nan"), 0, "none"))
    return pd.DataFrame(rows, columns=["entry_id", "series_key", "s", "dof", "source"])


def points(df):
    """One row per (entry, series, commit), in time order per entry.

    ``value`` is the geometric mean of the commit's runs, ``ci_lo`` and
    ``ci_hi`` its 95% confidence interval (missing when the noise is
    unknown), ``n`` the run count, ``segment`` the index of the unbroken
    stretch of the entry's history it belongs to, and ``break_fields`` the
    key fields that changed at the first point of a new stretch.
    """
    import numpy as np
    import pandas as pd

    nz = noise(df).set_index(["entry_id", "series_key"])
    d = df.assign(_log=_logs(df))
    rows = []
    for entry, e in d.groupby("entry_id", sort=True):
        order = {c: i for i, c in enumerate(_commit_order(e))}
        grouped = e.groupby(["series_key", "git_commit"], sort=False)
        pts = []
        for (key, commit), g in grouped:
            pts.append((order[commit], g["run_timestamp"].min(), key, commit, g))
        pts.sort(key=lambda p: (p[1], p[0], p[2]))
        segment, prev = -1, None
        for _, when, key, commit, g in pts:
            first = g.iloc[0]
            if prev is None or key != prev[0]:
                segment += 1
                changed = (
                    []
                    if prev is None
                    else [f for f in BREAK_FIELDS if f in g and str(first[f]) != str(prev[1][f])]
                )
            else:
                changed = []
            n = len(g)
            mean = float(g["_log"].mean())
            s, dof = nz.loc[(entry, key), ["s", "dof"]]
            if dof >= 1 and not math.isnan(s):
                half = t975(dof) * s / math.sqrt(n)
                lo, hi = math.exp(mean - half), math.exp(mean + half)
            else:
                lo = hi = float("nan")
            rows.append(
                {
                    "entry_id": entry,
                    "series_key": key,
                    "segment": segment,
                    "git_commit": commit,
                    "run_timestamp": when,
                    "n": n,
                    "value": math.exp(mean),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "break_fields": ", ".join(changed),
                    "machine": first.get("machine"),
                }
            )
            prev = (key, first)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["n"] = out["n"].astype(np.int64)
    return out


def _split(logs: Sequence, s: float, k: float, min_effect: float, lo: int, hi: int, found: list):
    """Binary segmentation of points ``lo..hi-1``; each point is a list of logs."""
    best = None
    for c in range(lo + 1, hi):
        left = [x for p in logs[lo:c] for x in p]
        right = [x for p in logs[c:hi] for x in p]
        step = sum(right) / len(right) - sum(left) / len(left)
        se = s * math.sqrt(1 / len(left) + 1 / len(right))
        t = step / se if se > 0 else math.inf * (1 if step > 0 else -1 if step < 0 else 0)
        if t != t:  # 0 / 0: no step and no noise
            continue
        if best is None or abs(t) > abs(best[1]):
            best = (c, t, step)
    if best is None:
        return
    c, t, step = best
    if abs(t) >= k and abs(step) >= math.log1p(min_effect):
        found.append((c, t, step))
        _split(logs, s, k, min_effect, lo, c, found)
        _split(logs, s, k, min_effect, c, hi, found)


def change_points(df, *, k: float = 4.0, min_effect: float = 0.02):
    """The change points of every series, one row each.

    ``before`` and ``after`` are the commits either side of the step,
    ``ratio`` the after/before ratio of the geometric means of the two
    stretches the step separates, ``t`` the step's t statistic, and
    ``kind`` ``"regression"`` or ``"improvement"`` by the metric's direction
    (:data:`HIGHER_IS_BETTER`). A series whose noise is unknown has none.
    """
    import pandas as pd

    if k <= 0 or min_effect < 0:
        raise ValueError("k must be positive and min_effect non-negative")
    metric = _one(df, "metric")
    higher = metric in HIGHER_IS_BETTER
    nz = noise(df).set_index(["entry_id", "series_key"])
    pts = points(df)
    d = df.assign(_log=_logs(df))
    rows = []
    for (entry, segment), seg in pts.groupby(["entry_id", "segment"], sort=True):
        key = seg["series_key"].iloc[0]
        s = float(nz.loc[(entry, key), "s"])
        if math.isnan(s) or len(seg) < 2:
            continue
        runs = d[(d["entry_id"] == entry) & (d["series_key"] == key)]
        logs = [list(runs.loc[runs["git_commit"] == c, "_log"]) for c in seg["git_commit"]]
        found: list = []
        _split(logs, s, k, min_effect, 0, len(logs), found)
        commits = list(seg["git_commit"])
        for c, t, step in sorted(found):
            ratio = math.exp(step)
            worse = ratio < 1 if higher else ratio > 1
            rows.append(
                {
                    "entry_id": entry,
                    "series_key": key,
                    "segment": segment,
                    "before": commits[c - 1],
                    "after": commits[c],
                    "index": int(seg.index[c] - seg.index[0]),
                    "ratio": ratio,
                    "t": t,
                    "kind": "regression" if worse else "improvement",
                }
            )
    cols = ["entry_id", "series_key", "segment", "before", "after", "index", "ratio", "t", "kind"]
    return pd.DataFrame(rows, columns=cols)


def _one(df, col: str):
    values = sorted(set(df[col].dropna())) if col in df else []
    if len(values) != 1:
        raise ValueError(f"a trend has one {col}; the frame has {values or 'none'}")
    return values[0]


def summary(df):
    """One row per entry: its points, series, breaks and noise, for a report."""
    import pandas as pd

    pts = points(df)
    nz = noise(df)
    rows = []
    for entry, p in pts.groupby("entry_id", sort=True):
        breaks = [
            f"{str(r.git_commit)[:7]} ({r.break_fields or 'series key'})"
            for r in p.itertuples()
            if r.segment > 0 and r.Index == p.index[p["segment"] == r.segment][0]
        ]
        n = nz[nz["entry_id"] == entry]
        latest = n[n["series_key"] == p["series_key"].iloc[-1]].iloc[0]
        rows.append(
            {
                "entry_id": entry,
                "commits": len(p),
                "runs": int(p["n"].sum()),
                "series": int(p["segment"].max()) + 1,
                "breaks": "; ".join(breaks),
                "noise_pct": (math.exp(latest["s"]) - 1) * 100 if latest["dof"] else None,
                "noise_source": latest["source"],
                "noise_dof": int(latest["dof"]),
            }
        )
    return pd.DataFrame(rows)
