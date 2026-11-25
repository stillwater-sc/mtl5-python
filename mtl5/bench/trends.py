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


def _assign(df):
    """The runs in the order they happened, each with its segment and point.

    Per entry, runs are ordered by time (then run id). A run whose series
    key differs from the previous run's starts a new segment, even if its
    commit already has a point earlier on; a point is a stretch of
    consecutive runs of one segment at one commit. So a rerun of an older
    commit after another series ran joins neither that series nor its own
    earlier point.
    """
    d = df.assign(_log=_logs(df), _order=range(len(df)))
    d = d.sort_values(["entry_id", "run_timestamp", "run_id", "_order"], kind="stable")
    segments, pts = [], []
    prev = None
    seg = pt = -1
    for entry, key, commit in zip(d["entry_id"], d["series_key"], d["git_commit"]):
        if prev is None or entry != prev[0]:
            seg = pt = -1
        if prev is None or entry != prev[0] or key != prev[1]:
            seg += 1
            pt += 1
        elif commit != prev[2]:
            pt += 1
        segments.append(seg)
        pts.append(pt)
        prev = (entry, key, commit)
    return d.assign(_segment=segments, _point=pts)


def noise(df):
    """Run-to-run noise per (entry, series): ``s`` on the log scale, ``dof``, ``source``.

    ``source`` is ``"repeats"`` when the runs at one point give the degrees
    of freedom (two or more, or one when nothing better exists),
    ``"differences"`` when successive points had to stand in, and ``"none"``
    when neither is possible.
    """
    import numpy as np
    import pandas as pd

    rows = []
    d = _assign(df)
    for (entry, key), g in d.groupby(["entry_id", "series_key"], sort=True):
        within = g.groupby("_point")["_log"]
        dof = int((within.count() - 1).clip(lower=0).sum())
        ss = float(within.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum())
        if dof >= 2:
            rows.append((entry, key, math.sqrt(ss / dof), dof, "repeats"))
            continue
        diffs = np.diff(within.mean().sort_index().to_numpy())
        if len(diffs) >= 2:
            # Median absolute difference: robust to the few steps a trend has.
            # A difference of two noisy means has sqrt(2) times their noise.
            s = 1.4826 * float(np.median(np.abs(diffs))) / math.sqrt(2)
            rows.append((entry, key, s, len(diffs) - 1, "differences"))
        elif dof == 1:
            # One point's pair of sessions: an estimate, with a wide interval.
            rows.append((entry, key, math.sqrt(ss), 1, "repeats"))
        else:
            rows.append((entry, key, float("nan"), 0, "none"))
    return pd.DataFrame(rows, columns=["entry_id", "series_key", "s", "dof", "source"])


def points(df):
    """One row per point, in the order the runs happened, per entry.

    A point is consecutive runs of one series at one commit. ``value`` is
    the geometric mean of its runs, ``ci_lo`` and ``ci_hi`` its 95%
    confidence interval (missing when the noise is unknown), ``n`` the run
    count, ``segment`` the index of the unbroken stretch of the entry's
    history it belongs to, ``point`` its index in the entry's history, and
    ``break_fields`` the key fields that changed at the first point of a new
    stretch.
    """
    import numpy as np
    import pandas as pd

    nz = noise(df).set_index(["entry_id", "series_key"])
    d = _assign(df)
    rows = []
    for entry, e in d.groupby("entry_id", sort=True):
        prev = None
        for point, g in e.groupby("_point", sort=True):
            first = g.iloc[0]
            key = first["series_key"]
            if prev is not None and first["_segment"] != prev["_segment"]:
                changed = [f for f in BREAK_FIELDS if f in g and str(first[f]) != str(prev[f])]
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
                    "segment": int(first["_segment"]),
                    "point": int(point),
                    "git_commit": first["git_commit"],
                    "run_timestamp": g["run_timestamp"].min(),
                    "n": n,
                    "value": math.exp(mean),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "break_fields": ", ".join(changed),
                    "machine": first.get("machine"),
                }
            )
            prev = first
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["n"] = out["n"].astype(np.int64)
    return out


def _split(logs: Sequence, s: float, k: float, min_effect: float, lo: int, hi: int, cuts: list):
    """Binary segmentation of points ``lo..hi-1``; each point is a list of logs.

    Candidates are ranked by the step over its scale, ``|step| / sqrt(1/nl +
    1/nr)``: the same order as ``|t|`` when the noise is positive, and still
    an order when it is zero (identical repeats make every ``t`` infinite).
    """
    best = None
    for c in range(lo + 1, hi):
        left = [x for p in logs[lo:c] for x in p]
        right = [x for p in logs[c:hi] for x in p]
        step = sum(right) / len(right) - sum(left) / len(left)
        scale = math.sqrt(1 / len(left) + 1 / len(right))
        rank = abs(step) / scale
        if best is None or rank > best[1]:
            best = (c, rank, step, scale)
    if best is None:
        return
    c, rank, step, scale = best
    if step == 0:
        return
    t = math.inf if s == 0 else rank / s
    if t >= k and abs(step) >= math.log1p(min_effect):
        cuts.append(c)
        _split(logs, s, k, min_effect, lo, c, cuts)
        _split(logs, s, k, min_effect, c, hi, cuts)


def change_points(df, *, k: float = 4.0, min_effect: float = 0.02):
    """The change points of every series, one row each.

    ``before`` and ``after`` are the commits either side of the step.
    ``ratio`` and ``t`` describe the step between the two **adjacent**
    stretches the change point separates, after all change points are
    found: the after/before ratio of their geometric means, and its t
    statistic (infinite when the noise is zero). ``index`` is the position of
    the ``after`` point within its segment. ``kind`` is ``"regression"`` or
    ``"improvement"`` by the metric's direction (:data:`HIGHER_IS_BETTER`).
    A series whose noise is unknown has none.
    """
    import pandas as pd

    if k <= 0 or min_effect < 0:
        raise ValueError("k must be positive and min_effect non-negative")
    metric = _one(df, "metric")
    higher = metric in HIGHER_IS_BETTER
    nz = noise(df).set_index(["entry_id", "series_key"])
    d = _assign(df)
    rows = []
    for (entry, segment), seg in d.groupby(["entry_id", "_segment"], sort=True):
        key = seg["series_key"].iloc[0]
        s = float(nz.loc[(entry, key), "s"])
        groups = list(seg.groupby("_point", sort=True))
        if math.isnan(s) or len(groups) < 2:
            continue
        logs = [list(g["_log"]) for _, g in groups]
        commits = [g["git_commit"].iloc[0] for _, g in groups]
        cuts: list = []
        _split(logs, s, k, min_effect, 0, len(logs), cuts)
        cuts.sort()
        bounds = [0, *cuts, len(logs)]
        for i, c in enumerate(cuts):
            left = [x for p in logs[bounds[i] : c] for x in p]
            right = [x for p in logs[c : bounds[i + 2]] for x in p]
            step = sum(right) / len(right) - sum(left) / len(left)
            scale = math.sqrt(1 / len(left) + 1 / len(right))
            t = math.copysign(math.inf, step) if s == 0 else step / (s * scale)
            ratio = math.exp(step)
            worse = ratio < 1 if higher else ratio > 1
            rows.append(
                {
                    "entry_id": entry,
                    "series_key": key,
                    "segment": int(segment),
                    "before": commits[c - 1],
                    "after": commits[c],
                    "index": c,
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
