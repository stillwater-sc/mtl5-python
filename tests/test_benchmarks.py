"""Smoke tests for benchmarks/bench_blas.py.

The harness is not run in anger by CI — a timing sweep on a shared runner is
noise. What CI must catch is the harness going stale: a renamed binding, a
dtype dropped from the list, a kernel whose entry point moved. So these run the
real code path at the smallest possible sizes and assert on structure, never on
timing.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import mtl5

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "bench_blas.py"

pytestmark = pytest.mark.skipif(not BENCH.exists(), reason="benchmarks/ not present")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_blas", BENCH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the harness uses `from __future__ import
    # annotations`, so @dataclass resolves its field annotations by looking the
    # defining module up in sys.modules. Loading it out-of-band without this
    # leaves that lookup returning None and every dataclass raises.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_every_default_dtype_is_bound(bench):
    """The sweep list must not drift from what the extension actually has."""
    available = set(mtl5.dtypes())
    missing = [d for d in bench.DEFAULT_DTYPES if d not in available]
    assert not missing, f"benchmark lists unbound dtypes: {missing}"


def test_aliases_resolve_to_bound_dtypes(bench):
    available = set(mtl5.dtypes())
    for alias, canonical in bench.DTYPE_ALIASES.items():
        assert canonical in available, f"alias {alias} -> {canonical} is not bound"


def test_every_kernel_has_a_flop_count_and_sizes(bench):
    for k in bench.KERNELS:
        assert k in bench.FLOPS, f"{k} has no FLOP count"
        assert k in bench.DEFAULT_SIZES, f"{k} has no default sizes"
        assert k in bench.QUICK_SIZES, f"{k} has no quick sizes"
        assert bench.FLOPS[k](8) > 0


def test_operands_and_call_resolve_for_every_kernel_and_dtype(bench):
    """The part most likely to rot: entry points differ per kernel AND per
    dtype family, so exercise every combination at n=4 rather than trusting
    that the dispatch in _make_call still matches the bindings."""
    for kernel in bench.KERNELS:
        for dtype in bench.DEFAULT_DTYPES:
            operands = bench._make_operands(kernel, dtype, 4)
            fn = bench._make_call(kernel, dtype, operands)
            # Every kernel now resolves for every swept dtype: lu and qr gained
            # Universal instantiations in #69, so the sweep has no holes left.
            assert fn() is not None, f"{kernel}/{dtype} produced nothing"


def test_end_to_end_quick_run(bench, tmp_path):
    out = tmp_path / "bench.json"
    rc = bench.main(
        [
            "--quick",
            "--dtypes",
            "f64",
            "posit32",
            "--kernels",
            "dot",
            "gemm",
            "--min-time",
            "0.001",
            "--repeat",
            "1",
            "--json",
            str(out),
        ]
    )
    assert rc == 0

    report = json.loads(out.read_text())
    assert report["meta"]["baseline_dtype"] == "f64"
    assert "build_info" in report["meta"]
    assert set(report["meta"]["call_floor_s"]) == {"dot", "gemm"}

    rows = report["results"]
    assert rows, "no results produced"
    assert {r["dtype"] for r in rows} == {"f64", "posit32"}

    baseline_rows = [r for r in rows if r["dtype"] == "f64" and r["status"] == "ok"]
    assert baseline_rows, "baseline produced no successful row"
    for r in baseline_rows:
        assert r["slowdown_vs_baseline"] == pytest.approx(1.0)

    # Structure only, deliberately: that the ratio was computed and attached at
    # all is what can break (a missed baseline key leaves it None). Asserting
    # posit32 > 1.0 would be true by a factor of ~10^3 and so not flaky in
    # practice, but it would make a correctness test depend on host
    # performance, and this file promises not to do that.
    for r in rows:
        if r["status"] == "ok":
            assert r["slowdown_vs_baseline"] is not None
            assert r["slowdown_vs_baseline"] > 0.0
            assert r["seconds_per_op"] > 0.0


def test_unavailable_is_reported_not_raised(bench, monkeypatch):
    """A kernel a dtype lacks must land as a row, not an exception — a sweep
    that aborts on the first gap is useless for exactly the types we care about.

    Driven through an injected TypeError rather than a real gap: since #69 gave
    lu and qr their Universal instantiations, no swept combination is
    unavailable any more. The guarantee still has to hold for the next dtype
    added ahead of its kernels, so it is tested at the mechanism rather than
    deleted along with the gap that used to demonstrate it.
    """

    class Args:
        seed = 0
        min_time = 0.001
        repeat = 1
        max_op_seconds = 2.0

    def _raise(*_a, **_k):
        raise TypeError("qr is not available for dtype 'posit32' — it supports ...")

    monkeypatch.setattr(bench, "_make_call", _raise)

    r = bench._measure("qr", "posit32", 8, Args())
    assert r.status == "unavailable"
    assert r.seconds_per_op is None
    assert r.detail


# ---------------------------------------------------------------------------
# The accumulator axis (#73 item 2)
# ---------------------------------------------------------------------------


def test_accumulator_defaults_to_the_element_precision_only(bench):
    """Without --accumulators the sweep must be exactly what it always was, so
    the axis cannot silently multiply everyone's run time."""
    assert bench.DEFAULT_ACC == "default"

    class Args:
        accumulators = [bench.DEFAULT_ACC]

    for kernel in bench.KERNELS:
        assert bench._accumulators_for(kernel, "posit32", Args()) == [bench.DEFAULT_ACC]


def test_accumulators_are_read_off_the_extension_not_hardcoded(bench):
    """A quire exists only for the families Universal ships an fdp.hpp for.

    The harness must ask mtl5.mixed.accumulators() rather than carry its own
    list, or it drifts the moment a type is added — which is exactly how
    accumulators() itself once came to advertise a quire that dot() rejected.
    """

    class Args:
        accumulators = ["quire"]

    # posit32 has one; takum32 does not.
    assert "quire" in bench._accumulators_for("dot", "posit32", Args())
    assert "quire" not in bench._accumulators_for("dot", "takum32", Args())
    assert "quire" not in bench._accumulators_for("dot", "f64", Args())


def test_the_default_is_always_swept_and_always_first(bench):
    """Every other row's ratio is measured against it, so it cannot be
    omitted, and it must be measured before the rows that reference it."""

    class Args:
        accumulators = ["quire", "f64"]

    accs = bench._accumulators_for("dot", "posit32", Args())
    assert accs[0] == bench.DEFAULT_ACC
    assert set(accs) == {bench.DEFAULT_ACC, "quire", "f64"}


def test_lu_and_qr_have_no_accumulator_axis(bench):
    """They take no such parameter; sweeping them would invent an axis."""

    class Args:
        accumulators = ["quire", "f64"]

    for kernel in ("lu", "qr"):
        assert bench._accumulators_for(kernel, "posit32", Args()) == [bench.DEFAULT_ACC]
    for kernel in bench.ACC_KERNELS:
        assert len(bench._accumulators_for(kernel, "posit32", Args())) > 1


def test_native_gemm_reports_an_accumulator_as_unavailable(bench):
    """mtl5.matmul has no accumulator parameter, unlike mixed.matmul. That must
    land as an `unavailable` row rather than crashing the sweep or, worse,
    silently timing the default under another label."""

    class Args:
        seed = 0
        min_time = 0.001
        repeat = 1
        max_op_seconds = 2.0

    r = bench._measure("gemm", "f64", 8, Args(), "f64")
    assert r.status == "unavailable"
    assert r.accumulator == "f64"
    assert "accumulator" in r.detail


def test_accumulator_sweep_end_to_end(bench, tmp_path):
    # f64 is included because the harness refuses to report a sweep whose
    # baseline produced nothing — a posit32-only run exits 1 by design. It also
    # exercises the other half of the axis: f64 has no quire, so it must come
    # back with the default row alone rather than an error row.
    out = tmp_path / "acc.json"
    rc = bench.main(
        [
            "--kernels",
            "dot",
            "--dtypes",
            "f64",
            "posit32",
            "--accumulators",
            "quire",
            "--sizes",
            "512",
            "--min-time",
            "0.001",
            "--repeat",
            "1",
            "--json",
            str(out),
        ]
    )
    assert rc == 0

    rows = json.loads(out.read_text())["results"]
    assert {r["accumulator"] for r in rows if r["dtype"] == "f64"} == {"default"}

    by_acc = {r["accumulator"]: r for r in rows if r["status"] == "ok" and r["dtype"] == "posit32"}
    assert set(by_acc) == {"default", "quire"}

    # The default row is the reference for the ratio, so it carries none.
    assert by_acc["default"]["slowdown_vs_default_accumulator"] == pytest.approx(1.0)
    # The quire row must carry a ratio against it — structure, not magnitude.
    assert by_acc["quire"]["slowdown_vs_default_accumulator"] is not None
    assert by_acc["quire"]["slowdown_vs_default_accumulator"] > 0.0
