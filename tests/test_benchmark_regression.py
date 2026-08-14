"""Regression guard on the benchmark numbers (#73 item 3).

The harness emits JSON with build flags, thread count and platform precisely so
results can be tracked, and until now nothing consumed it: a build change that
made posit32 GEMM 3x slower would not have been noticed.

Guarding timings is a good way to build a flaky test, so both *what* is
asserted and *which kernel* it is asserted on were chosen from measured
stability rather than taste. Four consecutive runs per configuration on an idle
machine, ratios taken WITHIN each run:

    kernel        ratio               spread over 4 runs
    gemm n=64     posit32/takum32     1.04x     <-- asserted
    gemm n=64     posit32/f64         1.03x     <-- asserted
    gemv n=256    posit32/takum32     1.42x
    gemv n=256    posit32/f64         1.24x
    dot  n=10000  posit32/takum32     3.29x     <-- unusable
    dot  n=64     posit32/takum32     2.54x     <-- unusable

**Only gemm is guarded.** dot and gemv were tried and rejected on the numbers
above, not on taste: a guard that swings 3x between identical runs cannot
distinguish a regression from a Tuesday. dot stays unstable even at n=10000
where the operands are far above per-call overhead and even with best-of-3, so
the instability is not something a larger size or more repeats fixes.

Why within-run ratios at all: they cancel machine speed, core count and
scheduler noise almost exactly, because both operands run in the same process
on the same hardware moments apart. An absolute ns/op threshold would have to
be re-baselined per runner and would still flap.

Bounds are order-of-magnitude. The goal is catching a structural regression — a
vectoriser silently off, an emulated path falling back, a format losing its
fast route — not detecting a 20% drift that a busier runner produces anyway.
Every bound has at least 3x headroom against every value measured across all
configurations tried during #69-#73, including deliberately contended runs.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "bench_blas.py"

pytestmark = [
    pytest.mark.skipif(not BENCH.exists(), reason="benchmarks/ not present"),
    # Deselect with `pytest -m "not perf"` where timing is meaningless — an
    # emulated or heavily oversubscribed runner.
    pytest.mark.perf,
]

# gemm at n=64: large enough that the float64 baseline escapes per-call
# overhead (asserted below, not assumed), small enough that posit32 stays near
# a second per call. The whole guard costs a few seconds.
KERNEL = "gemm"
SIZE = 64
DTYPES = ["f64", "f32", "takum32", "posit32"]


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_blas_guard", BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sweep(bench):
    """One sweep shared by every assertion.

    repeat=3 (the harness default) rather than 1: best-of-3 discards a
    scheduler hiccup, and every ratio below is between two rows of this same
    run, so nothing here compares across processes.
    """

    class Args:
        kernels = [KERNEL]
        dtypes = DTYPES
        sizes = [SIZE]
        quick = False
        accumulators = [bench.DEFAULT_ACC]
        min_time = 0.05
        repeat = 3
        max_op_seconds = 30.0
        seed = 0
        verbose = False

    return {r["dtype"]: r for r in bench.run(Args()).results if r["status"] == "ok"}


def _seconds(sweep, dtype):
    if dtype not in sweep:
        pytest.skip(f"{KERNEL}/{dtype} produced no measurement")
    return sweep[dtype]["seconds_per_op"]


def test_the_baseline_escaped_per_call_overhead(sweep):
    """The precondition for every ratio below, asserted rather than assumed.

    If the float64 baseline were overhead-dominated the harness would flag it,
    the ratios would become lower bounds, and their run-to-run spread would
    jump — which is exactly what makes dot unusable. Fail loudly rather than
    guard a number that has quietly stopped meaning anything.
    """
    if "f64" not in sweep:
        pytest.skip("no baseline measurement")
    assert not sweep["f64"]["baseline_overhead_dominated"], (
        f"the f64 {KERNEL} baseline at n={SIZE} is now overhead-dominated, so "
        "the ratios below are lower bounds and no longer stable enough to "
        "assert — raise SIZE"
    )


def test_posit32_is_several_times_costlier_than_takum32(sweep):
    """The tightest guard, because both operands are emulated: they pay the
    same per-call overhead and run the same generic kernel, so only the number
    systems' arithmetic differs. Measured 9.2-10.1 across every run."""
    r = _seconds(sweep, "posit32") / _seconds(sweep, "takum32")
    assert 3.0 < r < 30.0, (
        f"posit32/takum32 = {r:.1f} on {KERNEL}, outside the 3-30x band. "
        "Either a format changed cost class or one of them stopped using the "
        "path it used to."
    )


def test_emulation_cost_stays_within_an_order_of_magnitude_band(sweep):
    """posit32 against the native baseline.

    Wider than the emulated-vs-emulated band on purpose: this ratio moved
    between 2900x and 6600x across builds during #69-#73 without anything being
    wrong, because it also tracks how well the NATIVE side is vectorised.
    """
    r = sweep["posit32"]["slowdown_vs_baseline"] if "posit32" in sweep else None
    if r is None:
        pytest.skip("posit32 produced no measurement")
    assert 10.0 < r < 100_000.0, (
        f"posit32 {KERNEL} = {r:.0f}x float64, outside the 10-100000x band. "
        "Below the floor usually means the NATIVE path regressed — a "
        "vectoriser silently off makes emulation look cheap. Above the ceiling "
        "means the emulated path lost its route."
    )


def test_native_types_stay_comparable(sweep):
    """f32 and f64 run the same blocked, vectorised kernel, so a large gap is a
    build problem rather than a precision property. This is the assertion that
    catches Highway being off for one of them."""
    r = _seconds(sweep, "f32") / _seconds(sweep, "f64")
    assert 0.05 < r < 20.0, (
        f"f32/f64 = {r:.2f} on {KERNEL}, outside 0.05-20x. The two native types "
        "share a kernel; a large gap points at the build, not at precision."
    )


def test_the_guard_is_not_vacuous(sweep):
    """Every assertion above skips when its dtype is missing, so a sweep that
    quietly produced nothing would pass the lot."""
    missing = [d for d in DTYPES if d not in sweep]
    assert not missing, f"no successful {KERNEL} measurement for {missing}"
