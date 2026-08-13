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
            native = dtype in bench.NATIVE_DTYPES
            if kernel in ("lu", "qr") and not native:
                # Documented gap, not a failure — assert it is still a *clean*
                # TypeError so the harness keeps classifying it as
                # "unavailable" rather than crashing the sweep.
                with pytest.raises(TypeError):
                    fn()
            else:
                assert fn() is not None


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


def test_unavailable_is_reported_not_raised(bench):
    """lu/qr on an emulated type must land as a row, not an exception — a sweep
    that aborts on the first gap is useless for exactly the types we care
    about."""

    class Args:
        seed = 0
        min_time = 0.001
        repeat = 1
        max_op_seconds = 2.0

    r = bench._measure("qr", "posit32", 8, Args())
    assert r.status == "unavailable"
    assert r.seconds_per_op is None
    assert r.detail
