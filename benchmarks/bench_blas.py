#!/usr/bin/env python3
"""BLAS kernel throughput as a function of number type.

Answers the question mixed-precision algorithm research actually has: *how
much does it cost me to run this experiment in an emulated format?* The
headline number is therefore the slowdown ratio against float64, not raw
FLOP/s — an experiment plan is built out of "posit32 GEMM costs 40x double",
not out of an absolute GFLOP/s figure.

Usage
-----
    python benchmarks/bench_blas.py                     # default sweep
    python benchmarks/bench_blas.py --quick             # small sizes, fast
    python benchmarks/bench_blas.py --json out.json     # machine-readable
    python benchmarks/bench_blas.py --dtypes f64 posit32 --kernels gemm

Reading the results
-------------------
One asymmetry is structural, not measurement noise, and is reported rather
than hidden: `f32`/`f64` gemv/gemm go through MTL5's blocked,
Highway-vectorised kernels, while the emulated types go through the generic
ones. The ratio therefore measures what a researcher actually pays, which is
the right question — but it is not a like-for-like kernel comparison, and a
chunk of the gap is the vectoriser, not the number system. `lu` and `qr` have
no such split: every type runs the same kernel.

An `unavailable` row means the kernel is not instantiated for that dtype. Since
#69 that no longer happens for any of the nine swept formats, but the status is
kept because it is the honest way to report a gap, and it is what keeps a sweep
from aborting on the first one.

A NUMBER FROM A NARROW FORMAT IS NOT AUTOMATICALLY A RESULT. `fp8` and
`fixpnt8` QR degenerate: every Householder reflector rounds to zero, so Q comes
back as the exact identity and the factorization is meaningless while timing
perfectly normally. This harness measures throughput and cannot see that. See
tests/test_universal_factorizations.py for where each format stops being
usable.

Build flags dominate the numbers, so every run records `mtl5.build_info()`,
thread count and host architecture in its output. A figure quoted without that
context is not reproducible. For scale: the top-level CMakeLists records
double-precision matmul at 0.79 GF/s with the blocked GEMM alone against
2.15-3.51 GF/s with both flags off (GCC -O3, single thread, no -march=native,
n=200..1000) — roughly 3.4x slower without Highway to vectorise the
micro-kernel. Released wheels use static dispatch at the baseline ISA unless
built with MTL5_NATIVE_ARCH=ON, so they are not at the top of that range.

Every ratio this file or its README quotes is an example from one recorded run
on one machine, not a portable constant. Re-run it on the host you plan to
compute on.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

import mtl5

# --------------------------------------------------------------------------
# What to sweep
# --------------------------------------------------------------------------

# The nine formats issue #69 asks for. `float`/`double` are accepted as
# aliases on the command line because that is how the request is usually
# phrased, but results are reported under MTL5's own names.
DEFAULT_DTYPES = [
    "f64",
    "f32",
    "cfloat32",
    "posit32",
    "lns32",
    "takum32",
    "dd_cascade",
    "td_cascade",
    "qd_cascade",
]

DTYPE_ALIASES = {"double": "f64", "float": "f32", "float64": "f64", "float32": "f32"}

BASELINE = "f64"

KERNELS = ["dot", "gemv", "gemm", "lu", "qr"]

# Sizes are per kernel because the work scales differently: dot is O(n), gemv
# O(n^2), and gemm/lu/qr O(n^3). These defaults keep a float64 run in the
# millisecond range; the emulated types are held in check by --max-op-seconds
# rather than by shrinking the sweep for everyone.
DEFAULT_SIZES = {
    "dot": [1_000, 10_000, 100_000],
    "gemv": [64, 256, 512],
    "gemm": [32, 64, 128],
    "lu": [32, 64, 128],
    "qr": [32, 64, 128],
}

QUICK_SIZES = {
    "dot": [1_000],
    "gemv": [64],
    "gemm": [32],
    "lu": [32],
    "qr": [32],
}

# FLOP counts for a square problem of order n. LU and QR use the standard
# leading-order counts for the unblocked real factorizations.
FLOPS: dict[str, Callable[[int], float]] = {
    "dot": lambda n: 2.0 * n,
    "gemv": lambda n: 2.0 * n * n,
    "gemm": lambda n: 2.0 * n**3,
    "lu": lambda n: (2.0 / 3.0) * n**3,
    "qr": lambda n: (4.0 / 3.0) * n**3,
}

NATIVE_DTYPES = {"f32", "f64"}


@dataclass
class Result:
    kernel: str
    dtype: str
    n: int
    status: str  # "ok" | "unavailable" | "too_slow"
    seconds_per_op: float | None = None
    gflops: float | None = None
    slowdown_vs_baseline: float | None = None
    reps: int = 0
    detail: str = ""
    # True when the f64 baseline for this (kernel, n) is small enough that
    # per-call binding overhead is a material part of it. The ratio is then a
    # LOWER bound: the baseline is inflated by overhead the emulated types pay
    # too but barely notice, so the real kernel-to-kernel gap is wider.
    baseline_overhead_dominated: bool = False


@dataclass
class Report:
    meta: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# Operand construction — done once per (kernel, dtype, n), never timed
# --------------------------------------------------------------------------


def _make_operands(kernel: str, dtype: str, n: int, seed: int = 0):
    """Build the MTL5-side operands for one measurement.

    Everything is generated in float64 and converted once. Conversion cost is
    deliberately excluded: the question is kernel throughput, and a real
    experiment converts once and then iterates.
    """
    rng = np.random.default_rng(seed)

    if kernel == "dot":
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        return (mtl5.convert(a, dtype), mtl5.convert(b, dtype))

    if kernel == "gemv":
        A = rng.standard_normal((n, n))
        x = rng.standard_normal(n)
        return (mtl5.convert(A, dtype), mtl5.convert(x, dtype))

    if kernel == "gemm":
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        return (mtl5.convert(A, dtype), mtl5.convert(B, dtype))

    if kernel in ("lu", "qr"):
        # Diagonally dominant so the factorization is well conditioned and we
        # are timing arithmetic rather than pivot pathology.
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        return (mtl5.convert(A, dtype),)

    raise ValueError(f"unknown kernel {kernel!r}")


def _make_call(kernel: str, dtype: str, operands) -> Callable[[], object]:
    """Resolve the callable for one (kernel, dtype), or raise Unavailable.

    The native and emulated types reach the same mathematical operation by
    different entry points: `mtl5.mixed.*` carries the Universal containers,
    while gemv/gemm for f32/f64 live on the top-level module (register_
    mixed_native binds only dot and the norms). Both are the fastest published
    route to that operation for that type, which is what we want to compare.
    """
    native = dtype in NATIVE_DTYPES

    if kernel == "dot":
        a, b = operands
        return lambda: mtl5.mixed.dot(a, b)

    if kernel == "gemv":
        A, x = operands
        if native:
            return lambda: mtl5.matvec(A, x)
        return lambda: mtl5.mixed.matvec(A, x)

    if kernel == "gemm":
        A, B = operands
        if native:
            return lambda: mtl5.matmul(A, B)
        return lambda: mtl5.mixed.matmul(A, B)

    if kernel == "lu":
        (A,) = operands
        return lambda: mtl5.lu(A)

    if kernel == "qr":
        (A,) = operands
        return lambda: mtl5.qr(A)

    raise ValueError(f"unknown kernel {kernel!r}")


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def _time_call(fn: Callable[[], object], min_time: float, repeat: int) -> tuple[float, int]:
    """Return (best seconds per op, reps used).

    Calibrates the repetition count so one timed batch lasts at least
    `min_time`, then takes the best of `repeat` batches. Best-of rather than
    mean: we want the machine's capability with scheduler noise excluded, and
    noise only ever adds time.
    """
    # Calibrate from a single call, doubling until the batch is long enough.
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        elapsed = time.perf_counter() - t0
        if elapsed >= min_time or reps >= 1 << 30:
            break
        # Scale to the target with a little headroom, and always progress.
        growth = max(2, int(min_time / max(elapsed, 1e-9)) + 1)
        reps = min(reps * growth, 1 << 30)

    best = elapsed / reps
    for _ in range(repeat - 1):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best, reps


def measure_call_floor(kernel: str, args) -> float:
    """Seconds for the smallest possible instance of THIS kernel's binding.

    This is why the smallest sizes cannot be read naively. A float64 dot over
    64 elements is 128 flops; if crossing the binding costs microseconds, the
    baseline is nearly all overhead and every slowdown ratio computed against
    it is understated — flatteringly so, since the emulated types pay the same
    fixed cost but are far too slow to notice it.

    The floor is measured at n=1 on the same kernel and the same dtype as the
    baseline, so it captures argument marshalling and nanobind's overload
    resolution for that exact signature. A generic empty binding
    (get_num_threads) measures none of that and reported a floor ~50x too
    optimistic, which is how an obviously overhead-bound row escaped the flag.
    """
    operands = _make_operands(kernel, BASELINE, 1, seed=args.seed)
    fn = _make_call(kernel, BASELINE, operands)
    fn()  # warm up
    sec, _ = _time_call(fn, args.min_time, args.repeat)
    return sec


# A baseline within this multiple of its kernel's call floor is treated as
# overhead-contaminated. 20x leaves the overhead under ~5% of the measurement.
OVERHEAD_GUARD = 20.0


def _measure(kernel: str, dtype: str, n: int, args) -> Result:
    try:
        operands = _make_operands(kernel, dtype, n, seed=args.seed)
        fn = _make_call(kernel, dtype, operands)
    except (TypeError, ValueError) as exc:
        # mtl5.qr/lu raise TypeError for dtypes they are not instantiated for.
        # That is a binding gap, reported as such rather than as a failure.
        return Result(kernel, dtype, n, "unavailable", detail=str(exc).split("—")[0].strip())

    # One untimed warmup: it populates caches and, more importantly, tells us
    # whether a single call already blows the per-op budget. Without this an
    # emulated qd_cascade gemm could wedge the sweep for minutes.
    t0 = time.perf_counter()
    try:
        fn()
    except (TypeError, ValueError) as exc:
        return Result(kernel, dtype, n, "unavailable", detail=str(exc).split("—")[0].strip())
    warm = time.perf_counter() - t0

    if warm > args.max_op_seconds:
        return Result(
            kernel,
            dtype,
            n,
            "too_slow",
            seconds_per_op=warm,
            detail=f"single call {warm:.2f}s exceeds --max-op-seconds {args.max_op_seconds}",
        )

    sec, reps = _time_call(fn, args.min_time, args.repeat)
    gflops = FLOPS[kernel](n) / sec / 1e9
    return Result(kernel, dtype, n, "ok", seconds_per_op=sec, gflops=gflops, reps=reps)


# --------------------------------------------------------------------------
# Driving the sweep
# --------------------------------------------------------------------------


def run(args) -> Report:
    sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES
    if args.sizes:
        sizes = {k: list(args.sizes) for k in KERNELS}

    results: list[Result] = []
    for kernel in args.kernels:
        for dtype in args.dtypes:
            for n in sizes[kernel]:
                r = _measure(kernel, dtype, n, args)
                results.append(r)
                if args.verbose:
                    print(f"  {kernel:5} {dtype:11} n={n:<7} {r.status}", file=sys.stderr)
                # Once a type is too slow at one size it will only be worse at
                # the next, so stop climbing rather than burn the budget.
                if r.status == "too_slow":
                    break
                # `unavailable` is size-independent — skip the rest of the sweep
                # for this pair instead of re-reporting the same gap N times.
                if r.status == "unavailable":
                    break

    floors = {k: measure_call_floor(k, args) for k in args.kernels}

    baseline = {
        (r.kernel, r.n): r.seconds_per_op
        for r in results
        if r.dtype == BASELINE and r.status == "ok"
    }
    for r in results:
        base = baseline.get((r.kernel, r.n))
        if base and r.seconds_per_op:
            r.slowdown_vs_baseline = r.seconds_per_op / base
            r.baseline_overhead_dominated = base < OVERHEAD_GUARD * floors[r.kernel]

    return Report(
        meta={
            "mtl5_version": mtl5.__version__,
            "build_info": mtl5.build_info(),
            "num_threads": mtl5.get_num_threads(),
            "baseline_dtype": BASELINE,
            "call_floor_s": floors,
            "platform": {
                "machine": platform.machine(),
                "processor": platform.processor(),
                "system": platform.system(),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "timing": {
                "min_time_s": args.min_time,
                "repeat": args.repeat,
                "max_op_seconds": args.max_op_seconds,
                "seed": args.seed,
            },
        },
        results=[asdict(r) for r in results],
    )


def print_table(report: Report) -> None:
    bi = report.meta["build_info"]
    flags = ", ".join(k for k, v in bi.items() if v) or "none"
    print()
    print(f"mtl5 {report.meta['mtl5_version']} — build flags: {flags}")
    print(
        f"{report.meta['platform']['machine']}, "
        f"{report.meta['num_threads']} thread(s), "
        f"baseline {BASELINE}"
    )

    by_kernel: dict[str, list[dict]] = {}
    for r in report.results:
        by_kernel.setdefault(r["kernel"], []).append(r)

    for kernel, rows in by_kernel.items():
        print()
        print(f"== {kernel} " + "=" * (66 - len(kernel)))
        print(f"{'dtype':12} {'n':>7} {'s/op':>12} {'GFLOP/s':>10} {'vs f64':>10}")
        print("-" * 55)
        flagged = False
        for r in rows:
            if r["status"] == "ok":
                slow = r["slowdown_vs_baseline"]
                mark = ""
                if slow and r["baseline_overhead_dominated"]:
                    mark = ">"  # ratio is a lower bound; see the footnote
                    flagged = True
                slow_s = f"{mark}{slow:8.1f}x" if slow else "  baseline"
                print(
                    f"{r['dtype']:12} {r['n']:>7} {r['seconds_per_op']:12.3e} "
                    f"{r['gflops']:10.3f} {slow_s:>10}"
                )
            else:
                note = "not instantiated" if r["status"] == "unavailable" else "too slow"
                print(f"{r['dtype']:12} {r['n']:>7} {note:>12} {'-':>10} {'-':>10}")
        if flagged:
            print(
                "  > baseline is within "
                f"{OVERHEAD_GUARD:.0f}x of per-call overhead — ratio is a LOWER "
                "bound; prefer the larger sizes"
            )


def _positive_int(value: str) -> int:
    """argparse type for counts that must be >= 1.

    n=0 would divide by zero in the FLOP rate, and --repeat 0 silently degrades
    to one batch rather than the documented count.
    """
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than zero, got {result}")
    return result


def _positive_seconds(value: str) -> float:
    """argparse type for durations.

    The finiteness check is not decorative: a NaN --min-time makes every
    `elapsed >= min_time` comparison false, so calibration runs to
    `int(nan)` and dies with an unhelpful ValueError deep in the timing loop.
    """
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(f"must be finite and greater than zero, got {value!r}")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Measure BLAS kernel throughput as a function of number type.\n"
            "\n"
            "Runs dot, gemv, gemm, lu and qr across the bound number systems and\n"
            "reports SLOWDOWN AGAINST float64 rather than raw GFLOP/s — that is the\n"
            "figure a mixed-precision experiment plan is built from ('posit32 GEMM\n"
            "costs 4000x double' decides whether a sweep fits in a night).\n"
            "\n"
            "Reading the output:\n"
            "  * f32/f64 gemv/gemm run MTL5's blocked, Highway-vectorised kernels\n"
            "    while emulated types run the generic ones, so part of that gap is\n"
            "    the vectoriser. lu and qr run the same kernel on both sides and\n"
            "    are the like-for-like comparison.\n"
            "  * cfloat32 is the control: IEEE binary32 through the same emulation\n"
            "    layer, so posit32/cfloat32 separates format cost from emulation\n"
            "    cost.\n"
            "  * A '>' on a row means the float64 baseline is close to per-call\n"
            "    overhead, so that ratio is a LOWER bound — prefer the larger sizes.\n"
            "  * Timing cannot see a wrong answer. fp8 and fixpnt8 QR return the\n"
            "    identity for Q and time perfectly normally.\n"
            "\n"
            "Ratios move with build flags, ISA dispatch, thread count and\n"
            "architecture, so every report embeds all four. Full methodology:\n"
            "benchmarks/README.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--kernels", nargs="+", default=KERNELS, choices=KERNELS)
    p.add_argument("--dtypes", nargs="+", default=DEFAULT_DTYPES)
    p.add_argument("--sizes", nargs="+", type=_positive_int, help="override sizes for every kernel")
    p.add_argument("--quick", action="store_true", help="smallest sizes only")
    p.add_argument(
        "--min-time", type=_positive_seconds, default=0.05, help="seconds per timed batch"
    )
    p.add_argument("--repeat", type=_positive_int, default=3, help="timed batches; best is kept")
    p.add_argument(
        "--max-op-seconds",
        type=_positive_seconds,
        default=2.0,
        help="skip a case whose single call exceeds this (keeps emulated sweeps finite)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    args.dtypes = [DTYPE_ALIASES.get(d, d) for d in args.dtypes]
    known = set(mtl5.dtypes())
    unknown = [d for d in args.dtypes if d not in known]
    if unknown:
        p.error(f"unknown dtype(s) {unknown}; available: {sorted(known)}")

    report = run(args)
    print_table(report)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(asdict(report), fh, indent=2)
        print(f"\nwrote {args.json}")

    # A sweep where the baseline itself failed is not a usable result.
    if not any(r["dtype"] == BASELINE and r["status"] == "ok" for r in report.results):
        print("\nERROR: baseline f64 produced no successful measurement", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
