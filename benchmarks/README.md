# Benchmarks

## `bench_blas.py` — BLAS throughput by number type

Measures `dot`, `gemv`, `gemm`, `lu` and `qr` across the bound number systems
and reports **slowdown against float64**. That ratio, not raw GFLOP/s, is the
number mixed-precision experiment planning is built from: "posit32 GEMM costs
~4000x double" tells you directly whether a sweep fits in a night.

```bash
python benchmarks/bench_blas.py                  # default sweep
python benchmarks/bench_blas.py --quick          # smallest sizes, seconds
python benchmarks/bench_blas.py --json out.json  # machine-readable
python benchmarks/bench_blas.py --dtypes f64 posit32 --kernels gemm -v
```

`float` and `double` are accepted as aliases for `f32`/`f64`; results are
reported under MTL5's own dtype names.

### Reading the output

Three things about the numbers are structural, and the harness reports them
rather than hiding them.

**The native baselines use a different kernel.** `f32`/`f64` gemv/gemm run
through MTL5's blocked, Highway-vectorised path; the emulated types run through
the generic one. The ratio is an honest answer to "what do I pay?", but part of
every emulated-type gap is the vectoriser rather than the number system. Use
`cfloat32` to separate the two: it is IEEE binary32 semantics through the same
emulation layer, so `posit32 / cfloat32` isolates format-specific cost from
generic emulation cost.

**`lu` and `qr` are float32/float64 only.** Every emulated type reports
`not instantiated`. That is a binding gap (see issue #69), not a slow result —
do not read it as "too slow to measure".

**Slow cases are skipped, not silently truncated.** A case whose single call
exceeds `--max-op-seconds` (default 2s) is reported as `too slow` and the sweep
stops climbing sizes for that pair. Without it, a `qd_cascade` GEMM at n=128
would wedge the run for minutes.

### Build flags dominate

Every report embeds `mtl5.build_info()`, and a number quoted without it is not
reproducible. The blocked GEMM path is roughly **3.4x slower** than the generic
one when Highway is not there to vectorise it, and released wheels use static
dispatch at the baseline ISA unless built with `MTL5_NATIVE_ARCH=ON`. Wheels for
x86_64 and aarch64 therefore have different vectorisation baselines (SSE-era vs
NEON), so ratios do not necessarily transfer between them.

For a like-for-like comparison against a tuned BLAS, build with
`-C cmake.define.MTL5_WITH_BLAS=ON`.

### Methodology

- Operands are generated in float64 and converted **once**; conversion is not
  timed. The question is kernel throughput, and a real experiment converts once
  then iterates.
- Repetition count is calibrated so each timed batch lasts at least
  `--min-time`, then the **best** of `--repeat` batches is kept. Best-of rather
  than mean, because scheduler noise only ever adds time.
- `lu`/`qr` operands are diagonally dominant, so the measurement is arithmetic
  throughput rather than pivot pathology.
- FLOP counts are the standard leading-order ones: `2n` (dot), `2n²` (gemv),
  `2n³` (gemm), `⅔n³` (lu), `⁴⁄₃n³` (qr).

### JSON output

`--json` writes `{"meta": {...}, "results": [...]}`, where `meta` carries the
mtl5 version, `build_info()`, thread count, platform and timing parameters, and
each result row carries `status` (`ok` / `unavailable` / `too_slow`),
`seconds_per_op`, `gflops` and `slowdown_vs_baseline`. Keep the JSON, not the
table, if results are to be compared over time.
