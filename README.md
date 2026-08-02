# mtl5-python

Python bindings for [MTL5](https://github.com/stillwater-sc/mtl5) — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch.

Built with [nanobind](https://github.com/wjakob/nanobind) for minimal overhead and zero-copy array interop.

## Install

```bash
pip install .
```

Requires Python 3.10+ and a C++20 compiler (GCC 12+, Clang 15+, MSVC 2022).

## Quick start

```python
import numpy as np
import mtl5

# Vectors and norms
v = mtl5.vector(np.array([3.0, 4.0]))
print(mtl5.norm(np.array([3.0, 4.0])))  # 5.0

# Dot product
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print(mtl5.dot(a, b))  # 32.0

# Solve Ax = b
A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([5.0, 7.0])
x = mtl5.solve(A, b)
print(x)  # [1.6, 1.8]
```

## Mixed precision

A mixed-precision operation has three independent precisions. MTL5 supplies the
precision-generic kernels, [Universal](https://github.com/stillwater-sc/universal)
supplies the number systems, and this package composes them:

| | chosen by |
|---|---|
| **element** (storage) | the container you pass in — `mtl5.convert(x, "posit16")` |
| **accumulator** (compute) | `accumulator=` |
| **result** (delivery) | `result=` |

```python
import numpy as np, mtl5

x = np.random.default_rng(0).standard_normal(4000)
v = mtl5.convert(x, "posit16")  # store narrow

mtl5.mixed.dot(v, v)  # accumulate in posit16 too
mtl5.mixed.dot(v, v, accumulator="f64")  # ...or in double
mtl5.mixed.dot(v, v, accumulator="quire")  # ...or exactly, in Universal's quire
```

Accumulating 4000 posit16 products, against the exact value of the same
posit16 data:

| accumulator | relative error |
|---|---|
| none (posit16) | 1.8 × 10⁻¹ |
| `"f32"` | 3.8 × 10⁻⁷ |
| `"f64"` | 1.1 × 10⁻¹⁶ |
| `"fma"` | 1.1 × 10⁻¹⁶ |
| `"quire"` | **0** — bit-exact |

`mtl5.mixed.accumulators(dtype)` lists what a given element type supports. The
quire is available for the posit, cfloat, lns and fixpnt families; `f32`/`f64`
have none (Universal defines no quire for the native types). Exactness varies by
family — it is genuinely exact for posit and fixpnt, while the cfloat and lns
quires have known upstream limitations documented in
`python/include/mtl/math/quire_accumulator.hpp`. The accumulated norms are
computed locally rather than through `mtl::two_norm<Acc>`, which cannot take a
quire ([stillwater-sc/mtl5#324](https://github.com/stillwater-sc/mtl5/issues/324)).

`accumulator=` is available on `dot`, `norm` (ord=2), `frobenius_norm`,
`matvec` and `matmul`.

### Iterative refinement

Factor cheaply in a low precision, then recover accuracy with a residual formed
in float64:

```python
x, info = mtl5.mixed.lu_iterative_refine(A, b, working="posit16", rel_tol=1e-14)
# info -> {'iters': 7, 'rel_residual': ..., 'converged': True}
mtl5.mixed.backward_error(A, x, b)
```

On a 100×100 system, forward error of the refined solution:

| working precision | iterations | forward error |
|---|---|---|
| `fp16` | 5 | 1.9 × 10⁻⁸ |
| `posit16` | 7 | 1.9 × 10⁻¹⁵ |
| `f32` | 2 | 3.8 × 10⁻¹⁶ |
| `f64` | 0 | 3.1 × 10⁻¹⁶ |

The result is always the *best* iterate found, so an over-long `max_iter` never
degrades the answer.

`mtl5.mixed.iterative_refine(A, M, b)` is the sparse counterpart, refining
through any factorization exposing `solve()`. Today that means the ILU(0)/IC(0)
preconditioners; the sparse direct factorizations that make this the
mixed-precision workhorse arrive with the sparse-solver bindings.

> **Known upstream defect:** MTL5's `ilu_0::solve` returns wrong values for
> every input (it sums the diagonal into the off-diagonal term of its back
> substitution). `mtl5.sparse.ilu0` is affected; IC(0) is correct. Filed as
> [stillwater-sc/mtl5#323](https://github.com/stillwater-sc/mtl5/issues/323) and
> pinned by a strict-xfail regression in `tests/test_mixed_precision.py`.

## Performance

### Threading

MTL5's kernels are parallel but **serial by default**. The thread pool is sized
once, on the first MTL5 call, and cannot be resized afterwards — so configure it
before you do any work:

```python
import mtl5

mtl5.set_num_threads(8)  # must precede the first MTL5 operation
print(mtl5.get_num_threads())
```

Equivalently, set `MTL5_NUM_THREADS` in the environment before importing:

```bash
MTL5_NUM_THREADS=8 python my_script.py
```

`set_num_threads()` raises `RuntimeError` rather than silently doing nothing if
the pool has already been sized. The count is clamped to hardware concurrency.

All non-trivial kernels release the GIL, so MTL5 work overlaps with other Python
threads instead of blocking the interpreter.

Double-precision `matmul`, 1000×1000, on a 20-core box:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| GFLOP/s | 14.4 | 27.2 | 48.0 | 72.7 |

### Build options

The wheel builds MTL5's blocked GEMM vectorised through
[Google Highway](https://github.com/google/highway) by default. Everything else
is opt-in, since it needs a library on the build machine or makes the binary
non-portable:

```bash
pip install . -C cmake.define.MTL5_WITH_BLAS=ON      # external BLAS
pip install . -C cmake.define.MTL5_WITH_LAPACK=ON    # geev/syev/gesdd dispatch
pip install . -C cmake.define.MTL5_NATIVE_ARCH=ON    # -march=native, local builds only
```

`MTL5_NATIVE_ARCH=ON` lets Highway target AVX2/AVX-512 rather than the x86-64
baseline — a large additional speedup, but the resulting binary only runs on
machines like the one that built it.

Check what a given install actually has:

```python
>>> mtl5.build_info()
{'blas': False, 'lapack': False, 'native_fast_gemm': True,
 'highway_simd': True, 'kpu': False}
>>> mtl5.get_backend()
'native'
```

`set_backend()` validates against this build rather than silently accepting a
backend that was never compiled in; backend selection itself is compile-time.

## Development

```bash
pip install -e ".[dev]"
pytest -v
```
