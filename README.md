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
through any factorization exposing `solve()` — the sparse direct factorizations
below, or the ILU(0)/IC(0) preconditioners.

## Matrix Market I/O and sparsity pictures

```python
A = mtl5.io.mm_read("circuit.mtx")  # -> CSR
D = mtl5.io.mm_read_dense("small.mtx")  # -> dense
mtl5.io.mm_write_sparse("out.mtx", A, "comment")

mtl5.io.spy(A, "pattern.png")  # binary non-zero pattern
mtl5.io.spy_magnitude(A, "mag.png", log_scale=True)
mtl5.io.spy_density(A, "dens.png", max_pixels=512)
```

Two things differ from what a SciPy user expects.

**The function picks the container, not the file.** `scipy.io.mmread` returns an
ndarray for an `array` file and COO for a `coordinate` one. Here `mm_read`
always gives CSR and `mm_read_dense` always gives dense, and *both* accept
either file format — so reading a dense `.mtx` into CSR is a deliberate call
rather than an error.

**The PNGs are uncompressed.** MTL5 writes them with a from-scratch encoder
emitting DEFLATE *stored* blocks, which is what lets `spy` work with no image
library and no plotting stack in the process. File size is therefore about
`width × height × channels` — measured at the default `max_pixels=1024`, that
is ~1.0 MB for `spy` (grayscale) and ~3.1 MB for `spy_magnitude` /
`spy_density` (RGB). Pipe the output through any PNG optimizer if that matters.

`.gz` inputs are read transparently only when MTL5 is built with zlib
(`-C cmake.define.MTL5_WITH_ZLIB=ON`); `mtl5.build_info()["zlib"]` reports it,
and without it a `.gz` path raises rather than silently mis-parsing.

## Test matrices

A named catalog of matrices with known pathologies — the inputs a
mixed-precision experiment actually wants, and tedious to hand-roll correctly:

```python
g = mtl5.generators

g.hilbert(8)  # cond ~1.5e10 — the canonical ill-conditioned matrix
g.clement(7)  # eigenvalues exactly -6,-4,-2,0,2,4,6
g.randspd(4, [1, 10, 100, 1000])  # SPD with exactly that spectrum
g.randsym(4, [-5, -1, 2, 7])  # controlled *indefinite* matrix
g.randsvd(20, 20, kappa=1e6)  # condition number exactly 1e6
g.laplacian_2d(64, 64)  # sparse, the usual solver benchmark
```

Also `frank`, `pascal`, `wilkinson`, `rosser`, `magic`, `lehmer`, `lotkin`,
`minij`, `ones`, `forsythe`, `kahan`, `moler`, `companion`, `vandermonde`,
`randorth`, `laplacian_1d`, `poisson2d`, and the published catalog via
`testsuite_names()` / `testsuite_kappa(name)`.

Every dense generator takes `dtype=`, which is what makes them useful here:

```python
H = g.hilbert(8, dtype="posit16")  # correctly rounded posit16 Hilbert
mtl5.cholesky(H)  # ...then watch it fail
```

Generation happens in float64 and `dtype=` rounds. That is the right semantics
for a test matrix — the definitions are over the reals, so you want the
correctly rounded representation of the exact entry, not the result of
evaluating the formula in low-precision arithmetic.

Range vectors follow NumPy, and take `dtype=` too:

```python
mtl5.linspace(0, 1, 5)
mtl5.arange(0, 10, 3)
mtl5.logspace(0, 3, 4)
mtl5.geomspace(1, 1000, 4)  # 1, 10, 100, 1000 — a true geometric
# progression, not logspace's exponents
```

## Dense factorizations

```python
qr = mtl5.qr(A)  # Householder QR; tall or square
x = qr.solve(b)  # least squares
qr.Q, qr.R

lq = mtl5.lq(A)  # the row-space counterpart
lq.L, lq.Q

ld = mtl5.ldlt(A)  # symmetric, indefinite allowed
ld.solve(b)
ld.diagonal()  # D — its signs are the inertia
```

All accept an MTL5 matrix or a float32/float64 NumPy array, alongside the
existing `mtl5.lu` and `mtl5.cholesky`.

### Cholesky vs LDLᵀ across number systems

`ldlt` and `cholesky` are both available for float32, float64 and **all ten
Universal dtypes** — the integer element types are not supported — which is what
makes the interesting comparison possible. Cholesky takes square roots, so
it refuses a matrix that has drifted out of positive-definiteness — the failure
mode of a Kalman covariance update in low precision. LDLᵀ has no square roots,
survives, and records what happened in `D`:

```python
P = np.eye(6)
P[3, 3] = -1e-3  # covariance went indefinite

mtl5.cholesky(mtl5.convert(P, "posit16"))  # RuntimeError: not SPD
d = mtl5.ldlt(mtl5.convert(P, "posit16")).diagonal()
(d < 0).any()  # True — D names the bad direction
```

`mtl5.bunch_kaufman(A)` is the pivoting variant, for a symmetric matrix that
plain `ldlt` rejects on a zero pivot. float32/float64 only.

## Eigenvalues, BLAS 2/3, and matrix properties

The eigen entry points mirror `numpy.linalg`, and return NumPy arrays:

```python
mtl5.eigvalsh(A)  # symmetric eigenvalues, real, ascending
w, Q = mtl5.eigh(A)  # ...with eigenvectors:  A = Q diag(w) Qᵀ
mtl5.eigvals(A)  # general spectrum, complex
w, V = mtl5.eig(A)  # ...with right eigenvectors
mtl5.spectral_radius(A)
mtl5.inertia(A)  # {'positive': …, 'negative': …, 'zero': …}
```

BLAS levels 2 and 3 write into a caller-supplied output, as BLAS does — that
in-place accumulation is the point:

```python
mtl5.ger(alpha, x, y, A)  # A += alpha x yᵀ
mtl5.symv(alpha, A, x, beta, y)  # y = alpha A x + beta y
mtl5.trsv(A, x, upper=True)  # x = A⁻¹x
mtl5.trmm(alpha, A, B, upper=True)  # B = alpha A B
mtl5.trsm(alpha, A, B, upper=True)  # solve A X = alpha B
mtl5.symm(alpha, A, B, beta, C)  # C = alpha A B + beta C
mtl5.syrk(alpha, A, beta, C)  # C = alpha A Aᵀ + beta C
mtl5.syr2k(alpha, A, B, beta, C)
```

Property predicates come in two cost classes. The **docstrings are
authoritative**; the split is:

- **O(n²) or cheaper** — the structural checks (`is_square`, `is_empty`,
  `is_symmetric`, `is_hermitian`, `is_triangular` and the upper/lower variants,
  `is_diagonal`, `is_banded`, `is_diagonally_dominant`) and every vector
  predicate.
- **O(n³)** — anything that forms a product, factorizes, or runs an
  eigensolve: `is_orthogonal`, `is_unitary`, `is_normal`, `is_spd`,
  `is_positive_definite`, `is_singular`, `is_nonsingular`, `is_invertible`,
  `spectral_radius`, `inertia`, `is_indefinite`. Don't put these inside a loop.

SVD and the queries built on it:

```python
U, s, V = mtl5.svd(A)  # s is the vector of singular values (NumPy's convention)
mtl5.svdvals(A)  # singular values only — cheaper
mtl5.condition_number(A)  # σ_max / σ_min
mtl5.rcond(A)  # σ_min / σ_max, safer near singular
mtl5.numerical_rank(A)
mtl5.nullity(A)
```

`svd` takes a `tol`. `V` and the reconstruction are accurate to machine
precision regardless, but `U`'s orthogonality is bounded by the iteration's
tolerance rather than by eps. Over 160 matrices (n = 3..20, four tolerances)
`‖UᵀU − I‖` had a median of about 1.4×tol and a worst case of 4.5×tol; the
regression test asserts 20×tol to leave headroom. Treat `tol` as the knob:
tighten it if you need an orthonormal `U` specifically, rather than relying on
a particular multiple.

## Complex numbers

`mtl5.vector()` and `mtl5.matrix()` accept `complex64` and `complex128` arrays
and give the same zero-copy views as the real types, with dtypes `c64`/`c128`:

```python
A = mtl5.matrix(np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]]))
b = mtl5.vector(np.array([3 + 1j, 1 + 4j]))
x = mtl5.solve(A, b)  # complex LU with partial pivoting
mtl5.norm(x, 2)  # a real float, not a complex
A.real.to_numpy()  # real part, as a real matrix
```

Three things differ from the real case, and getting them wrong is quiet rather
than loud, so they are worth stating.

**`dot` is Hermitian.** It computes `sum(conj(a[i]) * b[i])`, conjugating the
*first* argument — that is NumPy's `vdot`, not NumPy's `dot`. The unconjugated
bilinear product is `mtl5.dot_real`, which is what `np.dot` does for 1-D
complex. Both exist because both are wanted; the names say which is which.

**`.T` does not conjugate.** MTL5's transpose is the plain one. `.H` (or
`mtl5.adjoint`) is the conjugate transpose. For real elements the two coincide,
which is exactly why the distinction has to be explicit here.

**Hermitian and symmetric are different properties.** `mtl5.is_hermitian(A)`
tests `A == Aᴴ` and `mtl5.is_symmetric(A)` tests `A == Aᵀ`; for complex those
are not the same matrix, and which one you have decides which solver is right.

What is available: containers and factories, all four norms, `dot`/`dot_real`,
`matmul`/`matvec`, `solve`/`lu`/`inv`, `transpose`/`adjoint`/`conj`, and
`ldlt_solve`. What is not: `cholesky`, `qr`, `lq`, `bunch_kaufman`, the eigen
and SVD family, and the Krylov solvers — MTL5 has no complex implementation of
any of them, and complex input raises a `TypeError` naming the alternative
rather than silently taking a real part.

`mtl5.ldlt_solve` is the one place a guard was needed. MTL5's `ldlt` is LDLᵀ
with no conjugation, so it is correct for a complex *symmetric* matrix and
wrong for a Hermitian one — and it reports success either way. Hermitian input
is refused here; use `mtl5.solve`, which handles it correctly.

```python
S = mtl5.matrix(np.array([[2 + 1j, 1 - 1j], [1 - 1j, 3 + 2j]]))  # A == A^T
mtl5.ldlt_solve(S, b)  # fine

H = mtl5.matrix(np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]]))  # A == A^H
mtl5.ldlt_solve(H, b)  # ValueError, by design
```

Complex is not in `mtl5.dtypes()`, which lists what `mtl5.convert()` accepts —
the Universal number systems are real-only, so there is no complex target.

## Sparse direct solvers

Seven factorizations, one interface — construct, `.solve(b)`, `.refactor(A2)`:

| | for | notes |
|---|---|---|
| `ms.splu` | general square | Gilbert–Peierls, threshold pivoting |
| `ms.klu` | circuit matrices | block triangular form + per-block LU |
| `ms.supernodal_lu` | general square | dense block updates; `.nsuper` |
| `ms.cholesky` | symmetric positive definite | cheapest when it applies |
| `ms.ldlt` | symmetric, possibly indefinite | `.diagonal()` gives the inertia |
| `ms.supernodal_ldlt` | symmetric | dense block updates |
| `ms.qr` | least squares, rectangular | `min ‖Ax − b‖₂` |

```python
import mtl5.sparse as ms

lu = ms.splu(A, ordering="amd")  # analyze (ordering + symbolic) then factor
x = lu.solve(b)

lu.refactor(A2)  # numeric only — same pattern, new values
x2 = lu.solve(b2)

k = ms.klu(A)  # block triangular form + per-block LU
k.nblocks  # how reducible the matrix turned out to be
```

Two things `scipy.sparse.linalg.splu` cannot do.

**Refactorization.** A sequence of matrices sharing one sparsity pattern — the
circuit-transient case — pays for the ordering and symbolic analysis once. On a
2-D Laplacian, n=3600, nnz(A)=17,760:

| | nnz(factor) | factor | refactor | speedup |
|---|---|---|---|---|
| `splu` | 205,636 | 12.5 ms | 4.5 ms | 2.8× |
| `klu` | 119,530 | 6.7 ms | 2.2 ms | 3.0× |
| `supernodal_lu` | 205,636 | 19.6 ms | 4.0 ms | **5.0×** |
| `cholesky` | 59,765 | 22.3 ms | 20.8 ms | 1.1× |
| `ldlt` | 56,165 | 8.4 ms | 7.0 ms | 1.2× |
| `supernodal_ldlt` | 56,165 | 4.7 ms | 2.2 ms | 2.1× |

Two things to read off that table. Exploiting symmetry cuts the fill to about a
quarter, and `supernodal_ldlt` is both the sparsest and the fastest option for a
symmetric matrix. And the refactor win is small for `cholesky`/`ldlt` — they do
not pivot, so their analysis is a cheap symbolic pass and there is little to
skip; the LU-family factorizations, which must otherwise redo ordering *and* the
pivot search, gain the most.

The saving is the analysis, so it scales with how much of the runtime that is.
On a random pattern with catastrophic fill, numeric work dominates and refactor
is no faster (we measured a slight loss). Structured sparsity is where it pays.

**A factor narrower than the residual.** The factor's precision is chosen by
the dtype of `A`, independent of the precision you refine in:

```python
lu32 = ms.splu(A.astype(np.float32), ordering="amd")
x, info = mtl5.mixed.iterative_refine(ms.from_scipy(A), lu32, b, rel_tol=1e-14)
```

On a 2-D Laplacian, n=1600:

| | forward error |
|---|---|
| float64 factor, direct solve | 2.0 × 10⁻¹⁵ |
| float32 factor, direct solve | 8.9 × 10⁻⁷ |
| **float32 factor + float64 refinement** (3 iters) | **7.0 × 10⁻¹⁶** |

Half the factorization memory and traffic, and the refined answer is *better*
than the float64 direct solve. Every square factorization above works this way;
`qr` is excluded because least squares is not the square system refinement
corrects.

### Orderings

`ms.orderings()` lists `amd`, `colamd`, `rcm` and `natural`. Each is also
available standalone as a permutation, for inspection or for use on a scipy
matrix directly:

```python
p = ms.amd(A)  # or ms.colamd(A), ms.rcm(A), ms.ordering(A, name)
A[p][:, p]
```

The choice matters. Nonzeros in L+U for the 2-D Laplacian above (n=1600,
7840 nonzeros in A):

| ordering | `amd` | `colamd` | `rcm` | `natural` |
|---|---|---|---|---|
| nnz(L+U) | 41,542 | 62,944 | 90,040 | 128,078 |

`colamd` is the default because it suits unsymmetric matrices; `amd` is the
better choice when the pattern is symmetric, as here.

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
