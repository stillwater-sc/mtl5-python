# GAP Analysis — mtl5-python bindings vs. MTL5 `c6ff006` (audited 2026-08-01)

**Analyst:** automated survey of `mtl5-python@968c602` against `mtl5@c6ff006`
**Bindings source of truth:** `python/src/mtl5_module.cpp` (1338 LOC), `mtl5/__init__.py`, `mtl5/sparse/__init__.py`, `mtl5/pandas_ext.py`
**MTL5 source of truth:** `include/mtl/**` at `c6ff006` (`main` as of 2026-08-01), `CHANGELOG.md`, `git tag`

Every statement below is against that revision. `main` moves, so the commit is
the reproducible reference.

> **Status, 2026-08-04.** This document is the original audit, kept as written so
> the findings and the reasoning behind the sequencing stay legible. §2 and §4
> have been brought up to date; §3 carries per-subsection updates inline. Every
> phase in §5 is merged, and every gap found here is closed or tracked. The
> five defects this work surfaced upstream — MTL5
> [#352](https://github.com/stillwater-sc/mtl5/issues/352),
> [#353](https://github.com/stillwater-sc/mtl5/issues/353),
> [#355](https://github.com/stillwater-sc/mtl5/issues/355),
> [#359](https://github.com/stillwater-sc/mtl5/issues/359),
> [#379](https://github.com/stillwater-sc/mtl5/issues/379) — are all fixed.

---

## 1. Executive summary

The bindings were last extended on **2026-04-14** against MTL5 as of **2026-04-06**
(`build/_deps/mtl5-src` is pinned at `67d81d7`). Since then MTL5 has landed **165 commits
and five minor releases** (`v5.2.1` → `v5.7.0`). `pyproject.toml` still declares
`version = "5.2.0"`, and the project's own versioning policy says the minor component
tracks MTL5 upstream — so the declared version is **five minor releases stale**.

Three findings, in order of impact:

1. **No compile breakage in the default-feature build.** `mtl5_module.cpp` still compiles
   clean (`g++ -std=c++20 -fsyntax-only`, exit 0, warnings only about symbol visibility)
   against `mtl5@c6ff006`. That is the extent of the evidence: it says nothing about runtime
   behaviour, the Python surface, or the BLAS/LAPACK/Highway-gated code paths, none of which
   this check compiled. On that basis the gap looks like **unexposed surface rather than
   drift**, and catching up looks additive.

2. **The strategic differentiator is 0% exposed.** MTL5's mixed-precision layer — epic #157's
   `accumulator_traits`, `mtl::convert`, the accumulator policies on `dot`/`gemm`/`gemv`/norms,
   `lu_iterative_refine`, `normwise_backward_error`, and `sparse::iterative_refine` — has **no
   Python surface at all**. The bindings' Universal path is copy-convert only: a `posit32`
   matrix is stored *and* accumulated in `posit32`, so the quire/wide-accumulator accuracy
   result that motivates the number systems is unreachable from Python. This is the single
   largest gap relative to the project's purpose.

3. **The wheel ships MTL5's slowest configuration.** `python/CMakeLists.txt` links `MTL5::mtl5`
   without enabling any acceleration option. All of `MTL5_NATIVE_FAST_GEMM`, `MTL5_WITH_BLAS`,
   `MTL5_WITH_LAPACK`, and `MTL5_WITH_HIGHWAY` default to `OFF` in MTL5's CMakeLists
   (lines 47–53), so:
   - `mtl::mult` never reaches the blocked GEMM / SIMD path (`operation/mult.hpp:17,217,278,345`
     are all `#ifdef MTL5_NATIVE_FAST_GEMM`) — the entire #82/#99–#107 GEMM epic, and the
     10–16× SIMD-widening GEMM from #176, are compiled out.
   - `get_backend()` can only ever return `"reference"`; the `MTL5_HAS_BLAS` / `MTL5_HAS_KPU`
     branches in `mtl5_module.cpp:1267–1290` are dead code as built.
   - Threading *is* available (runtime-gated on `MTL5_NUM_THREADS`, `detail/thread_pool.hpp:36`)
     but is undocumented and unexposed in Python, so it stays serial unless the caller happens
     to know the environment variable. Separately, no binding releases the GIL (zero
     `nb::gil_scoped_release` in the module): with `MTL5_NUM_THREADS > 1` the kernels *do*
     run in parallel on MTL5's native worker threads, but the calling Python thread holds the
     GIL for the whole call, so every other Python thread is blocked until it returns.

---

## 2. What is bound today

Updated 2026-08-04. The audit-time state is in the second column, for contrast.

| Area | Bound now | At the audit |
|---|---|---|
| Dense containers | `dense_vector<T>`, `dense2D<T>` over **16** element types | f32/f64/i32/i64 |
| Universal containers | fp8, fp16, posit8/16/32/64, fixpnt8/16, lns16/32 | same (storage only) |
| Complex containers | c64, c128, zero-copy over NumPy complex | — |
| Sparse containers | CSR, **COO**, **ELL**, f32/f64 | CSR only |
| N-D arrays | `mtl5.array`: ranks 1–4, zero-copy, slicing, reductions | — |
| Dense ops | norms, `dot`/`dot_real`, `matmul`, `matvec`, `transpose`/`adjoint`/`conj`, `det`, `inv`, `solve` | norms, `dot`, `matmul`, `matvec`, `transpose`, `det`, `inv`, `solve` |
| Dense factorizations | `lu`, `cholesky`, `qr`, `lq`, `ldlt`, `bunch_kaufman` | `lu`, `cholesky` |
| Eigen / SVD | `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, `svdvals`, `condition_number`, `rcond`, `numerical_rank`, `nullity` | — |
| BLAS L2/L3 | `ger`, `symm`, `symv`, `syr2k`, `syrk`, `trmm`, `trmv`, `trsm`, `trsv` | — |
| Property predicates | 28, from `is_spd` to `inertia` and `spectral_radius` | — |
| Mixed precision | `accumulator=` on `dot`/`norm`/`frobenius_norm`/`matvec`/`matmul`, `convert`, `lu_iterative_refine`, `backward_error`, `iterative_refine` | — |
| Sparse direct | `splu`, `klu`, `cholesky`, `ldlt`, `qr`, `supernodal_lu`, `supernodal_ldlt`, each with `accumulator=` | — |
| Sparse ordering | `amd`, `colamd`, `rcm`, `natural` via `ordering`/`orderings` | — |
| Sparse ops | `matvec`, **`rmatvec`**, `tocoo`, `to_csr_arrays`, SciPy interop | `matvec`, `to_csr_arrays`, SciPy interop |
| Krylov | `cg`, `gmres`, `bicgstab` | same |
| Preconditioners | `ilu0`, `ic0` | same |
| I/O | `mm_read`/`mm_read_dense`/`mm_write`/`mm_write_sparse`, `spy`/`spy_magnitude`/`spy_density` | — |
| Generators | 25, incl. `randspd`/`randsym`/`randsvd` with prescribed spectra | — |
| Ecosystem | pandas `ExtensionDtype` for posit16 | same |
| Infra | `build_info`, threading, honest `get_backend`/`set_backend` | stubs |

Roughly **14 numerical entry points** at the audit; **over 150** now.

---

## 3. Gap inventory

### 3.1 Mixed precision — 9 of ~10 exposed (Phase 1, #23; accumulator policy on the sparse factorizations, #32)

The differentiator. `mtl/math/accumulator_traits.hpp`, `operation/convert.hpp`,
`operation/lu_iterative_refine.hpp`, `operation/backward_error.hpp`, `sparse/iterative_refine.hpp`.

| MTL5 | Status |
|---|---|
| `math::accumulator_traits<Acc, Value>` — element / accumulate / result precision triple | ✅ `accumulator=` / `result=` |
| accumulator policy on `dot`, `dot_real` (#159) | ✅ `mixed.dot` |
| accumulator policy on `mult`/`gemm`, result type from `C` (#161) | ✅ `mixed.matmul` |
| accumulator policy on `gemv` (#160), `two_norm`/`frobenius_norm` (#162) | ✅ `mixed.matvec`, `mixed.norm`, `mixed.frobenius_norm` |
| `math::fma_accumulator` (#259) | ✅ `accumulator='fma32'`/`'fma64'` |
| `mtl::convert` — element-wise re-quantization (#164) | ✅ `mtl5.convert` |
| `lu_iterative_refine<Working>(A, b, x, opt)` (#273) | ✅ `mixed.lu_iterative_refine` |
| `normwise_backward_error(A, x, b)` (#273) | ✅ `mixed.backward_error` |
| `sparse::iterative_refine` — refinement through any factorization (#119, #167) | ✅ `mixed.iterative_refine` |
| `cg` accumulator policy — posit32+quire accuracy gain (#238) | ❌ still unbound — see §3.7 |

The quire is composed at this layer rather than in MTL5: `accumulator_traits` ships the
contract, and the specialization for Universal's quire lives in
`python/include/mtl/math/quire_accumulator.hpp`.

The norms went through local sum-of-squares loops until MTL5
[#379](https://github.com/stillwater-sc/mtl5/issues/379) gave `two_norm`/`frobenius_norm`
the `Result` parameter `dot<Accumulator, Result>` already had — without it the accumulated
value was rounded back to the element type, and `accumulator=` was unobservable from
Python. Filed from this work, fixed upstream, loops deleted in #36.

**Resolved:** the headline claim `mult<float>(A_bf16, B_bf16, C_bf16)` (store narrow,
accumulate wide) now has a Python equivalent. `posit32`+quire in CG does not — the Krylov
accumulator policy is the one item in this table still open.

### 3.2 Sparse direct solvers — 7 of ~9 exposed (Phase 2, #24)

`mtl/sparse/factorization/*`. Python users must round-trip to SciPy's SuperLU/UMFPACK,
which cannot do mixed precision.

`sparse_lu_{symbolic,numeric,solve,refactor}`, `sparse_cholesky_*`, `sparse_ldlt_*`,
`sparse_qr_*`, `supernodal_lu_{symbolic_analyze,numeric,refactor,solve,solve_refined}`,
`supernodal_ldlt_*`, `native_klu_{factor,refactor,solve}`, `triangular_solve`,
`level_schedule` — all unbound.

The **analyze/factor/refactor** split (#153/#154/#184, 1.9–3.2× on same-pattern refactor) is
exactly the SPICE-transient workflow, and it is the one thing SciPy structurally cannot offer.

**Orderings and analysis, also unbound:** `amd`, `colamd`, `rcm`, `minimum_degree`,
`dulmage_mendelsohn` / `block_triangular_form`, `elimination_tree`, `column_elimination_tree`,
`tree_postorder`, `find_supernodes`.

### 3.3 Eigenvalue / SVD — 10 of ~10 exposed (Phase 3, #26, after MTL5 #337 was fixed)

Nothing in the eigen/SVD family reaches Python, despite epic #202 completing in July.

`eigenvalue`, `eigen`, `eigenvalue_symmetric`, `eigen_symmetric`,
`eigenvalue_symmetric_generic`, `svd`, `singular_values`;
matrix-free `itl::power_iteration`, `itl::lanczos`, `itl::arnoldi`;
`sparse_eigs`, `sparse_eigs_shift_invert`, `shift_invert_operator`.

This is a natural `numpy.linalg` / `scipy.sparse.linalg.eigs` mapping and the absence is
conspicuous for any ecosystem-facing package.

### 3.4 Dense factorizations beyond LU/Cholesky — QR/LQ/LDLᵀ/Bunch–Kaufman bound (#25, #26); primitives still unbound

`qr_factor`/`qr_solve`, `lq_factor`, `ldlt_factor`/`ldlt_solve`,
`ldlt_bk_factor`/`ldlt_bk_solve` (Bunch–Kaufman, for indefinite systems),
`hessenberg`/`hessenberg_factor`/`tridiagonalize`, `householder`, Givens rotations.

Note: open issue #18 (mixed-precision UKF, Cholesky vs LDLᵀ) was blocked on `ldlt` bindings and
is unblocked — `ldlt` and `cholesky` are both available for float32/float64 and all ten
Universal dtypes, which is what makes that comparison possible.

### 3.5 BLAS Level 2/3 and elementwise — 12 of ~22 exposed (Phase 3, #26)

Only GEMM (`matmul`) and GEMV (`matvec`) are bound. Missing, all landed in #229–#232:
`ger`, `symv`, `trmv`, `trsv`, `trmm`, `trsm`, `symm`, `syrk`, `syr2k`.
Also unbound: `axpy`, `scale`, `sum`, `product`, `min`, `max`, `trace`, `kron`,
`diag`/`diagonal`, `reorder`/`reorder_rows`/`reorder_cols`, `fill`, `set_to_zero`,
`random_matrix`/`random_vector`, `project_onto`/`embed_into`/`saturating_cast`,
and the transcendental family (`exp`, `log`, `sqrt`, `sin`, … ~25 functions).

### 3.6 Property predicates — 28 of ~30 exposed (Phase 3, #26)

The whole #244 module (`matrix_properties.hpp`, `vector_properties.hpp`,
`factorization_properties.hpp`, `spectral_properties.hpp`, `tensor/properties.hpp`):

`is_square`, `is_empty`, `is_symmetric`, `is_hermitian`, `is_upper/lower_triangular`,
`is_triangular`, `is_diagonal`, `is_banded`, `is_diagonally_dominant`, `is_orthogonal`,
`is_unitary`, `is_normal`; `is_zero`, `is_finite`, `has_nan`, `has_inf`, `is_normalized`,
`is_unit`, `is_orthogonal_to`; `is_spd`, `is_positive_definite`, `is_singular`,
`is_nonsingular`, `is_invertible`, `determinant`; `spectral_radius`, `condition_number`,
`rcond`, `numerical_rank`, `nullity`, `inertia`, `is_indefinite`.

Cheap to bind, and `condition_number`/`rcond`/`numerical_rank` map directly onto
`np.linalg.cond` / `np.linalg.matrix_rank` expectations.

### 3.7 Iterative solvers — 10 of 10 Krylov, 8 of 8 preconditioners (#38)

**Missing Krylov:** `bicg`, `bicgstab_ell`, `cgs`, `idr_s`, `minres`, `qmr`, `tfqmr`.
**Missing preconditioners:** `diagonal` (Jacobi), `block_diagonal`, `ildl`, `ilut`, `ssor`,
and `identity` — which the Krylov bindings construct internally but never hand to Python,
so it cannot be used to run an unpreconditioned solve through a preconditioned entry point.
**Missing smoothers (entire module):** `jacobi`, `gauss_seidel` + backward/symmetric,
`sor` + backward/symmetric.
**Missing multigrid (entire module):** `multigrid`, `prolongation`, `restriction`.

Also note the Krylov solvers are exposed as private `_sparse_cg`/`_sparse_gmres`/`_sparse_bicgstab`
with a Python wrapper — a pattern that will not scale to 10 solvers × 9 preconditioners; a
policy-object dispatch is warranted before expanding.

### 3.8 Containers, views, expressions — COO and ELL bound (#30); views still unbound

**Matrix/vector types:** `coordinate2D` (COO), `ell_matrix` (ELL), `block_diagonal2D`,
`identity2D`, `permutation_matrix`, `sparse_vector`, `unit_vector`, `strided_vector_ref`,
and the inserter API (`compressed2D_inserter`, `shifted_inserter`).

**Views:** `transposed_view`, `lower_view`, `upper_view`, `strict_lower_view`,
`strict_upper_view`, `banded_view`, `hermitian_view`, `map_view`.

**Storage formats** — COO and ELL bound in Phase 4d. CSC is not, and the reason is
upstream rather than a binding decision.

| format | MTL5 | bound |
|---|---|---|
| CSR | `compressed2D` | ✅ since Phase 2 |
| COO | `coordinate2D` | ✅ `SparseMatrixCOO_*` — duplicates accumulate, matching scipy |
| ELL | `ell_matrix` | ✅ `SparseMatrixELL_*` — plus `padding_ratio`, the fit metric |
| CSC | — | ❌ not implemented |

**`compressed2D`'s orientation parameter was inert; it is now rejected.** It is templated
on `tag::row_major` | `tag::col_major` (`mat/parameter.hpp:14`), but nothing read the tag:
the inserter, `operator()` and `mult` all treated the storage as row-major. Measured at
`c666ae4` on a 2×3 matrix, a `col_major` `compressed2D` built a `major` array of length 3
(nrows + 1) where CSC needs 4 (ncols + 1); a `row_major` and a `col_major` instance built
from the same elements had byte-identical `major`/`minor`/`data`; and genuine scipy CSC
arrays produced the transpose while reporting success. Filed as
[stillwater-sc/mtl5#355](https://github.com/stillwater-sc/mtl5/issues/355) and fixed
upstream in MTL5 `3ac75bd`, which takes the smaller of the two options offered and
`static_assert`s the orientation to `row_major` — so the silent wrong answer is now a
compile error, but there is still no CSC container. `from_scipy` continues to convert CSC
input to CSR at the boundary.

What CSC was wanted for in practice — a transpose-SpMV so a `LinearOperator`'s `rmatvec`
need not round-trip through scipy — is now `SparseMatrix.rmatvec`, built on `mtl::trans`,
which *is* correct for sparse (verified on a non-symmetric 2×3).

**ELL has no element setter.** `ell_matrix(nrows, ncols, max_width)` produces an
all-padding matrix with no way to fill it, so the CSR constructor is the only usable
entry point and the only one bound.

**Complex scalars** — bound in Phase 4c, but only where MTL5 actually supports them. The
boundary was established by compiling each operation against `std::complex<double>` and then
checking the answer, not just the compile:

| | complex |
|---|---|
| containers, factories, `to_numpy`, `.real`/`.imag` | ✅ |
| `one_norm` / `two_norm` / `infinity_norm` / `frobenius_norm` | ✅ (returns a real magnitude) |
| `dot` (Hermitian, = `np.vdot`), `dot_real` (= `np.dot`) | ✅ |
| `mult` — matvec, matmul, sparse matvec | ✅ |
| `lu_factor`/`lu_solve`, `inv` | ✅ |
| `trans` (**plain** transpose — does not conjugate) | ✅ |
| `conj`, `is_hermitian`, `is_symmetric` | ✅ |
| `ldlt` / `ldlt_h` | ✅ both, dispatched on the matrix |
| `qr`, `lq` (Householder, incl. least squares) | ✅ since MTL5 `aa3b52c` |
| `cholesky` (Hermitian, A = L·Lᴴ) | ✅ since MTL5 `95270f7` |
| `eigenvalue`, `svd` | ❌ would need `complex<complex>` |
| `cg` / `gmres` / `bicgstab` and the preconditioners | ❌ Krylov layer is real-only |

Three gaps were found here and all three were fixed upstream:

* **`ldlt` was silently wrong on Hermitian input** — `operation/ldlt.hpp` had no
  conjugation anywhere, so it computed A = L D Lᵀ and returned `info = 0` on Hermitian
  input too. Measured on `[[2, 1-i], [1+i, 3]]`: `x = [1.9+0.3i, -0.2+0.6i]` where the
  answer is `[1, i]`. Filed as
  [stillwater-sc/mtl5#352](https://github.com/stillwater-sc/mtl5/issues/352) and fixed in
  MTL5 `95270f7`, which added `ldlt_h_factor`/`ldlt_h_solve` (A = L D Lᴴ) — the most
  ambitious of the three resolutions offered, since it adds the capability rather than only
  refusing the input. Re-verified: the reproducer now returns `[1, i]` exactly, and
  non-Hermitian input gets `LDLT_NOT_HERMITIAN` (-2). `mtl5.ldlt_solve` dispatched on the
  matrix from that point on, replacing the refusal it had shipped with.
* **`qr`/`lq` did not compile** — filed as
  [stillwater-sc/mtl5#353](https://github.com/stillwater-sc/mtl5/issues/353) and implemented
  in MTL5 `aa3b52c` ("complex Householder, QR and LQ"). Re-verified for the answer, not the
  compile: on a 4×2 complex matrix ‖QR − A‖ = 1.4e-15, ‖QᴴQ − I‖ = 4.4e-16, and `qr_solve`'s
  least-squares residual is orthogonal to range(A) to 3.5e-15 — which is what confirms it
  applies Qᴴ rather than Qᵀ. Bound as `mtl5.qr` / `mtl5.lq`.
* **`cholesky` did not compile either**, and #353 was initially closed on the QR half only.
  MTL5 `95270f7` added `cholesky_h_factor`/`cholesky_h_solve` (A = L·Lᴴ) and made
  `cholesky_factor` `static_assert` against complex with a message naming them. Re-verified:
  err 2.29e-16 on the Hermitian test matrix, ‖L·Lᴴ − H‖ = 4.44e-16, and a non-real diagonal
  reports `CHOLESKY_NOT_HERMITIAN` (-2) rather than a definiteness failure. `mtl5.cholesky`
  routes complex there; the real path is untouched, since it also serves the Universal dtypes.

Both Hermitian variants also accept real symmetric input, so they are supersets rather than
parallel paths — but the real bindings still use the original entry points, which are the
ones instantiated for the Universal number systems.

`hermitian_view` exists but has no binding.

### 3.9 Tensor / N-D array layer — `mtl/array` bound in Phase 4e; `mtl/tensor` still P2

**`mtl/array/*` — bound.** `mtl5.array` exposes `ndarray` at ranks 1–4 for float32/float64:
zero-copy `asarray` from any strided NumPy layout, `to_numpy` that carries the strides back,
element access, `transpose`, `reshape`, NumPy-style slicing, the whole-array reductions
(`sum`/`prod`/`mean`/`min`/`max`), `sum_axis`/`mean_axis`, elementwise arithmetic, and
`as_ndarray`/`as_matrix`/`as_vector` interop with the dense containers.

Rank is a template parameter (`ndarray<T, N, Order>`), so the ranks have to be fixed at build
time; 1–4 are instantiated and rank 5+ raises rather than silently flattening.

Four things did not come across, each for a stated reason:

* **`ndarray::reshape` was unsound for a non-C-contiguous array — fixed upstream.** It
  guarded on `is_contiguous()`, meaning "contiguous in *either* C or F order"
  (`shape.hpp:158`), then rebuilt with the array's own `Order`; a transposed C-order array is
  F-contiguous, so it passed the guard and was read back with C-order strides. Measured:
  `[[1,2,3],[4,5,6]].transpose().reshape(6)` returned `[1,2,3,4,5,6]` where NumPy gives
  `[1,4,2,5,3,6]`, with no error. Filed as
  [stillwater-sc/mtl5#359](https://github.com/stillwater-sc/mtl5/issues/359) and fixed — it
  now tests own-order contiguity and throws on the case it cannot alias. The binding still
  does not call it, for a different reason: throwing is right for a C++ caller but not
  NumPy's contract, where reshape never errors. So `mtl5.array` computes it — a view when it
  can alias, a copy otherwise.
* **`mtl::array::slice` is unreachable from Python.** It derives the result rank at compile
  time from the argument *types* (`count_kept_dims<Args...>`), so a runtime index tuple would
  need every {int, range, all} combination instantiated — 3^N per rank per dtype. `__getitem__`
  computes the shape, strides and base offset itself and builds the view through ndarray's
  (pointer, shape, strides) constructor, the same primitive `slice` uses.
* **Broadcasting is same-rank only.** `broadcast_shape` takes two `shape<N>` of equal `N`, so
  an extent of 1 stretches but NumPy's rank promotion does not. Reshape the operand first.

* **`flatten` was wrong on an F-contiguous array, from the same root cause — also fixed.** It
  walked elements via `for_each_element`, whose fast path triggered on the same
  `is_contiguous()` and then indexed raw memory: flatten of the transpose of
  `[[1,2,3],[4,5,6]]` gave `[1,2,3,4,5,6]` where NumPy gives `[1,4,2,5,3,6]`. Part of #359 and
  now correct. Worth recording, because it kept the fix narrow: the commutative reductions
  share that code path but were never affected — `sum`/`prod`/`min`/`max`/`mean` do not care
  what order they visit in — and `sum_axis`/`mean_axis` index explicitly and were correct on
  strided input throughout. `mtl5.array` exposes `ravel` (a view when possible) and `flatten`
  (always a copy), splitting NumPy's two behaviours, which upstream has no equivalent of.

`transform`/`reduce` are not bound: they take C++ callables, so exposing them would mean a
Python call per element — `to_numpy()` is the better answer.

**`mtl/tensor/*` — still unbound, P2.** Rank-N `tensor`, `Index`/`contract`/`outer`/`bind`,
metric raise/lower, symmetric/antisymmetric storage. Distinct from `mtl/array`: index-notation
tensor algebra rather than NumPy-shaped N-D data.

### 3.10 I/O — 7 of ~8 exposed (Phase 4b, #28)

`io::mm_read`/`mm_write` (Matrix Market, with gzip and direct-CRS large-file loading, #197),
`read_el`/`write_el`, and the visualization stack `spy`/`spy_density`/`spy_magnitude`/`spy_grid`
plus the from-scratch PNG writer (#252/#253/#257).

`spy` → PNG with no matplotlib dependency is a genuinely attractive Python feature.

### 3.11 Test-matrix generators — 25 of ~28 exposed (Phase 4a, #27)

`clement`, `companion`, `forsythe`, `frank`, `hilbert`, `kahan`, `laplacian_1d/2d`, `lehmer`,
`lotkin`, `magic`, `minij`, `moler`, `ones`, `pascal`, `poisson2d_dirichlet`, `randorth`,
`randspd`, `randsvd`, `randsym`, `rosser`, `vandermonde`, `wilkinson`; the `testsuite` catalog
(`by_name`, `names`, `kappa_table`); and the range family `arange`/`linspace`/`logspace`/`geomspace`.

Lowest effort-to-value ratio in this document: pure factory functions, no lifetime concerns,
and they immediately improve the bindings' own test suite (which currently hand-rolls matrices).

### 3.12 Build, threading, and dispatch — done (Phase 0, #22)

| Issue | Evidence |
|---|---|
| No acceleration options enabled in the wheel | `python/CMakeLists.txt` — no `MTL5_WITH_*`, no `MTL5_NATIVE_FAST_GEMM` |
| `get_backend()` structurally cannot report anything but `"reference"` | `mtl5_module.cpp:1281–1290` vs. MTL5 `CMakeLists.txt:125–135` |
| `set_backend()` validates the name but never selects anything | `mtl5_module.cpp:1291` — rejects unknown names and `"kpu"`, then no-ops for `"cpu"`/`"reference"`/`"blas"`, so it reports success for a backend the build lacks |
| Threading unexposed and undocumented | `MTL5_NUM_THREADS` read in `detail/thread_pool.hpp:36`; no Python accessor |
| GIL never released | zero `nb::gil_scoped_release` in the module |
| External solver interfaces unbound | `interface/{umfpack,superlu,klu,cholmod,spqr,blas,lapack}.hpp` |

The two threading rows compound. #221/#297 made every substantial kernel parallel, but the
pool is serial unless `MTL5_NUM_THREADS` is set, and the Python surface neither exposes nor
documents that — so the default is serial. Turning it on gets native parallelism inside the
kernel, yet because no binding releases the GIL the rest of the interpreter is still blocked
for the duration of the call. Two independent fixes: expose the thread count, and release
the GIL.

---

## 4. Coverage scorecard

The unit differs by row and is named in the **Unit** column — a container row counts
class templates, an operations row counts namespace-scope functions, and the build row
counts CMake options. Mixing them in one number would be meaningless, so compare
percentages only within a row. Counts marked `~` are approximate; see §6.

Updated 2026-08-04. **Available** is held at the 2026-08-01 baseline so the
percentages stay comparable to the original audit; upstream has grown since, and
where that matters it is named in the notes. **Was** is the audit-time figure.

| Module | Unit | Was | Exposed | Available | Coverage |
|---|---|---|---|---|---|
| Dense containers | class templates | 2 | 2 | 2 | ✅ full — now over 16 element types |
| Universal number types | instantiated types | 10 | 10 | 10 | ✅ full — storage, LDLᵀ and Cholesky |
| Complex element types | instantiated types | — | 2 | 2 | ✅ full (c64, c128) |
| Mixed-precision / accumulator | functions | 0 | 9 | ~10 | ✅ ~90% |
| Dense factorizations | functions | 2 | 10 | 10 | ✅ full |
| Eigen / SVD | functions | 0 | 10 | ~10 | ✅ full |
| BLAS L2/L3 + elementwise | functions | 2 | 12 | ~22 | ⚠️ 55% |
| Property predicates | functions | 0 | 28 | ~30 | ✅ 93% |
| Sparse containers | class templates | 1 | 3 | 6 | ⚠️ 50% (CSR, COO, ELL) |
| Sparse direct solvers | functions | 0 | 7 | ~9 | ✅ 78% |
| Sparse ordering / analysis | functions | 0 | 5 | ~9 | ⚠️ 56% |
| Krylov solvers | solvers | 3 | 10 | 10 | ✅ full |
| Preconditioners | preconditioners | 2 | 8 | 8 | ✅ full |
| Smoothers / multigrid | functions | 0 | 12 | 12 | ✅ full |
| Views / expressions | class templates | 0 | 8 | ~8 | ✅ full |
| `mtl/array` N-D layer | functions | 0 | 5 | ~8 | ✅ 63% |
| `mtl/tensor` | functions | 0 | 0 | ~7 | ❌ 0% |
| I/O | functions | 0 | 7 | ~8 | ✅ 88% |
| Generators | generators | 0 | 25 | ~28 | ✅ 89% |
| Build acceleration | CMake options | 0 | 5 | 5 | ✅ full |

The original **Tensor / ndarray** row bundled `mtl/array` and `mtl/tensor` at ~15
functions; they are split here, because one is bound and the other is untouched.

Notes on the rows that are easy to miscount:

- **Universal number types** — the ten instantiated element types listed in §2
  (`fp8`, `fp16`, `posit8/16/32/64`, `fixpnt8/16`, `lns16/32`), not the twelve dtype
  strings you get by adding `f32`/`f64`. Complex is a separate row: `c64`/`c128` are
  not in `mtl5.dtypes()`, since that lists what `convert()` accepts and there is no
  complex target.
- **Dense factorizations** — the factor/solve entry points for LU, Cholesky, QR, LQ,
  LDLᵀ and Bunch–Kaufman, all now bound. The lower-level primitives §3.4 also lists —
  `hessenberg`, `tridiagonalize`, `householder`, Givens rotations — remain unbound and
  were not in this count.
- **Sparse containers** — CSR, COO and ELL of the six in §3.8. CSC is not a seventh
  container: it was `compressed2D` under `tag::col_major`, which upstream now rejects
  at compile time (#355). The rest are `block_diagonal2D`, `identity2D` and
  `permutation_matrix`.
- **Sparse ordering** — AMD, COLAMD, RCM and natural, reachable through `ordering()`
  and enumerated by `orderings()`. The unexposed remainder is the symbolic-analysis
  surface (elimination trees, postordering) rather than orderings as such.
- **BLAS L2/L3** — the nine named BLAS routines plus `matmul`, `matvec` and
  `transpose`. The gap to ~22 is elementwise and rank-k variants, not core BLAS.
- **`mtl/array`** — `asarray`, `zeros`, `as_ndarray`, `as_matrix`, `as_vector` as free
  functions; slicing, reshape, ravel/flatten, transpose and the reductions are methods
  on the array class and are not counted separately. `transform`/`reduce` are excluded
  by choice: they take C++ callables, so binding them would cost a Python call per
  element.
- **Preconditioners** — the eight in `itl/pc/` excluding `solver.hpp`, which is a
  dispatch helper rather than a preconditioner. `identity` is constructed internally by
  the Krylov bindings but is not exposed as a Python object, so it counts as unexposed.
- **Smoothers / multigrid** — twelve functions across six headers: `jacobi`;
  `gauss_seidel` + backward + symmetric; `sor` + backward + symmetric; `multigrid`,
  `make_prolongation_1d`, `prolongate`, `make_restriction_1d`, `restrict`.
- **Build acceleration** — `MTL5_NATIVE_FAST_GEMM`, `MTL5_WITH_BLAS`, `MTL5_WITH_LAPACK`,
  `MTL5_WITH_HIGHWAY`, `MTL5_NATIVE_ARCH`; all five are now declared at top level.
  `MTL5_WITH_ZLIB` was added afterwards for gzip Matrix Market input and is a sixth,
  outside the audited five.

**What is left.** One module: `mtl/tensor` — index-notation tensor algebra, distinct from
the NumPy-shaped `mtl/array` that is bound. Everything else this audit inventoried is
reachable from Python.

Views were bound in #42. They materialise rather than aliasing: upstream each holds a
`const Matrix&`, which is a lifetime hazard for a caller passing a temporary — the same
shape as the bug that segfaulted `pc::ssor` — and the laziness buys nothing without an
expression template to feed. Two carry traps worth knowing and are guarded: `banded` takes
bandwidths rather than signed diagonal offsets, and `hermitian` reconstructs a Hermitian
matrix from the upper triangle rather than being the adjoint.

Smoothers and multigrid were bound in #40, with two additions upstream does not have: a
sparse `galerkin(R, A, P)`, because `R * A * P` through `operator*` returns a *dense*
matrix and the coarse operator is exactly what must stay sparse; and `multigrid_1d`,
which builds a consistent hierarchy rather than asking the caller to supply levels,
restrictors, prolongators, a smoother factory and a coarse solver separately. Measured at
~0.03 residual reduction per V-cycle on 1-D Poisson, holding as n grows.

The cross-cutting item is done: §3.7's note called for refactoring the binding pattern to
a solver/preconditioner dispatch *before* expanding from 3×2 to 10×8, and #38 did exactly
that. Type-erasing the preconditioner made it 20 instantiations rather than 160. The sweep
that preceded the design also turned up
[stillwater-sc/mtl5#392](https://github.com/stillwater-sc/mtl5/issues/392) — `bicg`
computed `rho` from the wrong vectors and worked only with the identity preconditioner —
which was fixed upstream before the binding was built on it.

One limit remains, and it is upstream: every MTL5 preconditioner implements
`adjoint_solve` as its forward solve, which is exact for a symmetric preconditioner and
wrong otherwise. `bicg` and `qmr` are the only solvers that apply Mᵀ, so they refuse a
preconditioner measured to be non-symmetric rather than break down.

---

## 5. Recommended sequencing

All five phases are merged. The plan is kept as written, annotated with what
shipped, because the reasoning about ordering is the part worth rereading.

**Phase 0 — build configuration (days, no new numerical bindings)** — merged, mtl5-python#22
Turn on `MTL5_NATIVE_FAST_GEMM`; add opt-in `MTL5_WITH_BLAS`, `MTL5_WITH_LAPACK` and
`MTL5_WITH_HIGHWAY` via `[tool.scikit-build.cmake.define]` — MTL5 declares them under
exactly those `MTL5_`-prefixed names, and an unprefixed `WITH_LAPACK` would silently
enable nothing.

That is four of the five options in the §4 scorecard. The fifth,
`MTL5_NATIVE_ARCH`, is deliberately **out of scope for Phase 0**: it adds
`-march=native`, so the resulting wheel only runs on machines resembling the one that
built it. It should be available to someone building locally for their own hardware,
but never on by default in a distributed artifact.

Make `get_backend()` truthful, and `set_backend()` honest. Today it does validate its
argument — unknown names raise, and `"kpu"` raises "not yet available" — but it accepts
`"cpu"`, `"reference"` and `"blas"` and then does nothing, including on a build with no
BLAS linked. So `set_backend("blas")` succeeds on a wheel that cannot use BLAS. It should
either validate against what was actually compiled in, or not be public.

Add `set_num_threads()`/`get_num_threads()` — the only new bindings in this phase, and
infrastructure rather than numerics. Release the GIL around every kernel that can run
longer than a few microseconds. Sync `pyproject.toml` to `5.7.x`. This is the highest
performance-per-hour item in the document — MTL5's entire 2026 performance program is
currently compiled out of the wheel.

**Phase 1 — mixed precision (the differentiator)** — merged, mtl5-python#23
`convert`, an accumulator-policy parameter on `dot`/`matmul`/`matvec`/`norm`,
`lu_iterative_refine` + `normwise_backward_error`, `sparse::iterative_refine`.
This is what makes the package worth using over NumPy/SciPy.

**Phase 2 — sparse direct solvers** — merged, mtl5-python#24; the accumulator policy that
completes it landed later in #32
`sparse_lu` and `native_klu` with the analyze/factor/refactor split exposed as a Python
factorization object; then supernodal LU/LDLᵀ, `sparse_cholesky`, `sparse_qr`, and orderings.
Pair with mixed-precision refinement from Phase 1 — that combination has no SciPy equivalent.

**Phase 3 — dense completeness** — merged, mtl5-python#25 and #26
QR/LQ/LDLᵀ/Bunch–Kaufman (unblocks #18), eigen/SVD family, BLAS L2/L3, property predicates.

**Phase 4 — ecosystem depth** — merged in five parts: #27 generators, #28 Matrix Market and
spy, #29 complex scalars, #30 COO/ELL, #31 the `mtl/array` layer
Generators + `linspace` family (quick win, improves own tests), Matrix Market I/O, `spy`
visualization, complex scalar support, COO/ELL containers, the `mtl/array` N-D layer.
CSC is blocked upstream — see §3.8.

**Cross-cutting — still open.** Refactor the Krylov binding pattern to a solver/preconditioner dispatch
before expanding from 3×2 to 10×9 combinations.

---

## 6. Method and caveats

- Coverage determined by enumerating `nb::class_` / `m.def` in `mtl5_module.cpp` and the
  `__all__` lists in `mtl5/__init__.py` and `mtl5/sparse/__init__.py`, then diffing against
  namespace-scope function declarations across `include/mtl/**`.
- Compile check: `g++ -std=c++20 -fsyntax-only` on `mtl5_module.cpp` against
  `include/mtl` at `c6ff006`, with nanobind and Universal headers from the existing
  `build/_deps` tree. Exit 0; only `-Wattributes` visibility warnings. It was run **without**
  MTL5's optional feature macros, so LAPACK/BLAS/Highway-gated code paths were not compiled;
  enabling those in Phase 0 needs its own build validation.
- "Available" counts use the unit named per row in §4 — class templates, functions,
  solvers, or CMake options — because a single unit across all of them would be
  meaningless. Internal `detail::` helpers are excluded. Counts marked `~` are approximate
  and intended for proportion, not precision; the unmarked ones are exact at `c6ff006`.
- Issue numbers in parentheses refer to **MTL5** issues unless stated as mtl5-python.
