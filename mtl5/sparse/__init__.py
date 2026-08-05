"""MTL5 sparse matrix operations — SciPy interop, iterative solvers, preconditioners.

This submodule provides:

- `SparseMatrix_f32` / `SparseMatrix_f64` — MTL5 CSR sparse matrices
- `SparseMatrixCOO_*` / `SparseMatrixELL_*` — the COO and ELL storage formats
- `from_scipy(sp)` — convert a scipy.sparse matrix to MTL5
- `to_scipy(A)` — convert any MTL5 sparse matrix back to scipy.sparse
- `csr_matrix(data, indices, indptr, shape)` — direct CSR construction
- `coo_matrix(row, col, data, shape)` — triplet construction; duplicates sum
- `ell_matrix(A)` — fixed-width layout, for near-uniform row widths
- `as_linear_operator(A)` — wrap an MTL5 sparse matrix as a SciPy
  `LinearOperator`, enabling use with scipy.sparse.linalg iterative solvers
- `accumulator=` on `splu`/`klu`/`supernodal_lu`/`supernodal_ldlt` — type the
  dense numeric workspace, so a float32 factor accumulates in float64. Measured
  on an ill-conditioned matrix (cond ~2.8e10) factored in float32, the forward
  error always improves; by how much varies with ordering and platform (roughly
  1.3x-3x). Paired with `mtl5.mixed.iterative_refine` it is 4 iterations to
  1.4e-10 instead of 6 to 2.0e-9. See the README.
- `cg`, `gmres`, `bicgstab` — iterative Krylov solvers returning (x, info)
- `ilu0`, `ic0` — incomplete LU/Cholesky preconditioners

scipy is an optional dependency: importing this module without scipy installed
yields the bare MTL5 SparseMatrix bindings, but the conversion helpers raise
ImportError when called.

**There is no CSC.** MTL5's `compressed2D` took an orientation parameter that
was inert — the inserter, element access and `mult` all treated the storage as
row-major whichever tag you passed. Upstream now rejects `col_major` at compile
time, but has no column-major container, so `from_scipy` converts a `csc_matrix`
to CSR at the boundary rather than pretending to hold it. The thing CSC is usually
wanted for here — a transpose product that does not round-trip through scipy —
is available directly as `A.rmatvec(x)`, which uses MTL5's own transpose view.
"""

from __future__ import annotations

import numpy as np

from mtl5._core import (
    BlockDiagonal_f32,
    BlockDiagonal_f64,
    Diagonal_f32,
    Diagonal_f64,
    IC0_f32,
    IC0_f64,
    Identity_f32,
    Identity_f64,
    ILDL_f32,
    ILDL_f64,
    ILU0_f32,
    ILU0_f64,
    ILUT_f32,
    ILUT_f64,
    KLU_f32,
    KLU_f64,
    Preconditioner_f32,
    Preconditioner_f64,
    SparseCholesky_f32,
    SparseCholesky_f64,
    SparseLDLT_f32,
    SparseLDLT_f64,
    SparseLU_f32,
    SparseLU_f64,
    SparseMatrix_f32,
    SparseMatrix_f64,
    SparseMatrixCOO_f32,
    SparseMatrixCOO_f64,
    SparseMatrixELL_f32,
    SparseMatrixELL_f64,
    SparseQR_f32,
    SparseQR_f64,
    SSOR_f32,
    SSOR_f64,
    SupernodalLDLT_f32,
    SupernodalLDLT_f64,
    SupernodalLU_f32,
    SupernodalLU_f64,
    _krylov,
    _ordering,
    orderings,
    preconditioners,
    solvers,
)
from mtl5._core import vector as _vector

try:
    import scipy.sparse as _sp
    from scipy.sparse.linalg import LinearOperator as _LinearOperator

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    _sp = None
    _LinearOperator = None


def _ensure_scipy() -> None:
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for mtl5.sparse interop helpers. Install with: pip install scipy"
        )


def _sparse_class_for_dtype(dtype: np.dtype):
    """Map a NumPy dtype to the corresponding MTL5 sparse class."""
    dt = np.dtype(dtype)
    if dt == np.float64:
        return SparseMatrix_f64
    if dt == np.float32:
        return SparseMatrix_f32
    raise TypeError(f"Unsupported sparse dtype: {dt}. Supported: float32, float64.")


def csr_matrix(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
):
    """Construct an MTL5 CSR sparse matrix directly from the three CSR arrays.

    The dtype of `data` selects which SparseMatrix_* class to instantiate.
    Indices/indptr are converted to int64 if not already.
    """
    data = np.ascontiguousarray(data)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    nrows, ncols = int(shape[0]), int(shape[1])
    cls = _sparse_class_for_dtype(data.dtype)
    return cls(nrows, ncols, indptr, indices, data)


def _coo_class_for_dtype(dtype: np.dtype):
    dt = np.dtype(dtype)
    if dt == np.float64:
        return SparseMatrixCOO_f64
    if dt == np.float32:
        return SparseMatrixCOO_f32
    raise TypeError(f"Unsupported sparse dtype: {dt}. Supported: float32, float64.")


def coo_matrix(
    row: np.ndarray,
    col: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
):
    """Construct an MTL5 COO matrix from scipy-style (row, col, data) triplets.

    Duplicate coordinates accumulate rather than overwrite, matching
    `scipy.sparse.coo_matrix`: the duplicates stay in the triplet list, `nnz`
    counts them separately, reading an element sums them, and `tocsr()` folds
    them into one entry.
    """
    data = np.ascontiguousarray(data)
    row = np.ascontiguousarray(row, dtype=np.int64)
    col = np.ascontiguousarray(col, dtype=np.int64)
    nrows, ncols = int(shape[0]), int(shape[1])
    cls = _coo_class_for_dtype(data.dtype)
    return cls(nrows, ncols, row, col, data)


def ell_matrix(A):
    """Convert a CSR matrix (MTL5 or scipy) to MTL5's ELL layout.

    ELL stores `nrows x max_width` slots regardless of how many are occupied,
    where `max_width` is the widest row's nnz. That regularity is what makes it
    vectorisable, and the padding is what it costs — check `padding_ratio`
    before committing to it. A single long row forces every other row to carry
    empty slots, and CSR is the better choice there.

    There is no incremental build path: MTL5's `ell_matrix` has no element
    setter, so CSR is the only way in.
    """
    A = _coerce_matrix(A)
    if isinstance(A, SparseMatrix_f32):
        return SparseMatrixELL_f32(A)
    return SparseMatrixELL_f64(A)


def from_scipy(sp_matrix):
    """Convert a scipy.sparse matrix (any format) to an MTL5 SparseMatrix.

    Non-CSR formats are converted to CSR via scipy, including CSC — MTL5 has no
    column-major container, so that conversion is real work rather than a
    relabelling. Use `coo_matrix()` to build COO directly without going through
    CSR. The result has the same dtype as the input (float32 or float64).
    """
    _ensure_scipy()
    if not _sp.issparse(sp_matrix):
        raise TypeError(f"Expected a scipy sparse matrix, got {type(sp_matrix)}")
    csr = sp_matrix.tocsr()
    return csr_matrix(csr.data, csr.indices, csr.indptr, csr.shape)


def to_scipy(mtl5_sparse):
    """Convert an MTL5 sparse matrix back to scipy.sparse.

    CSR and ELL come back as `csr_matrix`; COO comes back as `coo_matrix`, so a
    round trip preserves the format rather than silently compressing it — and
    with it any duplicate triplets, which `tocsr()` would have summed.
    """
    _ensure_scipy()
    if isinstance(mtl5_sparse, (SparseMatrixCOO_f32, SparseMatrixCOO_f64)):
        row, col, data = mtl5_sparse.to_coo_arrays()
        return _sp.coo_matrix((data, (row, col)), shape=mtl5_sparse.shape)
    if isinstance(mtl5_sparse, (SparseMatrixELL_f32, SparseMatrixELL_f64)):
        indices, data = mtl5_sparse.to_ell_arrays()
        keep = indices >= 0
        rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1]).reshape(indices.shape)
        return _sp.coo_matrix(
            (data[keep], (rows[keep], indices[keep])), shape=mtl5_sparse.shape
        ).tocsr()
    indptr, indices, data = mtl5_sparse.to_csr_arrays()
    return _sp.csr_matrix((data, indices, indptr), shape=mtl5_sparse.shape)


def as_linear_operator(mtl5_sparse):
    """Wrap an MTL5 sparse matrix as a scipy.sparse.linalg.LinearOperator.

    Enables use with SciPy's iterative solvers (cg, gmres, bicgstab, ...)
    by providing _matvec and _rmatvec callbacks that dispatch to MTL5 SpMV.

    Both _matvec and _rmatvec dispatch to MTL5. The transpose product goes
    through `rmatvec`, which uses MTL5's own transpose view — no scipy round
    trip and no second copy of the matrix.
    """
    _ensure_scipy()

    n_rows, n_cols = mtl5_sparse.shape
    dtype = np.float64 if mtl5_sparse.dtype == "f64" else np.float32

    def matvec_fn(x: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(x.ravel(), dtype=dtype)
        y = mtl5_sparse.matvec(x_arr)
        return y.to_numpy()

    def rmatvec_fn(x: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(x.ravel(), dtype=dtype)
        return mtl5_sparse.rmatvec(x_arr).to_numpy()

    return _LinearOperator(
        shape=(n_rows, n_cols),
        matvec=matvec_fn,
        rmatvec=rmatvec_fn,
        dtype=dtype,
    )


# ===========================================================================
# Iterative solvers — public API following SciPy convention
# ===========================================================================


def _coerce_matrix(A):
    """Accept either an MTL5 SparseMatrix or a scipy sparse matrix."""
    if isinstance(A, (SparseMatrix_f32, SparseMatrix_f64)):
        return A
    if HAS_SCIPY and _sp.issparse(A):
        return from_scipy(A)
    raise TypeError(
        f"A must be an MTL5 SparseMatrix or scipy.sparse matrix, got {type(A).__name__}"
    )


def _coerce_vector(b, expected_dtype: np.dtype):
    """Materialize any input to a contiguous NumPy array of the expected dtype.

    Accepts NumPy arrays, Python sequences, or MTL5 VectorView objects. The
    expected_dtype is enforced even for already-MTL5 inputs so that mixing
    e.g. SparseMatrix_f32 with DenseVector_f64 works correctly.
    """
    if hasattr(b, "to_numpy"):
        arr = np.ascontiguousarray(np.asarray(b.to_numpy()).ravel(), dtype=expected_dtype)
    else:
        arr = np.ascontiguousarray(np.asarray(b).ravel(), dtype=expected_dtype)
    return _vector(arr)


def _check_unsupported_kwargs(callback, device):
    """Validate optional kwargs reserved for future support.

    These parameters are reserved on the public API surface so that callers
    can write code today that will work transparently when the underlying
    kernel gains preconditioner / callback / device-dispatch support, without
    breaking the function signature later.
    """
    if callback is not None:
        # TODO(#7-followup): Add per-iteration callback dispatching to Python.
        raise NotImplementedError(
            "callback= not yet supported. The C++ solver loop runs to completion before returning."
        )
    if device is not None and device != "cpu":
        # TODO(#7-followup): Wire up KPU dispatch via mtl5.set_backend('kpu').
        raise NotImplementedError(
            f"device='{device}' not yet supported. Currently CPU only; "
            "KPU dispatch is in development."
        )


_SOLVER_DOC = """
    Parameters
    ----------
    A : MTL5 SparseMatrix or scipy.sparse matrix
    b : array-like
    rtol : float
        Relative residual tolerance.
    maxiter : int
        Iteration cap. `info` is 1 if it is reached without converging.
    M : preconditioner, optional
        Any object from this module's preconditioner factories — `ilu0`,
        `ic0`, `diagonal`, `ssor`, `ilut`, `ildl`, `block_diagonal`,
        `identity`. Defaults to `identity`, i.e. unpreconditioned.
    callback : callable, optional
        Reserved for per-iteration monitoring. Currently raises
        NotImplementedError.
    device : str, optional
        Reserved for KPU dispatch. Currently raises NotImplementedError for
        anything other than 'cpu'.

    Returns
    -------
    x : np.ndarray
        Solution vector.
    info : int
        0 on convergence, nonzero otherwise.
    """


def _run(name, A, b, rtol, maxiter, M, callback, device, restart=30, ell=2, s=4):
    """Shared body: coerce, default the preconditioner, dispatch."""
    _check_unsupported_kwargs(callback, device)
    mat = _coerce_matrix(A)
    dtype = np.float64 if mat.dtype == "f64" else np.float32
    bv = _coerce_vector(b, dtype)
    if M is None:
        M = identity(mat)
    elif not isinstance(M, (Preconditioner_f32, Preconditioner_f64)):
        raise TypeError(
            f"M must be an mtl5.sparse preconditioner (one of {preconditioners()}), "
            f"got {type(M).__name__}. A scipy LinearOperator cannot be used here — "
            "it would have to be applied from Python once per iteration."
        )
    x_view, info, _iters, _resid = _krylov(name, mat, M, bv, rtol, maxiter, restart, ell, s)
    return x_view.to_numpy(), info


def iterative_solve(
    A,
    b,
    *,
    solver="gmres",
    rtol=1e-10,
    maxiter=1000,
    M=None,
    callback=None,
    device=None,
    restart=30,
    ell=2,
    s=4,
):
    """Solve `A @ x = b` with any of the Krylov solvers in `solvers()`.

    The named wrappers below — `cg`, `gmres`, and the rest — are thin calls
    onto this. Use this form when the solver is chosen at runtime.

    `restart` applies to gmres, `ell` to bicgstab_ell and `s` to idr_s; each is
    ignored by the others.
    """
    if solver not in solvers():
        raise ValueError(f"unknown solver {solver!r}; valid: {solvers()}")
    return _run(solver, A, b, rtol, maxiter, M, callback, device, restart, ell, s)


def cg(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None):
    (
        """Conjugate Gradient, for symmetric positive definite systems.
    """
        + _SOLVER_DOC
    )
    return _run("cg", A, b, rtol, maxiter, M, callback, device)


def minres(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None):
    (
        """MINRES, for symmetric systems that may be indefinite — where CG does not
    apply.
    """
        + _SOLVER_DOC
    )
    return _run("minres", A, b, rtol, maxiter, M, callback, device)


def bicgstab(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None):
    (
        """BiCGSTAB, for general non-symmetric systems. A good default.
    """
        + _SOLVER_DOC
    )
    return _run("bicgstab", A, b, rtol, maxiter, M, callback, device)


def bicgstab_ell(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None, ell=2):
    (
        """BiCGSTAB(ell), which smooths BiCGSTAB's irregular convergence by taking
    `ell` minimal-residual steps at a time.
    """
        + _SOLVER_DOC
    )
    return _run("bicgstab_ell", A, b, rtol, maxiter, M, callback, device, ell=ell)


def cgs(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None):
    (
        """Conjugate Gradient Squared, for non-symmetric systems. Avoids A^T but
    converges more erratically than BiCGSTAB.
    """
        + _SOLVER_DOC
    )
    return _run("cgs", A, b, rtol, maxiter, M, callback, device)


def gmres(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None, restart=30):
    (
        """GMRES with restarts, for general non-symmetric systems. `restart` bounds
    the Krylov subspace, trading memory against convergence.
    """
        + _SOLVER_DOC
    )
    return _run("gmres", A, b, rtol, maxiter, M, callback, device, restart=restart)


def idr_s(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None, s=4):
    (
        """IDR(s), for non-symmetric systems. Larger `s` usually means fewer
    iterations and more memory.
    """
        + _SOLVER_DOC
    )
    return _run("idr_s", A, b, rtol, maxiter, M, callback, device, s=s)


def tfqmr(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None):
    (
        """Transpose-free QMR, for non-symmetric systems.
    """
        + _SOLVER_DOC
    )
    return _run("tfqmr", A, b, rtol, maxiter, M, callback, device)


def bicg(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None):
    (
        """BiCG, for non-symmetric systems.

    Needs a **symmetric** preconditioner. BiCG and QMR are the only solvers
    here that apply M^T, and MTL5 implements a preconditioner's adjoint as its
    forward solve — exact when M is symmetric, wrong otherwise. A non-symmetric
    ILU(0) or SSOR is refused rather than allowed to break down; `identity`,
    `diagonal`, `ic0` and `ildl` are always safe.
    """
        + _SOLVER_DOC
    )
    return _run("bicg", A, b, rtol, maxiter, M, callback, device)


def qmr(A, b, *, rtol=1e-10, maxiter=1000, M=None, callback=None, device=None):
    (
        """QMR, for non-symmetric systems. Needs a symmetric preconditioner — see
    `bicg` for why.
    """
        + _SOLVER_DOC
    )
    return _run("qmr", A, b, rtol, maxiter, M, callback, device)


# ===========================================================================
# Preconditioners — incomplete factorizations usable both standalone and
# as M= parameter in scipy iterative solvers (via as_preconditioner_lo).
# ===========================================================================


def _pc(f32_cls, f64_cls, A, *args, **kwargs):
    """Build a preconditioner of the class matching A's element type."""
    mat = _coerce_matrix(A)
    cls = f32_cls if mat.dtype == "f32" else f64_cls
    return cls(mat, *args, **kwargs)


def identity(A):
    """Identity preconditioner — a no-op, for running a solver unpreconditioned.

    This is the default when a solver is called without `M=`.
    """
    return _pc(Identity_f32, Identity_f64, A)


def diagonal(A):
    """Jacobi (diagonal) preconditioner: M = diag(A)."""
    return _pc(Diagonal_f32, Diagonal_f64, A)


def ilu0(A):
    """Incomplete LU factorization with no fill-in.

    Accepts an MTL5 or scipy sparse matrix and returns a preconditioner object
    with a `.solve(r)` method, usable as `M=` on any solver.
    """
    return _pc(ILU0_f32, ILU0_f64, A)


def ildl(A):
    """Incomplete LDL^T, for a symmetric indefinite matrix."""
    return _pc(ILDL_f32, ILDL_f64, A)


def ssor(A, omega: float = 1.0):
    """Symmetric successive over-relaxation. `omega` must lie in (0, 2)."""
    return _pc(SSOR_f32, SSOR_f64, A, omega)


def ilut(A, fill: int = 10, tau: float = 1e-4):
    """Incomplete LU with a drop threshold `tau` and at most `fill` entries
    kept per row — more accurate than ILU(0), and more expensive."""
    return _pc(ILUT_f32, ILUT_f64, A, fill, tau)


def block_diagonal(A, block_size: int):
    """Block Jacobi: invert each diagonal block of size `block_size`."""
    return _pc(BlockDiagonal_f32, BlockDiagonal_f64, A, block_size)


def ic0(A):
    """Incomplete Cholesky factorization with no fill-in (SPD matrices only).

    Accepts an MTL5 or scipy sparse matrix and returns an IC0 preconditioner
    object with a .solve(r) method.
    """
    mat = _coerce_matrix(A)
    if mat.dtype == "f32":
        return IC0_f32(mat)
    return IC0_f64(mat)


# ===========================================================================
# Sparse direct solvers
#
# The reason to reach for these over scipy.sparse.linalg.splu:
#
#   * analyze / factor / refactor is split, so a sequence of matrices sharing
#     one sparsity pattern (the circuit-transient case) pays for the ordering
#     and symbolic analysis once;
#   * the factor's precision is independent of the residual precision, so a
#     float32 factorization refined by mtl5.mixed.iterative_refine against a
#     float64 residual is a genuine mixed-precision direct solve.
# ===========================================================================


def splu(
    A,
    ordering: str = "colamd",
    threshold: float = 1.0,
    pivot_perturb: float = 0.0,
    accumulator: str | None = None,
):
    """LU-factorize a square sparse matrix (Gilbert-Peierls, threshold pivoting).

    Accepts an MTL5 or scipy sparse matrix. The dtype of `A` selects the
    precision of the factors — pass a float32 matrix to factor cheaply and
    recover accuracy with `mtl5.mixed.iterative_refine`.

    `ordering` is one of `orderings()`: "colamd" (default; column minimum
    degree, suits unsymmetric matrices), "amd" (symmetric minimum degree),
    "rcm" (bandwidth reducing) or "natural" (no fill reduction).

    `threshold` is the partial-pivoting threshold: 1.0 is classic partial
    pivoting, smaller values trade stability for less fill. `pivot_perturb`,
    if nonzero, replaces a zero pivot instead of raising — check
    `.num_perturbed` afterwards.

    The returned object exposes `.solve(b)` and `.refactor(A2)`.

    `accumulator=` types the dense numeric workspace, so a float32 factor can
    accumulate its updates in float64 ("f64") or with an FMA ("fma32"/"fma64").
    The factor itself stays in the element type — only the arithmetic that
    produced it widens. This is the mixed-precision knob a fixed-precision
    library cannot offer; see `mtl5.sparse` module docs for the measurement.
    """
    mat = _coerce_matrix(A)
    cls = SparseLU_f32 if mat.dtype == "f32" else SparseLU_f64
    return cls(mat, ordering, threshold, pivot_perturb, accumulator)


def klu(
    A,
    threshold: float = 1.0,
    scale: bool = True,
    pivot_perturb: float = 0.0,
    accumulator: str | None = None,
):
    """Factor a square sparse matrix with native KLU.

    Permutes to block triangular form (Dulmage-Mendelsohn), orders each
    diagonal block with AMD, and LU-factorizes the blocks. Built for circuit
    matrices, which are highly reducible — check `.nblocks` to see how much
    structure was found.

    `scale=True` factors R*A with row equilibration for pivot stability, which
    matters most in low precision; the RHS is scaled inside `.solve()`.

    The returned object exposes `.solve(b)` and `.refactor(A2)`.

    `accumulator=` types the dense numeric workspace, so a float32 factor can
    accumulate its updates in float64 ("f64") or with an FMA ("fma32"/"fma64").
    The factor itself stays in the element type — only the arithmetic that
    produced it widens. This is the mixed-precision knob a fixed-precision
    library cannot offer; see `mtl5.sparse` module docs for the measurement.
    """
    mat = _coerce_matrix(A)
    cls = KLU_f32 if mat.dtype == "f32" else KLU_f64
    return cls(mat, threshold, scale, pivot_perturb, accumulator)


def cholesky(A, ordering: str = "amd"):
    """Cholesky factorization A = L L^T of a symmetric positive definite matrix.

    Cheaper and sparser than `splu` when it applies: it exploits symmetry and
    does no pivoting. Raises RuntimeError if A turns out not to be positive
    definite. For a symmetric matrix that may be indefinite, use `ldlt`.

    The returned object exposes `.solve(b)` and `.refactor(A2)`.
    """
    mat = _coerce_matrix(A)
    cls = SparseCholesky_f32 if mat.dtype == "f32" else SparseCholesky_f64
    return cls(mat, ordering)


def ldlt(A, ordering: str = "amd"):
    """LDL^T factorization of a symmetric matrix.

    Does not require positive definiteness, so it handles symmetric indefinite
    matrices — inspect `.diagonal()` for the inertia. It does not pivot, so a
    zero pivot raises RuntimeError.

    The returned object exposes `.solve(b)` and `.refactor(A2)`.
    """
    mat = _coerce_matrix(A)
    cls = SparseLDLT_f32 if mat.dtype == "f32" else SparseLDLT_f64
    return cls(mat, ordering)


def qr(A, ordering: str = "colamd"):
    """Householder QR for sparse linear least squares.

    Accepts a rectangular (tall) or square matrix; `.solve(b)` returns the
    least-squares solution of min ||A x - b||_2.

    The returned object exposes `.solve(b)` and `.refactor(A2)`.
    """
    mat = _coerce_matrix(A)
    cls = SparseQR_f32 if mat.dtype == "f32" else SparseQR_f64
    return cls(mat, ordering)


def supernodal_lu(
    A,
    ordering: str = "colamd",
    threshold: float = 1.0,
    max_super: int = 64,
    scale: bool = False,
    pivot_perturb: float = 0.0,
    accumulator: str | None = None,
):
    """Supernodal LU: columns grouped into supernodes, applied as dense blocks.

    The same factorization as `splu` organised to spend its time in dense block
    updates instead of scalar sparse ones. `.nsuper` reports how many supernodes
    were formed. `scale=True` row-equilibrates first, which matters most when
    the factor precision is narrow.

    `ordering` must be "amd", "colamd" or "rcm" — the supernodal analysis has no
    natural-ordering path upstream.

    The returned object exposes `.solve(b)` and `.refactor(A2)`.

    `accumulator=` types the dense numeric workspace — see `splu`.
    """
    mat = _coerce_matrix(A)
    cls = SupernodalLU_f32 if mat.dtype == "f32" else SupernodalLU_f64
    return cls(mat, ordering, threshold, max_super, scale, pivot_perturb, accumulator)


def supernodal_ldlt(A, ordering: str = "amd", accumulator: str | None = None):
    """Supernodal LDL^T of a symmetric matrix, using dense block updates.

    The returned object exposes `.solve(b)` and `.refactor(A2)`.

    `accumulator=` types the dense numeric workspace — see `splu`. This is the
    one factorization here whose `.refactor()` carries the policy through,
    because it re-runs the full numeric factorization rather than replaying a
    stored pivot sequence.
    """
    mat = _coerce_matrix(A)
    cls = SupernodalLDLT_f32 if mat.dtype == "f32" else SupernodalLDLT_f64
    return cls(mat, ordering, accumulator)


def ordering(A, name: str = "amd"):
    """Return a fill-reducing permutation of A as an int64 array.

    Exposed separately so an ordering can be inspected, compared, or applied
    to a scipy matrix directly (`A[p][:, p]`). See `orderings()` for the names.
    """
    return _ordering(_coerce_matrix(A), name)


def amd(A):
    """Approximate minimum degree ordering on the pattern of A + A^T."""
    return ordering(A, "amd")


def colamd(A):
    """Column approximate minimum degree ordering on the pattern of A^T A."""
    return ordering(A, "colamd")


def rcm(A):
    """Reverse Cuthill-McKee (bandwidth-reducing) ordering."""
    return ordering(A, "rcm")


def as_preconditioner_lo(precond, n: int, dtype=None):
    """Wrap an ILU0/IC0 preconditioner as a scipy LinearOperator.

    Use as the `M` argument to scipy.sparse.linalg.cg / gmres / bicgstab.
    The dtype is inferred from the preconditioner type by default — explicit
    dtype is only needed for unusual cases.
    """
    _ensure_scipy()

    if dtype is None:
        if isinstance(precond, (ILU0_f32, IC0_f32)):
            dtype = np.float32
        elif isinstance(precond, (ILU0_f64, IC0_f64)):
            dtype = np.float64
        else:
            raise TypeError(
                f"Unsupported preconditioner type: {type(precond).__name__}. "
                "Pass dtype= explicitly."
            )

    def matvec_fn(r: np.ndarray) -> np.ndarray:
        r_arr = np.ascontiguousarray(r.ravel(), dtype=dtype)
        return precond.solve(r_arr).to_numpy()

    return _LinearOperator(
        shape=(n, n),
        matvec=matvec_fn,
        dtype=dtype,
    )


__all__ = [
    "HAS_SCIPY",
    "IC0_f32",
    "IC0_f64",
    "ILU0_f32",
    "ILU0_f64",
    "KLU_f32",
    "KLU_f64",
    "SparseCholesky_f32",
    "SparseCholesky_f64",
    "SparseLDLT_f32",
    "SparseLDLT_f64",
    "SparseLU_f32",
    "SparseLU_f64",
    "SparseQR_f32",
    "SparseQR_f64",
    "SupernodalLDLT_f32",
    "SupernodalLDLT_f64",
    "SupernodalLU_f32",
    "SupernodalLU_f64",
    "SparseMatrix_f32",
    "SparseMatrix_f64",
    "SparseMatrixCOO_f32",
    "SparseMatrixCOO_f64",
    "SparseMatrixELL_f32",
    "SparseMatrixELL_f64",
    "BlockDiagonal_f32",
    "BlockDiagonal_f64",
    "Diagonal_f32",
    "Diagonal_f64",
    "ILDL_f32",
    "ILDL_f64",
    "ILUT_f32",
    "ILUT_f64",
    "Identity_f32",
    "Identity_f64",
    "Preconditioner_f32",
    "Preconditioner_f64",
    "SSOR_f32",
    "SSOR_f64",
    "amd",
    "as_linear_operator",
    "as_preconditioner_lo",
    "bicg",
    "bicgstab",
    "bicgstab_ell",
    "block_diagonal",
    "cg",
    "cgs",
    "cholesky",
    "colamd",
    "coo_matrix",
    "csr_matrix",
    "ell_matrix",
    "from_scipy",
    "gmres",
    "ic0",
    "identity",
    "idr_s",
    "ildl",
    "ilu0",
    "ilut",
    "iterative_solve",
    "klu",
    "ldlt",
    "diagonal",
    "minres",
    "ordering",
    "orderings",
    "preconditioners",
    "qmr",
    "solvers",
    "ssor",
    "tfqmr",
    "qr",
    "rcm",
    "splu",
    "supernodal_ldlt",
    "supernodal_lu",
    "to_scipy",
]
