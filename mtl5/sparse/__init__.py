"""MTL5 sparse matrix operations — SciPy interop, iterative solvers, preconditioners.

This submodule provides:

- `SparseMatrix_f32` / `SparseMatrix_f64` — MTL5 CSR sparse matrices
- `from_scipy(sp)` — convert a scipy.sparse matrix to MTL5
- `to_scipy(A)` — convert an MTL5 sparse matrix back to scipy.sparse.csr_matrix
- `csr_matrix(data, indices, indptr, shape)` — direct CSR construction
- `as_linear_operator(A)` — wrap an MTL5 sparse matrix as a SciPy
  `LinearOperator`, enabling use with scipy.sparse.linalg iterative solvers
- `cg`, `gmres`, `bicgstab` — iterative Krylov solvers returning (x, info)
- `ilu0`, `ic0` — incomplete LU/Cholesky preconditioners

scipy is an optional dependency: importing this module without scipy installed
yields the bare MTL5 SparseMatrix bindings, but the conversion helpers raise
ImportError when called.
"""

from __future__ import annotations

import numpy as np

from mtl5._core import (
    IC0_f32,
    IC0_f64,
    ILU0_f32,
    ILU0_f64,
    SparseMatrix_f32,
    SparseMatrix_f64,
    _sparse_bicgstab,
    _sparse_cg,
    _sparse_gmres,
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


def from_scipy(sp_matrix):
    """Convert a scipy.sparse matrix (any format) to an MTL5 SparseMatrix.

    Non-CSR formats are converted to CSR via scipy. The result has the
    same dtype as the input (must be float32 or float64).
    """
    _ensure_scipy()
    if not _sp.issparse(sp_matrix):
        raise TypeError(f"Expected a scipy sparse matrix, got {type(sp_matrix)}")
    csr = sp_matrix.tocsr()
    return csr_matrix(csr.data, csr.indices, csr.indptr, csr.shape)


def to_scipy(mtl5_sparse):
    """Convert an MTL5 SparseMatrix back to a scipy.sparse.csr_matrix."""
    _ensure_scipy()
    indptr, indices, data = mtl5_sparse.to_csr_arrays()
    return _sp.csr_matrix((data, indices, indptr), shape=mtl5_sparse.shape)


def as_linear_operator(mtl5_sparse):
    """Wrap an MTL5 sparse matrix as a scipy.sparse.linalg.LinearOperator.

    Enables use with SciPy's iterative solvers (cg, gmres, bicgstab, ...)
    by providing _matvec and _rmatvec callbacks that dispatch to MTL5 SpMV.

    For symmetric matrices, _rmatvec is implemented as _matvec on the same
    matrix. For non-symmetric matrices we currently fall back to converting
    via scipy — a true MTL5 transpose-SpMV will follow once compressed2D
    gains a CSC/transpose accessor.
    """
    _ensure_scipy()

    n_rows, n_cols = mtl5_sparse.shape
    dtype = np.float64 if mtl5_sparse.dtype == "f64" else np.float32

    def matvec_fn(x: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(x.ravel(), dtype=dtype)
        y = mtl5_sparse.matvec(x_arr)
        return y.to_numpy()

    # rmatvec via scipy round-trip — see docstring
    sp = to_scipy(mtl5_sparse)

    def rmatvec_fn(x: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(x.ravel(), dtype=dtype)
        return sp.T @ x_arr

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


def _check_unsupported_kwargs(M, callback, device):
    """Validate optional kwargs reserved for future support.

    These parameters are reserved on the public API surface so that callers
    can write code today that will work transparently when the underlying
    kernel gains preconditioner / callback / device-dispatch support, without
    breaking the function signature later.
    """
    if M is not None:
        # TODO(#7-followup): Plumb the preconditioner directly into the C++
        # solver loop. For now users should pass MTL5 preconditioners through
        # scipy: scipy_cg(A, b, M=msp.as_preconditioner_lo(msp.ilu0(A), n))
        raise NotImplementedError(
            "M= preconditioner not yet plumbed into mtl5.sparse solvers. "
            "Use scipy.sparse.linalg.cg with M=mtl5.sparse.as_preconditioner_lo("
            "mtl5.sparse.ilu0(A), n) for now."
        )
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


def cg(
    A,
    b,
    *,
    rtol: float = 1e-10,
    maxiter: int = 1000,
    M=None,
    callback=None,
    device=None,
):
    """Conjugate Gradient solver for symmetric positive-definite systems.

    Solves `A @ x = b` using CG. Accepts MTL5 or scipy.sparse matrices.

    Parameters
    ----------
    A : MTL5 SparseMatrix or scipy.sparse matrix
    b : array-like
    rtol : float
        Relative tolerance.
    maxiter : int
        Maximum number of iterations.
    M : preconditioner, optional
        Reserved for future preconditioner support. Currently raises
        NotImplementedError — use scipy.sparse.linalg.cg with
        as_preconditioner_lo() to apply MTL5 preconditioners.
    callback : callable, optional
        Reserved for per-iteration monitoring. Currently raises NotImplementedError.
    device : str, optional
        Reserved for KPU dispatch. Currently raises NotImplementedError for
        anything other than 'cpu'.

    Returns
    -------
    x : np.ndarray
        Solution vector.
    info : int
        0 on convergence, 1 if max_iter exceeded.
    """
    _check_unsupported_kwargs(M, callback, device)
    mat = _coerce_matrix(A)
    dtype = np.float64 if mat.dtype == "f64" else np.float32
    bv = _coerce_vector(b, dtype)
    x_view, info, _iters, _resid = _sparse_cg(mat, bv, rtol, maxiter)
    return x_view.to_numpy(), info


def gmres(
    A,
    b,
    *,
    rtol: float = 1e-10,
    maxiter: int = 1000,
    restart: int = 30,
    M=None,
    callback=None,
    device=None,
):
    """GMRES solver for general non-symmetric systems.

    Returns (x, info) following scipy convention. See `cg` for parameter docs;
    `M`, `callback`, and `device` are reserved for future support.
    """
    _check_unsupported_kwargs(M, callback, device)
    mat = _coerce_matrix(A)
    dtype = np.float64 if mat.dtype == "f64" else np.float32
    bv = _coerce_vector(b, dtype)
    x_view, info, _iters, _resid = _sparse_gmres(mat, bv, rtol, maxiter, restart)
    return x_view.to_numpy(), info


def bicgstab(
    A,
    b,
    *,
    rtol: float = 1e-10,
    maxiter: int = 1000,
    M=None,
    callback=None,
    device=None,
):
    """BiCGSTAB solver for general non-symmetric systems.

    Returns (x, info) following scipy convention. See `cg` for parameter docs;
    `M`, `callback`, and `device` are reserved for future support.
    """
    _check_unsupported_kwargs(M, callback, device)
    mat = _coerce_matrix(A)
    dtype = np.float64 if mat.dtype == "f64" else np.float32
    bv = _coerce_vector(b, dtype)
    x_view, info, _iters, _resid = _sparse_bicgstab(mat, bv, rtol, maxiter)
    return x_view.to_numpy(), info


# ===========================================================================
# Preconditioners — incomplete factorizations usable both standalone and
# as M= parameter in scipy iterative solvers (via as_preconditioner_lo).
# ===========================================================================


def ilu0(A):
    """Incomplete LU factorization with no fill-in.

    Accepts an MTL5 or scipy sparse matrix and returns an ILU0 preconditioner
    object with a .solve(r) method.
    """
    mat = _coerce_matrix(A)
    if mat.dtype == "f32":
        return ILU0_f32(mat)
    return ILU0_f64(mat)


def ic0(A):
    """Incomplete Cholesky factorization with no fill-in (SPD matrices only).

    Accepts an MTL5 or scipy sparse matrix and returns an IC0 preconditioner
    object with a .solve(r) method.
    """
    mat = _coerce_matrix(A)
    if mat.dtype == "f32":
        return IC0_f32(mat)
    return IC0_f64(mat)


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
    "SparseMatrix_f32",
    "SparseMatrix_f64",
    "as_linear_operator",
    "as_preconditioner_lo",
    "bicgstab",
    "cg",
    "csr_matrix",
    "from_scipy",
    "gmres",
    "ic0",
    "ilu0",
    "to_scipy",
]
