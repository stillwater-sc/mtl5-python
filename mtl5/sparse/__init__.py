"""MTL5 sparse matrix operations — SciPy interop, iterative solvers, preconditioners.

This submodule provides:

- `SparseMatrix_f32` / `SparseMatrix_f64` — MTL5 CSR sparse matrices
- `from_scipy(sp)` — convert a scipy.sparse matrix to MTL5
- `to_scipy(A)` — convert an MTL5 sparse matrix back to scipy.sparse.csr_matrix
- `csr_matrix(data, indices, indptr, shape)` — direct CSR construction
- `as_linear_operator(A)` — wrap an MTL5 sparse matrix as a SciPy
  `LinearOperator`, enabling use with scipy.sparse.linalg iterative solvers

scipy is an optional dependency: importing this module without scipy installed
yields the bare MTL5 SparseMatrix bindings, but the conversion helpers raise
ImportError when called.
"""

from __future__ import annotations

import numpy as np

from mtl5._core import SparseMatrix_f32, SparseMatrix_f64

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


__all__ = [
    "HAS_SCIPY",
    "SparseMatrix_f32",
    "SparseMatrix_f64",
    "as_linear_operator",
    "csr_matrix",
    "from_scipy",
    "to_scipy",
]
