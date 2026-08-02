"""MTL5 Python bindings — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import mtl5._core as _core

# Single source of truth: pyproject.toml → installed package metadata.
# The C++ module also reads this at build time via CMake.
# Falls back to the C++ extension's version when running from a source tree
# without installed metadata (e.g. an in-place build without pip install).
try:
    __version__ = _pkg_version("mtl5")
except PackageNotFoundError:
    try:
        from mtl5._core import __version__  # noqa: F811
    except ImportError:
        __version__ = "0.0.0-dev"

from mtl5._core import (
    # Cholesky factorization objects
    CholeskyFactor_f32,
    CholeskyFactor_f64,
    # Native IEEE types (zero-copy views)
    DenseMatrix_f32,
    DenseMatrix_f64,
    # Universal fixpnt types
    DenseMatrix_fixpnt8,
    DenseMatrix_fixpnt16,
    # Universal cfloat types
    DenseMatrix_fp8,
    DenseMatrix_fp16,
    DenseMatrix_i32,
    DenseMatrix_i64,
    # Universal lns types
    DenseMatrix_lns16,
    DenseMatrix_lns32,
    # Universal posit types
    DenseMatrix_posit8,
    DenseMatrix_posit16,
    DenseMatrix_posit32,
    DenseMatrix_posit64,
    DenseVector_f32,
    DenseVector_f64,
    DenseVector_fixpnt8,
    DenseVector_fixpnt16,
    DenseVector_fp8,
    DenseVector_fp16,
    DenseVector_i32,
    DenseVector_i64,
    DenseVector_lns16,
    DenseVector_lns32,
    DenseVector_posit8,
    DenseVector_posit16,
    DenseVector_posit32,
    DenseVector_posit64,
    # LU factorization objects
    LUFactor_f32,
    LUFactor_f64,
    # Backend management
    backends,
    # Build introspection
    build_info,
    cholesky,
    # Mixed precision
    convert,
    # Device management
    devices,
    # Operations
    dot,
    dtypes,
    get_backend,
    # Threading
    get_num_threads,
    inv,
    lu,
    matmul,
    matrix,
    matrix_copy,
    matrix_fixpnt8,
    matrix_fixpnt16,
    matrix_fp8,
    matrix_fp16,
    matrix_lns16,
    matrix_lns32,
    matrix_posit8,
    matrix_posit16,
    matrix_posit32,
    matrix_posit64,
    matvec,
    # Mixed-precision submodule (element / accumulator / result precisions)
    mixed,
    norm,
    set_backend,
    set_num_threads,
    solve,
    transpose,
    vector,
    vector_copy,
    vector_fixpnt8,
    vector_fixpnt16,
    vector_fp8,
    vector_fp16,
    vector_lns16,
    vector_lns32,
    vector_posit8,
    vector_posit16,
    vector_posit32,
    vector_posit64,
)
from mtl5._core import det as _det

_UNIVERSAL_DTYPES = (
    "fp8",
    "fp16",
    "posit8",
    "posit16",
    "posit32",
    "posit64",
    "fixpnt8",
    "fixpnt16",
    "lns16",
    "lns32",
)


_NUMPY_TO_MTL5 = {"float32": "f32", "float64": "f64"}


def _as_mtl5_matrix(prefix: str, A):
    """Normalize A to (mtl5_matrix, dtype_suffix).

    Accepts an MTL5 matrix from mtl5.matrix()/mtl5.convert(), or a NumPy
    float32/float64 array, which is wrapped zero-copy — matching what
    mtl5.cholesky() has always accepted.
    """
    import numpy as _np

    if isinstance(A, _np.ndarray):
        suffix = _NUMPY_TO_MTL5.get(A.dtype.name)
        if suffix is None:
            raise TypeError(
                f"{prefix.lower()}: NumPy arrays must be float32 or float64, "
                f"got {A.dtype.name}. For another number system, convert first: "
                f"mtl5.convert(a, 'posit32')."
            )
        return matrix(_np.ascontiguousarray(A)), suffix

    dt = getattr(A, "dtype", None)
    if not isinstance(dt, str):
        raise TypeError(
            f"{prefix.lower()}: expected an MTL5 matrix (from mtl5.matrix() or "
            f"mtl5.convert()) or a NumPy array, got {type(A).__name__}"
        )
    return A, dt


def _factor(prefix: str, A):
    """Construct the Factor class matching A's element type."""
    mat, dt = _as_mtl5_matrix(prefix, A)
    cls = getattr(_core, f"{prefix}_{dt}", None)
    if cls is None:
        raise TypeError(f"{prefix.lower()} is not available for dtype '{dt}'")
    return cls(mat)


def qr(A):
    """Householder QR of a tall-or-square matrix.

    Returns a factorization exposing `.solve(b)` (least squares), `.Q` and `.R`.
    """
    return _factor("QRFactor", A)


def lq(A):
    """Householder LQ. Returns a factorization exposing `.L` and `.Q`."""
    return _factor("LQFactor", A)


def ldlt(A):
    """LDL^T factorization of a symmetric matrix.

    Takes no square roots, so unlike `cholesky` it tolerates an indefinite
    matrix; `.diagonal()` returns D, whose signs give the inertia. It does not
    pivot, so a zero pivot raises.

    Available for float32/float64 and every Universal dtype, which is what
    makes a Cholesky-versus-LDL^T comparison across number systems possible.
    """
    return _factor("LDLTFactor", A)


# Convenience aliases — default to f64
DenseVector = DenseVector_f64
DenseMatrix = DenseMatrix_f64
LUFactor = LUFactor_f64
CholeskyFactor = CholeskyFactor_f64


# Re-export det (avoids shadowing the builtin namespace)
def det(A):
    """Compute the determinant of a matrix via LU factorization."""
    return _det(A)


_native_cholesky = cholesky


def cholesky(A):  # noqa: F811
    """Cholesky factorization A = L L^T of a symmetric positive definite matrix.

    Accepts float32/float64 matrices and NumPy arrays, and — since Phase 3 —
    the Universal dtypes as well, so it can be compared against `ldlt` in the
    same precision.
    """
    dt = getattr(A, "dtype", None)
    if isinstance(dt, str) and dt in _UNIVERSAL_DTYPES:
        return getattr(_core, f"CholeskyFactor_{dt}")(A)
    return _native_cholesky(A)


# Optional pandas extension types — only loaded if pandas is installed
try:
    from mtl5.pandas_ext import HAS_PANDAS  # noqa: F401

    if HAS_PANDAS:
        from mtl5.pandas_ext import Posit16Array, Posit16Dtype  # noqa: F401
except ImportError:
    pass


__all__ = [
    "__version__",
    # Typed vector classes — IEEE
    "DenseVector",
    "DenseVector_f32",
    "DenseVector_f64",
    "DenseVector_i32",
    "DenseVector_i64",
    # Typed vector classes — Universal cfloat
    "DenseVector_fp8",
    "DenseVector_fp16",
    # Typed vector classes — Universal posit
    "DenseVector_posit8",
    "DenseVector_posit16",
    "DenseVector_posit32",
    "DenseVector_posit64",
    # Typed vector classes — Universal fixpnt
    "DenseVector_fixpnt8",
    "DenseVector_fixpnt16",
    # Typed vector classes — Universal lns
    "DenseVector_lns16",
    "DenseVector_lns32",
    # Typed matrix classes — IEEE
    "DenseMatrix",
    "DenseMatrix_f32",
    "DenseMatrix_f64",
    "DenseMatrix_i32",
    "DenseMatrix_i64",
    # Typed matrix classes — Universal cfloat
    "DenseMatrix_fp8",
    "DenseMatrix_fp16",
    # Typed matrix classes — Universal posit
    "DenseMatrix_posit8",
    "DenseMatrix_posit16",
    "DenseMatrix_posit32",
    "DenseMatrix_posit64",
    # Typed matrix classes — Universal fixpnt
    "DenseMatrix_fixpnt8",
    "DenseMatrix_fixpnt16",
    # Typed matrix classes — Universal lns
    "DenseMatrix_lns16",
    "DenseMatrix_lns32",
    # Factorization classes
    "LUFactor",
    "LUFactor_f32",
    "LUFactor_f64",
    "CholeskyFactor",
    "CholeskyFactor_f32",
    "CholeskyFactor_f64",
    # Device & backend management
    "backends",
    "devices",
    "get_backend",
    "set_backend",
    # Mixed precision
    "mixed",
    "convert",
    "dtypes",
    # Build introspection & threading
    "build_info",
    "get_num_threads",
    "set_num_threads",
    # Operations
    "cholesky",
    "ldlt",
    "lq",
    "qr",
    "det",
    "dot",
    "inv",
    "lu",
    "matmul",
    "matrix",
    "matrix_copy",
    "matvec",
    "norm",
    "solve",
    "transpose",
    "vector",
    "vector_copy",
    # Universal type factories — cfloat
    "matrix_fp8",
    "matrix_fp16",
    "vector_fp8",
    "vector_fp16",
    # Universal type factories — posit
    "matrix_posit8",
    "matrix_posit16",
    "matrix_posit32",
    "matrix_posit64",
    "vector_posit8",
    "vector_posit16",
    "vector_posit32",
    "vector_posit64",
    # Universal type factories — fixpnt
    "matrix_fixpnt8",
    "matrix_fixpnt16",
    "vector_fixpnt8",
    "vector_fixpnt16",
    # Universal type factories — lns
    "matrix_lns16",
    "matrix_lns32",
    "vector_lns16",
    "vector_lns32",
]
