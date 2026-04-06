"""MTL5 Python bindings — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch."""

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
    __version__,
    # Backend management
    backends,
    cholesky,
    # Device management
    devices,
    # Operations
    dot,
    get_backend,
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
    norm,
    set_backend,
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

# Convenience aliases — default to f64
DenseVector = DenseVector_f64
DenseMatrix = DenseMatrix_f64
LUFactor = LUFactor_f64
CholeskyFactor = CholeskyFactor_f64


# Re-export det (avoids shadowing the builtin namespace)
def det(A):
    """Compute the determinant of a matrix via LU factorization."""
    return _det(A)


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
    # Operations
    "cholesky",
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
