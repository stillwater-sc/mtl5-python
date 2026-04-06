"""MTL5 Python bindings — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch."""

from mtl5._core import (
    # Cholesky factorization objects
    CholeskyFactor_f32,
    CholeskyFactor_f64,
    # Native IEEE types (zero-copy views)
    DenseMatrix_f32,
    DenseMatrix_f64,
    # Universal number types (copy-converting)
    DenseMatrix_fp8,
    DenseMatrix_fp16,
    DenseMatrix_i32,
    DenseMatrix_i64,
    DenseVector_f32,
    DenseVector_f64,
    DenseVector_fp8,
    DenseVector_fp16,
    DenseVector_i32,
    DenseVector_i64,
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
    matrix_fp8,
    matrix_fp16,
    matvec,
    norm,
    set_backend,
    solve,
    transpose,
    vector,
    vector_copy,
    vector_fp8,
    vector_fp16,
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


__all__ = [
    "__version__",
    # Typed vector classes
    "DenseVector",
    "DenseVector_f32",
    "DenseVector_f64",
    "DenseVector_fp8",
    "DenseVector_fp16",
    "DenseVector_i32",
    "DenseVector_i64",
    # Typed matrix classes
    "DenseMatrix",
    "DenseMatrix_f32",
    "DenseMatrix_f64",
    "DenseMatrix_fp8",
    "DenseMatrix_fp16",
    "DenseMatrix_i32",
    "DenseMatrix_i64",
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
    "matrix_fp8",
    "matrix_fp16",
    "matvec",
    "norm",
    "solve",
    "transpose",
    "vector",
    "vector_copy",
    "vector_fp8",
    "vector_fp16",
]
