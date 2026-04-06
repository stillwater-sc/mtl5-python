"""MTL5 Python bindings — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch."""

from mtl5._core import (
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
    __version__,
    # Device management
    devices,
    # Operations (auto-dispatch on input dtype for native types)
    dot,
    matrix,
    matrix_copy,
    matrix_fp8,
    matrix_fp16,
    norm,
    solve,
    vector,
    vector_copy,
    vector_fp8,
    vector_fp16,
)

# Convenience aliases — default to f64
DenseVector = DenseVector_f64
DenseMatrix = DenseMatrix_f64

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
    # Device management
    "devices",
    # Operations
    "dot",
    "matrix",
    "matrix_copy",
    "matrix_fp8",
    "matrix_fp16",
    "norm",
    "solve",
    "vector",
    "vector_copy",
    "vector_fp8",
    "vector_fp16",
]
