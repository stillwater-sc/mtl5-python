"""MTL5 Python bindings — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch."""

from mtl5._core import (
    DenseMatrix_f32,
    DenseMatrix_f64,
    DenseMatrix_i32,
    DenseMatrix_i64,
    DenseVector_f32,
    DenseVector_f64,
    DenseVector_i32,
    DenseVector_i64,
    __version__,
    dot,
    matrix,
    norm,
    solve,
    vector,
)

# Convenience aliases — default to f64 for backward compatibility
DenseVector = DenseVector_f64
DenseMatrix = DenseMatrix_f64

__all__ = [
    "__version__",
    # Typed vector classes
    "DenseVector",
    "DenseVector_f32",
    "DenseVector_f64",
    "DenseVector_i32",
    "DenseVector_i64",
    # Typed matrix classes
    "DenseMatrix",
    "DenseMatrix_f32",
    "DenseMatrix_f64",
    "DenseMatrix_i32",
    "DenseMatrix_i64",
    # Operations (auto-dispatch based on input dtype)
    "dot",
    "matrix",
    "norm",
    "solve",
    "vector",
]
