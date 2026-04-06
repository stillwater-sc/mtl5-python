"""MTL5 Python bindings — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch."""

from mtl5._core import (
    __version__,
    DenseMatrix,
    DenseVector,
    dot,
    matrix,
    norm,
    solve,
    vector,
)

__all__ = [
    "__version__",
    "DenseMatrix",
    "DenseVector",
    "dot",
    "matrix",
    "norm",
    "solve",
    "vector",
]
