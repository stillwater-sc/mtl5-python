"""Matrix Market I/O and PNG sparsity visualization.

Two behaviours differ from what a SciPy user will expect, so they are stated
here rather than left to be discovered:

**The function picks the container, not the file.** `scipy.io.mmread` returns an
ndarray for an `array` file and a COO matrix for a `coordinate` one. Here
`mm_read` always gives you CSR and `mm_read_dense` always gives you dense, and
both accept either file format. Reading a dense `.mtx` into CSR is a deliberate
call, not an error.

**The PNGs are uncompressed.** MTL5 writes them with a from-scratch encoder that
emits DEFLATE *stored* blocks, which is what lets `spy` work with no image
library and no plotting stack in the process. The cost is that file size is
essentially `width * height * channels`. Measured, at the default
`max_pixels=1024`:

  * `spy` is one-channel grayscale      — about 1.0 MB
  * `spy_magnitude` / `spy_density` are RGB — about 3.1 MB

Run the output through any PNG optimizer if the size matters.

`.gz` inputs are read transparently only when MTL5 was built with zlib; check
`mtl5.build_info()["zlib"]`. Without it, a `.gz` path raises rather than
silently mis-parsing.
"""

from __future__ import annotations

import numpy as np

from mtl5._core import io as _io
from mtl5._core import matrix as _matrix


def mm_read(path):
    """Read a Matrix Market file into a sparse (CSR) matrix.

    Accepts both `coordinate` and `array` files — the function chooses the
    container, not the file. Values are read as float64.
    """
    return _io.mm_read(str(path))


def mm_read_dense(path):
    """Read a Matrix Market file into a dense matrix.

    Accepts both `coordinate` and `array` files. Values are read as float64.
    """
    return _io.mm_read_dense(str(path))


def _coerce(A):
    """Accept a NumPy float32/float64 array wherever an MTL5 matrix is wanted."""
    if isinstance(A, np.ndarray):
        if A.dtype.name not in ("float32", "float64"):
            raise TypeError(
                f"NumPy arrays must be float32 or float64, got {A.dtype.name}. "
                "For another number system, convert first: mtl5.convert(a, 'posit32')."
            )
        return _matrix(np.ascontiguousarray(A))
    return A


def mm_write(path, A, comment: str = ""):
    """Write a dense matrix in Matrix Market `array` format."""
    return _io.mm_write(str(path), _coerce(A), comment)


def mm_write_sparse(path, A, comment: str = ""):
    """Write a sparse matrix in Matrix Market `coordinate` format."""
    return _io.mm_write_sparse(str(path), A, comment)


def spy(A, path, max_pixels: int = 1024):
    """Binary non-zero pattern as a grayscale PNG (MATLAB `spy`)."""
    return _io.spy(_coerce(A), str(path), max_pixels)


def spy_magnitude(A, path, max_pixels: int = 1024, log_scale: bool = False):
    """Colour each non-zero by |value| — where the mass is, not just the pattern."""
    return _io.spy_magnitude(_coerce(A), str(path), max_pixels, log_scale)


def spy_density(A, path, max_pixels: int = 1024, log_scale: bool = False):
    """Colour each pixel by how many non-zeros landed in it.

    The useful view once the matrix is bigger than the image, where `spy` would
    saturate every pixel to black regardless of how many entries it covers.
    """
    return _io.spy_density(_coerce(A), str(path), max_pixels, log_scale)


__all__ = [
    "mm_read",
    "mm_read_dense",
    "mm_write",
    "mm_write_sparse",
    "spy",
    "spy_density",
    "spy_magnitude",
]
