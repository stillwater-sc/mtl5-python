"""The NumPy array protocol on MTL5's dense containers (#17).

``np.asarray(x)`` asks ``x.__array__`` for an array. Without it, NumPy saw a
``DenseMatrix`` as an opaque object and built a 0-d object array, which
scikit-learn and anything else that validates its input with ``np.asarray``
rejects. A ``DenseVector`` fell back to the sequence protocol instead: a copy
built one ``__getitem__`` at a time.

Two kinds of container answer differently:

* **Native** (float, int, complex: the ones with ``is_view``). ``to_numpy()`` is
  already a zero-copy view of the storage, so ``np.asarray`` shares memory with
  the container, and through it with the NumPy array an ``mtl5.matrix(X)`` view
  borrows.
* **Universal** (posit, fp8, lns, ...). NumPy has no dtype for these, so
  ``to_numpy()`` converts to float64 and is always a copy. ``copy=False``
  therefore raises, as NumPy 2 requires of a protocol that cannot avoid a copy.
  ``to_universal()`` is the bit-exact way out (``_universal_interop``).

Sparse matrices are left out on purpose: ``np.asarray`` on them would densify
silently. Pass ``to_scipy()`` to libraries that take sparse input.
"""

from __future__ import annotations

import numpy as np

from mtl5 import _core


def _native_array(self, dtype=None, copy=None):
    a = self.to_numpy()
    if dtype is not None and np.dtype(dtype) != a.dtype:
        if copy is False:
            raise ValueError(
                f"{type(self).__name__} is {a.dtype}; converting to {np.dtype(dtype)} needs a copy"
            )
        return a.astype(dtype)
    return a.copy() if copy else a


def _converted_array(self, dtype=None, copy=None):
    if copy is False:
        raise ValueError(
            f"{type(self).__name__} has no NumPy dtype, so it converts to float64 "
            "with a copy; use to_universal() for the encodings"
        )
    a = self.to_numpy()
    return a if dtype is None else a.astype(dtype, copy=False)


def install() -> None:
    """Attach ``__array__`` to every dense vector and matrix class."""
    for name in dir(_core):
        if name.startswith(("DenseVector_", "DenseMatrix_")):
            cls = getattr(_core, name)
            cls.__array__ = _native_array if hasattr(cls, "is_view") else _converted_array
