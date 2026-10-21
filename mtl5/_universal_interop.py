"""Moving ``universal_dtypes`` arrays into and out of MTL5, bit for bit (#14).

``universal_dtypes`` registers the Universal number systems as real NumPy
dtypes, so ``np.array([...], dtype=universal_dtypes.posit16)`` is a NumPy array
of posits. MTL5 holds the same number systems in its own containers. This
module is the seam between the two. ``mtl5.vector()`` / ``mtl5.matrix()``
accept such an array, and ``.to_universal()`` gives one back, both without a
detour through float64.

The bits are COPIED, not viewed. ``universal_dtypes`` and this package are
built on different Universal releases, and what two releases agree on is the
encoding, not the C++ object layout. MTL5's Universal containers also own
their storage, so a view would have nothing to keep the NumPy array alive. The
copy is one pass, which is nothing next to emulated arithmetic.

``universal_dtypes`` is optional. Recognising its arrays needs only their dtype
object, so it is imported only when ``to_universal()`` has to build one.
"""

from __future__ import annotations

import functools

import numpy as np

from mtl5 import _core

# The universal_dtypes dtypes whose encoding is the one this build binds, under
# the same name on both sides. tests/test_universal_dtypes.py checks each one
# bit for bit against universal_dtypes' own decoding: every pattern for the 8-
# and 16-bit formats, a large random sample for the wider ones.
SUPPORTED = (
    "posit8",
    "posit16",
    "posit32",
    "posit64",
    "fixpnt8",
    "fixpnt16",
    "lns16",
    "lns32",
    "takum32",
    "fp16",
    "dd_cascade",
    "td_cascade",
    "qd_cascade",
)

# Same family, different format: an array of these must be refused, not read
# as the MTL5 type whose name it resembles.
_MISMATCHED = {
    "fp8e5m2": "MTL5's fp8 is Universal's cfloat<8,2> (e2m5), not e5m2",
}

_CASCADES = frozenset({"dd_cascade", "td_cascade", "qd_cascade"})
_WORD = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def universal_name(dtype) -> str | None:
    """The dtype's name if it is a ``universal_dtypes`` dtype, else None.

    Decided by the module of the dtype's class, so it never needs to import
    ``universal_dtypes``, and a plain NumPy dtype that happens to share a name
    cannot match.
    """
    module = type(dtype).__module__ or ""
    if module.split(".")[0] != "universal_dtypes":
        return None
    # The scalar type's name, not dtype.name: the latter is the DType class
    # spelling ('Posit16DType16'), while the scalar is universal_dtypes.posit16.
    return dtype.type.__name__


def _storage_word(name: str, dtype):
    """The array type the bits travel as: the unsigned integer of the
    element's width, or float64 limbs for a cascade (2, 3 or 4 per element)."""
    return np.float64 if name in _CASCADES else _WORD[dtype.itemsize]


def from_universal(a: np.ndarray, kind: str):
    """Copy a ``universal_dtypes`` array into an MTL5 ``kind`` ("vector" or
    "matrix") of the same number system, bit for bit."""
    name = universal_name(a.dtype)
    if name not in SUPPORTED:
        why = _MISMATCHED.get(name, "MTL5 has no element type with that encoding")
        raise TypeError(
            f"mtl5.{kind}(): universal_dtypes dtype '{name}' is not supported ({why}). "
            f"Supported: {', '.join(SUPPORTED)}. To round the values into one of "
            "those instead, use mtl5.convert(a.astype(np.float64), '<dtype>')."
        )
    want = 1 if kind == "vector" else 2
    if a.ndim != want:
        raise TypeError(f"mtl5.{kind}() takes a {want}-D array; got {a.ndim}-D")
    # Copying anyway, so a non-contiguous array costs nothing extra to accept.
    bits = np.ascontiguousarray(a).view(_storage_word(name, a.dtype))
    return getattr(_core, f"_{kind}_from_bits_{name}")(bits)


def to_universal(self):
    """Return this container as a ``universal_dtypes`` array of the same number
    system, bit for bit (``to_numpy()`` widens to float64 instead).

    Needs ``universal_dtypes``: ``pip install mtl5[universal]``.
    """
    try:
        import universal_dtypes
    except ImportError as exc:
        raise ImportError(
            "to_universal() needs the universal_dtypes package: pip install mtl5[universal]"
        ) from exc
    return self._to_bits().view(getattr(universal_dtypes, self.dtype))


_FACTORY_NOTE = (
    "\n\nAlso accepts a universal_dtypes array (posit, fixpnt, lns, takum, fp16, "
    "cascades), which is copied bit for bit into the MTL5 container of the same "
    "number system rather than rounded through float64. .to_universal() is the "
    "way back."
)


def wrap_factory(fn, kind: str):
    """Give an existing ``vector``/``matrix`` factory the universal_dtypes path."""

    @functools.wraps(fn)
    def factory(a, *args, **kwargs):
        if isinstance(a, np.ndarray) and universal_name(a.dtype) is not None:
            if args or kwargs:
                raise TypeError(f"mtl5.{kind}() takes no options for a universal_dtypes array")
            return from_universal(a, kind)
        return fn(a, *args, **kwargs)

    factory.__doc__ = (fn.__doc__ or "") + _FACTORY_NOTE
    return factory


def install() -> None:
    """Attach ``to_universal()`` to the containers of every supported dtype."""
    for name in SUPPORTED:
        for prefix in ("DenseVector_", "DenseMatrix_"):
            setattr(getattr(_core, prefix + name), "to_universal", to_universal)
