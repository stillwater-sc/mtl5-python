"""pandas ExtensionDtype/ExtensionArray for MTL5 Universal number types.

This module provides pandas extension types backed by MTL5's native
Universal number arrays (posit, fixpnt, lns, etc.). pandas Series and
DataFrames can store columns in these types and use standard pandas
operations (groupby, agg, describe) — internally values are converted
to float64 at boundaries with the rest of the pandas/NumPy ecosystem.

This is a pure-Python implementation built on the existing MTL5
DenseVector_* types. It does NOT register the type with NumPy (see
docs/designs/custom-dtype-feasibility.md for why).

Currently implemented for posit16 as a reference. The pattern extends
to other Universal types via the _make_extension_dtype factory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

import mtl5

if HAS_PANDAS:

    @register_extension_dtype
    class Posit16Dtype(ExtensionDtype):
        """pandas ExtensionDtype for MTL5 posit16."""

        name = "posit16"
        type = float
        kind = "f"  # floating
        _metadata = ()

        @classmethod
        def construct_array_type(cls):
            return Posit16Array

        @classmethod
        def construct_from_string(cls, string):
            if string == cls.name:
                return cls()
            raise TypeError(f"Cannot construct Posit16Dtype from '{string}'")

        def __repr__(self) -> str:
            return "Posit16Dtype()"

    class Posit16Array(ExtensionArray):
        """pandas ExtensionArray backed by mtl5.DenseVector_posit16.

        Stores values as float64 internally for fast pandas-side ops, and
        round-trips through MTL5 posit16 for arithmetic operations that
        should reflect the limited precision.
        """

        def __init__(self, values, copy: bool = False):
            if isinstance(values, Posit16Array):
                self._data = values._data.copy() if copy else values._data
            else:
                # Convert through mtl5 to enforce posit16 quantization
                arr = np.asarray(values, dtype=np.float64)
                if arr.ndim != 1:
                    raise ValueError("Posit16Array requires 1-D input")
                vec = mtl5.vector_posit16(arr)
                self._data = vec.to_numpy().astype(np.float64)

        # ---- Required ExtensionArray API -----------------------------------
        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            return cls(scalars, copy=copy)

        @classmethod
        def _from_factorized(cls, values, original):
            return cls(values)

        def __getitem__(self, item):
            result = self._data[item]
            if np.ndim(result) == 0:
                return float(result)
            return type(self)(result)

        def __setitem__(self, key, value):
            # Quantize the value through posit16 before storing
            if np.isscalar(value):
                vec = mtl5.vector_posit16(np.array([float(value)]))
                self._data[key] = vec.to_numpy()[0]
            else:
                vec = mtl5.vector_posit16(np.asarray(value, dtype=np.float64))
                self._data[key] = vec.to_numpy()

        def __len__(self) -> int:
            return len(self._data)

        def __eq__(self, other):
            if isinstance(other, Posit16Array):
                return self._data == other._data
            return self._data == other

        @property
        def dtype(self) -> Posit16Dtype:
            return Posit16Dtype()

        @property
        def nbytes(self) -> int:
            return 2 * len(self._data)  # 16 bits per element

        def isna(self) -> np.ndarray:
            return np.isnan(self._data)

        def take(self, indices, *, allow_fill: bool = False, fill_value: Any = None):
            from pandas.api.extensions import take as pd_take

            if fill_value is None:
                fill_value = np.nan
            result = pd_take(self._data, indices, allow_fill=allow_fill, fill_value=fill_value)
            return type(self)(result)

        def copy(self):
            return type(self)(self._data.copy())

        @classmethod
        def _concat_same_type(cls, to_concat):
            data = np.concatenate([arr._data for arr in to_concat])
            return cls(data)

        # ---- Conversion ----------------------------------------------------
        def __array__(self, dtype=None, copy=None):
            if dtype is None or dtype == np.float64:
                return self._data.copy() if copy else self._data
            return self._data.astype(dtype, copy=copy if copy is not None else True)

        def to_numpy(self, dtype=None, copy: bool = False, na_value=None):
            if dtype is None:
                dtype = np.float64
            result = self._data.astype(dtype) if dtype != np.float64 else self._data
            return result.copy() if copy else result

        # ---- Arithmetic via posit16 round-trip ------------------------------
        def _arith(self, other, op):
            if isinstance(other, Posit16Array):
                rhs = other._data
            elif np.isscalar(other):
                rhs = float(other)
            else:
                rhs = np.asarray(other, dtype=np.float64)
            result = op(self._data, rhs)
            return type(self)(result)

        def __add__(self, other):
            return self._arith(other, np.add)

        def __sub__(self, other):
            return self._arith(other, np.subtract)

        def __mul__(self, other):
            return self._arith(other, np.multiply)

        def __truediv__(self, other):
            return self._arith(other, np.divide)

        def __repr__(self) -> str:
            preview = ", ".join(f"{x:.4g}" for x in self._data[:6])
            ellipsis = ", ..." if len(self._data) > 6 else ""
            return f"<Posit16Array: [{preview}{ellipsis}], len={len(self._data)}>"


def _ensure_pandas() -> None:
    """Raise a clear error if pandas is not installed."""
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for mtl5.pandas_ext. Install with: pip install 'mtl5[pandas]'"
        )
