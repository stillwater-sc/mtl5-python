"""Index-notation tensor algebra (`mtl/tensor`).

Distinct from `mtl5.array`: that is NumPy-shaped N-D data with a runtime shape;
this is fixed-dimension tensor algebra with Einstein summation and a metric.

Rank and dimension are both compile-time in MTL5, so each `(rank, dim)` pair is
a separate type. Ranks 1, 2 and 4 over dimensions 2, 3 and 4 cover the module —
nothing upstream uses rank 3, and rank 4 arises only as `outer(rank2, rank2)`.

```python
import numpy as np, mtl5.tensor as mtt

A = mtt.asarray(np.arange(9.0).reshape(3, 3))
x = mtt.asarray(np.array([1.0, 2.0, 3.0]))

mtt.contract(A, "ij", x, "j")      # A @ x
mtt.contract(A, "ji", x, "j")      # A.T @ x
mtt.lower_index(x, mtt.euclidean_metric(3))
```

A metric must match its vector's dimension. `minkowski_metric()` is 4-D, so it
pairs with a four-component vector:

```python
v = mtt.asarray(np.array([1.0, 2.0, 3.0, 4.0]))
mtt.lower_index(v, mtt.minkowski_metric())   # [-1, 2, 3, 4]
```

`contract` takes its index strings at runtime. MTL5's own `contract` takes index
names as compile-time template parameters, which a Python string cannot reach —
but the space is small enough to enumerate, so this dispatches to MTL5's real
contraction rather than reimplementing it.
"""

from __future__ import annotations

import numpy as np

from mtl5._core import tensor as _t
from mtl5._core.tensor import (  # noqa: F401
    contract,
    dimensions,
    euclidean_metric,
    is_antisymmetric,
    is_symmetric,
    lower_first,
    lower_index,
    lower_second,
    minkowski_metric,
    outer,
    raise_first,
    raise_index,
    raise_second,
    ranks,
    zeros,
)

_RANKS = (1, 2, 4)
_DIMS = (2, 3, 4)


def asarray(a):
    """Build a tensor from a NumPy array of shape ``(dim,) * rank``.

    Rank comes from ``a.ndim`` and dimension from its extents, which must all be
    equal — a tensor has one dimension shared by every index.
    """
    a = np.ascontiguousarray(a)
    if a.dtype.name not in ("float32", "float64"):
        raise TypeError(f"tensor: dtype must be float32 or float64, got {a.dtype.name}")
    if a.ndim not in _RANKS:
        raise ValueError(
            f"tensor: rank must be one of {_RANKS}, got {a.ndim}. Rank 3 is "
            "absent because nothing in mtl/tensor produces or consumes it; use "
            "mtl5.array for general N-D data."
        )
    if len(set(a.shape)) != 1:
        raise ValueError(
            f"tensor: every extent must be equal, got shape {a.shape}. A tensor "
            "has a single dimension shared by all of its indices."
        )
    if a.shape[0] not in _DIMS:
        raise ValueError(f"tensor: dimension must be one of {_DIMS}, got {a.shape[0]}")
    return _t._from_numpy(a)


__all__ = [
    "asarray",
    "contract",
    "dimensions",
    "euclidean_metric",
    "is_antisymmetric",
    "is_symmetric",
    "lower_first",
    "lower_index",
    "lower_second",
    "minkowski_metric",
    "outer",
    "raise_first",
    "raise_index",
    "raise_second",
    "ranks",
    "zeros",
]
