---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Getting started

This page is executed every time the site is built, so every output below
comes from the current release.

## Install

```bash
pip install mtl5                # the core package
pip install "mtl5[scipy]"       # SciPy sparse interop
pip install "mtl5[all]"         # scipy, torch, jax, pandas, scikit-learn
```

To build from source you need a C++20 compiler (GCC 12+, Clang 15+ or MSVC
2022); then run `pip install .` in a checkout.

## Vectors and matrices

`mtl5.vector()` and `mtl5.matrix()` wrap a contiguous NumPy array without
copying it. Operations accept either the containers or NumPy arrays.

```{code-cell} python
import numpy as np

import mtl5

a = np.array([3.0, 4.0])
v = mtl5.vector(a)
print(mtl5.norm(a), mtl5.dot(a, a))
print("shares memory with the NumPy array:", np.shares_memory(np.asarray(v), a))
```

## Solving a dense system

```{code-cell} python
A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
b = np.array([1.0, 2.0, 3.0])
x = mtl5.solve(A, b)
print(x, "residual:", np.linalg.norm(A @ x - b))
```

## Solving a sparse system

`mtl5.sparse` takes SciPy sparse matrices directly. A factorization object
solves repeatedly, and `refactor()` reuses its analysis when only the values
change:

```{code-cell} python
import scipy.sparse as sp

import mtl5.sparse as ms

n = 1000
T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
lu = ms.splu(T)  # colamd ordering by default
x = lu.solve(np.ones(n)).to_numpy()
print(lu, "residual:", np.linalg.norm(T @ x - np.ones(n)))
```

The [sparse direct solvers guide](notebooks/sparse_direct_orderings) shows
what the ordering does to the factors, and when to choose which.

## Other number systems

Every element type has its own containers. `mtl5.dtypes()` lists the ones
`convert()` can produce:

```{code-cell} python
print(mtl5.dtypes())
p = mtl5.convert(np.array([[1.0 / 3.0]]), "posit16")
print(type(p).__name__, p.to_numpy())
```
