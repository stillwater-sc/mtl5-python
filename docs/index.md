# mtl5-python

Python bindings for [MTL5](https://github.com/stillwater-sc/mtl5), version 5
of the Matrix Template Library, with zero-copy interop with NumPy, SciPy, JAX
and PyTorch, and dispatch to hardware accelerators.

- **Zero-copy containers.** `mtl5.vector()` and `mtl5.matrix()` wrap NumPy
  memory without copying it, and NumPy and scikit-learn accept the containers
  back through `__array__`.
- **Number systems beyond IEEE.** Every container and many operations come in
  posit, fixed-point, logarithmic, takum and multi-word float types, next to
  `float32` and `float64`. That makes mixed-precision experiments a change of
  `dtype=`.
- **Dense and sparse linear algebra.** LU, QR, Cholesky, LDLᵀ, SVD and
  eigensolvers. Sparse direct factorizations with fill-reducing orderings and
  numeric refactorization. Krylov solvers with preconditioners, and SciPy
  sparse interop.

```bash
pip install mtl5
```

Prebuilt wheels cover Linux, macOS and Windows on CPython 3.10–3.12.

```{toctree}
:maxdepth: 2

getting-started
guide/index
api/index
```
