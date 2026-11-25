# API reference

Generated from the docstrings of the compiled module. Most functions are
overloaded once per element type; the signatures list every overload.

## Containers and conversion

```{eval-rst}
.. currentmodule:: mtl5

.. autosummary::
   :toctree: generated
   :nosignatures:

   vector
   matrix
   convert
   dtypes
   DenseVector_f64
   DenseMatrix_f64
```

`mtl5.DenseVector` and `mtl5.DenseMatrix` are aliases of the `float64`
classes above. Every other element type has the same pair of classes with a
different suffix, such as `DenseMatrix_posit16`. `mtl5.dtypes()` lists the
suffixes.

## Dense linear algebra

```{eval-rst}
.. currentmodule:: mtl5

.. autosummary::
   :toctree: generated
   :nosignatures:

   dot
   norm
   matmul
   matvec
   solve
   lu
   qr
   cholesky
   ldlt
   svd
   eig
   eigh
   inv
   det
   condition_number
```

## Sparse matrices and direct solvers

```{eval-rst}
.. currentmodule:: mtl5.sparse

.. autosummary::
   :toctree: generated
   :nosignatures:

   csr_matrix
   coo_matrix
   from_scipy
   to_scipy
   splu
   klu
   cholesky
   ldlt
   qr
   supernodal_lu
   supernodal_ldlt
   orderings
   ordering
   amd
   colamd
   rcm
   nested_dissection
   SparseLU_f64
   SupernodalLU_f64
   SparseCholesky_f64
   SparseLDLT_f64
   SupernodalLDLT_f64
   SparseQR_f64
   KLU_f64
```

## Sparse iterative solvers

```{eval-rst}
.. currentmodule:: mtl5.sparse

.. autosummary::
   :toctree: generated
   :nosignatures:

   cg
   gmres
   bicgstab
   minres
   iterative_solve
   ilu0
   ic0
   ilut
```

## Mixed precision

```{eval-rst}
.. currentmodule:: mtl5.mixed

.. autosummary::
   :toctree: generated
   :nosignatures:

   iterative_refine
   lu_iterative_refine
   dot
   matmul
   accumulators
```

## Input and output

```{eval-rst}
.. currentmodule:: mtl5.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   mm_read
   mm_write
   spy
   spy_magnitude
   spy_density
```

## Benchmark data

Ingestion of MTL5's benchmark results into a DuckDB database, with the
provenance that decides which measurements are comparable (#121). It needs the
`bench` extra: `pip install 'mtl5[bench]'`. The `mtl5-bench ingest` command
wraps {func}`mtl5.bench.ingest`.

```{eval-rst}
.. currentmodule:: mtl5.bench

.. autosummary::
   :toctree: generated
   :nosignatures:

   ingest
   connect
   IngestReport
   FileOutcome
```
