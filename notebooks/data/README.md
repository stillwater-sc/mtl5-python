# Notebook data

Test matrices that the notebooks read. They are stored here so the notebooks,
and CI, which runs them, need no network access.

| File | Source | Description |
|---|---|---|
| `west0479.mtx` | Harwell-Boeing collection (`CHEMWEST` set), via NIST [Matrix Market](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/chemwest/west0479.html) | Order 479, 1910 stored entries (22 of them explicit zeros). An eight-stage Westerberg chemical distillation column model, unsymmetric. |

`west0479.mtx` is byte-for-byte the file Matrix Market distributes
(`west0479.mtx.gz`, decompressed), SHA-256
`ce4b2b49d990753c663326d4625eded52fafcf8acf195c7c15292b64e7c12488`. The same
matrix is `HB/west0479` in the SuiteSparse Matrix Collection, and it is the
example on MATLAB's `colamd` page.

Reference: I. S. Duff, R. G. Grimes and J. G. Lewis, "Sparse matrix test
problems", *ACM Transactions on Mathematical Software* 15(1), 1989, 1–14.
