# Notebooks

Runnable documentation. Outputs are **not** committed — the notebooks are kept
clean so diffs stay reviewable, and every cell is verified to execute in CI's
Python versions before merge.

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

| Notebook | What it covers |
|---|---|
| `generators.ipynb` | The test-matrix catalog: matrices with exactly known spectra, conditioning on demand, matrices built to a prescribed spectrum or condition number, and the `dtype=` story that makes them useful for mixed-precision work. |
| `sparse_direct_orderings.ipynb` | Fill-reducing orderings and the sparse direct solve pipeline, after MATLAB's `colamd` example. Covers spy plots of `west0479` and its `L` and `U` factors under `natural`, `colamd`, `amd` and `rcm`; a symmetric problem where `amd` wins instead; and what `refactor()` saves when the pattern is fixed. Reads `data/west0479.mtx` (see `data/README.md`). |
| `benchmark_analysis.ipynb` | `mtl5.bench` end to end, on a sample of MTL5's committed benchmark data. It covers ingest and the provenance tiers; a machine's capability card, roofline and thread scaling; cross-machine tables, the machine balance and a comparison; a trend report with confidence intervals, a real series break and change-point detection; and publication output, with a figure redrawn byte for byte from its saved data. Reads `data/mtl5-bench` (see `data/README.md`), or a full MTL5 checkout through `MTL5_BENCH_DATA`. |

## `examples/`: MTL5's examples, in Python

Ports of the programs in MTL5's
[`examples/`](https://github.com/stillwater-sc/mtl5/tree/v5.25.0/examples),
one application per notebook. The site lists them in its Examples section
(`docs/examples/index.md`), with every MTL5 example and where it went
(issue #147). Each notebook names the MTL5 programs it ports in its metadata,
as `mtl5_examples`, and `tests/test_examples_index.py` checks that against the
index.

| Notebook | Ports | What it covers |
|---|---|---|
| `examples/basics.ipynb` | `intro/hello_mtl5`, `phase01_core_types/vectors_and_matrices`, `phase02_basic_operations/*` | Containers over NumPy memory, norms and their unit balls, products, and dense against sparse storage. |
| `examples/views.ipynb` | `phase07_sparse_formats/structured_views`, `phase10_matrix_views/*` | Triangular, banded, Hermitian and mapped views; triangular solves; permutations as index vectors; block-diagonal systems. |
| `examples/backends.ipynb` | `phase09_advanced_types/blas_dispatch` | What the build dispatches to, and `mtl5.matmul` against NumPy's BLAS. |
| `examples/factorizations.ipynb` | `phase05_dense_decompositions/*` | LU, Cholesky and QR on one system; Cholesky refusing an indefinite matrix, and LDLᵀ's inertia; a polynomial fit by QR and by the normal equations, against a quad-double reference. |
| `examples/vibrating_string.ipynb` | `phase06_eigenvalue_svd/vibrating_string` | Eigenvalues and modes of the 1D Laplacian against the closed form; the 2D Laplacian by Kronecker products; complex eigenvalues from an asymmetric change; singular values against eigenvalues. |
| `examples/pca.ipynb` | `phase06_eigenvalue_svd/pca_svd` | PCA by SVD: variance explained, the covariance route, and the rank-2 reconstruction error equal to the dropped singular value. |
