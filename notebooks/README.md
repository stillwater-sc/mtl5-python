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
| `conditioning.ipynb` | Condition numbers across eight number systems (fp16, posit16, takum16, f32, posit32, takum32, f64, dd_cascade), with seeded `randsvd` matrices. Covers the five `randsvd` spectra against each format's unit roundoff; the noise floor rounding puts under a spectrum; the condition number each format can actually hold; forward error against κ·u; and a precision-ladder heatmap of correct digits by format and κ. The first notebook of #166. |
| `spectra_inertia.ipynb` | Spectra and inertia of symmetric matrices: catalog and built-to-order spectra with their inertia; `randsym` with a chosen spectrum and seed; spectrum slicing by counting the negative pivots of A − σI, and bisection on that count, which splits a Wilkinson pair that `eigvalsh`'s default tolerance merges; when an LDLᵀ inertia in each of eight formats gets a small eigenvalue's sign wrong; Gershgorin discs. The second notebook of #166. |
| `nonnormality.ipynb` | Non-normal matrices, where eigenvalues mislead: the same spectrum at three degrees of non-normality with perturbed-eigenvalue clouds; pseudospectra of `kahan` and `frank`; the field of values and the numerical abscissa; transient growth of matrix powers with ρ < 1; eigenvalue condition numbers checked against measured perturbations and Frank's reciprocal pairs; what rounding to each of eight formats does to the eigenvalues. The third notebook of #166. |
| `stability.ipynb` | The stability of factorizations: backward against forward error of LU in each of eight formats; the growth factor of partial pivoting on random matrices and on Wilkinson's matrix, where LU fails and QR doesn't; pivot profiles against singular values; Cholesky, unpivoted LDLᵀ and Bunch–Kaufman on drifting inertia and on a well-conditioned saddle-point matrix. The fourth notebook of #166. |
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
| `examples/heat_transport.ipynb` | `phase03_iterative_solvers/*` | Variable-conductivity heat flow: CG against Jacobi-preconditioned CG, checked against the exact solution with second-order convergence; convection–diffusion, where CG fails and BiCGSTAB converges, and the cell Péclet number. |
| `examples/poisson_relaxation.ipynb` | `phase04_sparse_assembly/*` | The 2D 5-point Laplacian from `generators.poisson2d`: an exact-solution check, then GMRES(5), GMRES(20) and CG; Jacobi, Gauss–Seidel and SOR against their theoretical rates, and the smoothing property. |
| `examples/krylov_comparison.ipynb` | `phase13_krylov_multigrid/krylov_comparison`, `phase05_dense_decompositions/solve_three_ways` (Part 3) | Nine solvers on a nonsymmetric system, MINRES on symmetric indefinite ones, and what IC(0), ILU(0) and SSOR buy on the Poisson matrices. |
| `examples/multigrid.ipynb` | `phase13_krylov_multigrid/multigrid_poisson` | Restriction, prolongation and Galerkin coarsening; V-cycles whose count is independent of n with a full hierarchy (and not with 3 fixed levels); a symmetric V-cycle as a CG preconditioner. |
| `examples/sparse_formats.ipynb` | `phase07_sparse_formats/sparse_formats` | The 2-D Laplacian in COO, CSR and ELL; storage per format; ELL's padding on a circuit matrix's irregular rows; Matrix Market round trips. Reads `data/adder_dcop_01.mtx`. |
| `examples/fem_assembly.ipynb` | `phase11_assembly_reorder/fem_assembly_reorder` | FEM assembly by COO triplets; a P1 mesh numbered at random, checked against `generators.laplacian_2d`; RCM's bandwidth and profile; Cholesky fill by ordering; what reordering leaves unchanged. |
| `examples/circuit_klu.ipynb` | `phase15_sparse_direct/circuit_matrix_klu` | KLU's block triangular form on a Sandia circuit matrix; what it stores against `splu`; `refactor`, its pivot growth, and the case where its answer is wrong. Reads `data/adder_dcop_0{1,2}.mtx`. |
| `examples/accurate_sum_dot.ipynb` | `accurate_sum_dot` | Section 6 of Ogita, Rump and Oishi: Dot2 and DotK against the condition number, with their error bounds checked on every sample against an exact reference; the same in float32; Table 6.2's correct digits of summation; the cost of each algorithm. |
| `examples/ukf_cholesky_vs_ldlt.ipynb` | `ukf_cholesky_vs_ldlt/ukf_comparison`, `ukf_bearing_only/bearing_only_ukf` | Cholesky and LDLᵀ residuals on Hilbert matrices up to cond 1e16; Cholesky refusing an indefinite matrix and LDLᵀ's inertia; the two examples' filters with each factorization, in float64 and float32; why the float32 filter fails: the UKF weights at alpha = 1e-3. |
| `examples/zero_copy.ipynb` | `zero_copy/*` | The rules of a view over NumPy memory (write-through, refusals instead of copies, explicit copies, aliasing, strided arrays, lifetime); an RK4 integrator whose callback writes its buffer in place with `symv`; frames corrected in a shared region by two threads. |
