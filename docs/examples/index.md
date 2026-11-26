# Examples

MTL5's [`examples/`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples) teach the library through applications: a heat
equation, a vibrating string, a Kalman filter. The notebooks here bring those
examples to Python. Each one solves the same problem with mtl5, plots what the
C++ program prints, and asserts the program's claim, so a change that breaks
the claim fails the site's build.

Every notebook has the same parts:

1. **The problem**: the application and the question it asks.
2. **Uses**: the mtl5 functions it calls.
3. Building the problem, solving it, and looking at the result.
4. **Checks**: the claim, asserted in code. Only quantities that don't depend
   on the machine are checked (counts, errors, bounds, bytes), never times.
5. **From C++**: the MTL5 programs it comes from, and the C++ calls next to
   the Python ones.

The C++ links point to MTL5 `v5.25.0`, the version this release of mtl5 is built
against.

```{toctree}
:maxdepth: 1
:caption: First steps

../notebooks/examples/basics
../notebooks/examples/views
../notebooks/examples/backends
```

```{toctree}
:maxdepth: 1
:caption: Dense linear algebra

../notebooks/examples/factorizations
../notebooks/examples/vibrating_string
../notebooks/examples/pca
```

```{toctree}
:maxdepth: 1
:caption: PDEs and iterative solvers

../notebooks/examples/heat_transport
../notebooks/examples/poisson_relaxation
../notebooks/examples/krylov_comparison
../notebooks/examples/multigrid
```

```{toctree}
:maxdepth: 1
:caption: Sparse structure

../notebooks/examples/sparse_formats
../notebooks/examples/fem_assembly
../notebooks/examples/circuit_klu
```

```{toctree}
:maxdepth: 1
:caption: Numerical robustness and interop

../notebooks/examples/accurate_sum_dot
../notebooks/examples/ukf_cholesky_vs_ldlt
../notebooks/examples/zero_copy
```

## Every MTL5 example, and where it is

| MTL5 example | Here |
| --- | --- |
| [`intro/hello_mtl5.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/intro/hello_mtl5.cpp) | [Basics](../notebooks/examples/basics.ipynb) |
| [`phase01_core_types/vectors_and_matrices.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase01_core_types/vectors_and_matrices.cpp) | [Basics](../notebooks/examples/basics.ipynb) |
| [`phase01_core_types/concepts_and_traits.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase01_core_types/concepts_and_traits.cpp) | Not ported: C++20 concepts are compile-time checks |
| [`phase02_basic_operations/norms_and_products.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase02_basic_operations/norms_and_products.cpp) | [Basics](../notebooks/examples/basics.ipynb) |
| [`phase02_basic_operations/matrix_arithmetic.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase02_basic_operations/matrix_arithmetic.cpp) | [Basics](../notebooks/examples/basics.ipynb) |
| [`phase03_iterative_solvers/heat_equation_1d.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase03_iterative_solvers/heat_equation_1d.cpp) | [Heat and transport in 1D](../notebooks/examples/heat_transport.ipynb) |
| [`phase03_iterative_solvers/convection_diffusion.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase03_iterative_solvers/convection_diffusion.cpp) | [Heat and transport in 1D](../notebooks/examples/heat_transport.ipynb) |
| [`phase04_sparse_assembly/laplacian_2d.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase04_sparse_assembly/laplacian_2d.cpp) | [Poisson in 2D, and relaxation](../notebooks/examples/poisson_relaxation.ipynb) |
| [`phase04_sparse_assembly/smoother_convergence.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase04_sparse_assembly/smoother_convergence.cpp) | [Poisson in 2D, and relaxation](../notebooks/examples/poisson_relaxation.ipynb) |
| [`phase05_dense_decompositions/solve_three_ways.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase05_dense_decompositions/solve_three_ways.cpp) | [Three factorizations and a curve fit](../notebooks/examples/factorizations.ipynb); Part 3, preconditioners, in [Krylov solvers compared](../notebooks/examples/krylov_comparison.ipynb) |
| [`phase05_dense_decompositions/least_squares_qr.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase05_dense_decompositions/least_squares_qr.cpp) | [Three factorizations and a curve fit](../notebooks/examples/factorizations.ipynb) |
| [`phase06_eigenvalue_svd/vibrating_string.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase06_eigenvalue_svd/vibrating_string.cpp) | [Vibrating string](../notebooks/examples/vibrating_string.ipynb) |
| [`phase06_eigenvalue_svd/pca_svd.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase06_eigenvalue_svd/pca_svd.cpp) | [PCA by SVD](../notebooks/examples/pca.ipynb) |
| [`phase07_sparse_formats/sparse_formats.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase07_sparse_formats/sparse_formats.cpp) | [Sparse formats](../notebooks/examples/sparse_formats.ipynb) |
| [`phase07_sparse_formats/structured_views.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase07_sparse_formats/structured_views.cpp) | [Views](../notebooks/examples/views.ipynb) |
| [`phase08_expression_templates/expression_benchmark.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase08_expression_templates/expression_benchmark.cpp) | Not ported: expression templates fuse C++ loops at compile time |
| [`phase08_expression_templates/expression_concepts.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase08_expression_templates/expression_concepts.cpp) | Not ported: expression templates fuse C++ loops at compile time |
| [`phase09_advanced_types/sparse_vector.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase09_advanced_types/sparse_vector.cpp) | Not ported yet: no sparse vector binding |
| [`phase09_advanced_types/recursive_traversal.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase09_advanced_types/recursive_traversal.cpp) | Not ported: a C++ cache-oblivious traversal |
| [`phase09_advanced_types/blas_dispatch.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase09_advanced_types/blas_dispatch.cpp) | [Backends](../notebooks/examples/backends.ipynb) |
| [`phase10_matrix_views/triangular_views.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase10_matrix_views/triangular_views.cpp) | [Views](../notebooks/examples/views.ipynb) |
| [`phase10_matrix_views/permutation_block_diagonal.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase10_matrix_views/permutation_block_diagonal.cpp) | [Views](../notebooks/examples/views.ipynb) |
| [`phase11_assembly_reorder/fem_assembly_reorder.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase11_assembly_reorder/fem_assembly_reorder.cpp) | [FEM assembly and reordering](../notebooks/examples/fem_assembly.ipynb) |
| [`phase11_assembly_reorder/strided_views_unit_vectors.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase11_assembly_reorder/strided_views_unit_vectors.cpp) | Not ported: NumPy slicing gives strided views |
| [`phase12_transcendental_functions/signal_processing.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase12_transcendental_functions/signal_processing.cpp) | Not ported: NumPy's element-wise functions |
| [`phase12_transcendental_functions/activation_functions.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase12_transcendental_functions/activation_functions.cpp) | Not ported: NumPy's element-wise functions |
| [`phase12_transcendental_functions/radioactive_decay.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase12_transcendental_functions/radioactive_decay.cpp) | Not ported: NumPy's element-wise functions |
| [`phase12_transcendental_functions/coordinate_geometry.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase12_transcendental_functions/coordinate_geometry.cpp) | Not ported: NumPy's element-wise functions |
| [`phase13_krylov_multigrid/krylov_comparison.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase13_krylov_multigrid/krylov_comparison.cpp) | [Krylov solvers compared](../notebooks/examples/krylov_comparison.ipynb) |
| [`phase13_krylov_multigrid/multigrid_poisson.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase13_krylov_multigrid/multigrid_poisson.cpp) | [Multigrid for Poisson](../notebooks/examples/multigrid.ipynb) |
| [`phase14_io_generators/csv_io_prettyprint.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase14_io_generators/csv_io_prettyprint.cpp) | Not ported: NumPy reads and writes CSV; Matrix Market is `mtl5.io`, used in [the orderings notebook](../notebooks/sparse_direct_orderings.ipynb) |
| [`phase14_io_generators/poisson_diag_toolkit.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase14_io_generators/poisson_diag_toolkit.cpp) | Covered by [Test-matrix generators](../notebooks/generators.ipynb) |
| [`phase15_sparse_direct/sparse_direct_solvers.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase15_sparse_direct/sparse_direct_solvers.cpp) | Covered by [Sparse direct solvers](../guide/direct-solvers.md) |
| [`phase15_sparse_direct/circuit_matrix_klu.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/phase15_sparse_direct/circuit_matrix_klu.cpp) | [Circuit matrices with KLU](../notebooks/examples/circuit_klu.ipynb) |
| [`accurate_sum_dot/accurate_sum_dot.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/accurate_sum_dot/accurate_sum_dot.cpp) | [Accurate sum and dot product](../notebooks/examples/accurate_sum_dot.ipynb) |
| [`ukf_cholesky_vs_ldlt/ukf_comparison.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/ukf_cholesky_vs_ldlt/ukf_comparison.cpp) | [UKF sigma points: Cholesky against LDLᵀ](../notebooks/examples/ukf_cholesky_vs_ldlt.ipynb) |
| [`ukf_bearing_only/bearing_only_ukf.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/ukf_bearing_only/bearing_only_ukf.cpp) | [UKF sigma points: Cholesky against LDLᵀ](../notebooks/examples/ukf_cholesky_vs_ldlt.ipynb) |
| [`zero_copy/ode_rhs_callback.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/zero_copy/ode_rhs_callback.cpp) | [Zero copy](../notebooks/examples/zero_copy.ipynb) |
| [`zero_copy/shared_frame_pipeline.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/zero_copy/shared_frame_pipeline.cpp) | [Zero copy](../notebooks/examples/zero_copy.ipynb) |
| [`zero_copy/zero_copy_rules.cpp`](https://github.com/stillwater-sc/mtl5/blob/v5.25.0/examples/zero_copy/zero_copy_rules.cpp) | [Zero copy](../notebooks/examples/zero_copy.ipynb) |

The examples that are not ported teach C++ itself (concepts, expression
templates, compile-time traversal), or what NumPy already does in Python.
