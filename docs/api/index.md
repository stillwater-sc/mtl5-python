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
   SolveReport
   ilu0
   ic0
   ilut
```

## Accurate sums and dot products

```{eval-rst}
.. currentmodule:: mtl5

.. autosummary::
   :toctree: generated
   :nosignatures:

   sum_k
   dot_k
```

`mtl5.mixed.dot(x, y, accumulator="compensated")` is Dot2 as an accumulator,
and `accumulator="exact"` rounds the exact dot product once.
`mtl5.generators.gen_dot` and `mtl5.generators.gen_sum` build the
ill-conditioned test data of Ogita, Rump and Oishi.

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

The CSV dialects MTL5's older harnesses write, and `bench_blas.py`'s JSON, are
described in `mtl5.bench.formats`. Their values land in the `measurements`
table, one row per file row and metric.

```{eval-rst}
.. currentmodule:: mtl5.bench.formats

.. autosummary::
   :toctree: generated
   :nosignatures:

   Format
   FORMATS
```

### Figures

The core figures are drawn in two steps. `mtl5.bench.analyze` reads the
database into a DataFrame with one row per plotted point. `mtl5.bench.plot`
then draws a figure from that frame and nothing else. A figure's
{meth}`~mtl5.bench.plot.BenchFigure.save` writes vector PDF and SVG, a 300 dpi
PNG and the frame as CSV, and {func}`~mtl5.bench.plot.from_csv` draws the same
figure, byte for byte, from that CSV. Every figure carries a provenance footer
(machines, commits, runs, dates), which warns when a run's provenance is
incomplete. `mtl5-bench figure {card,roofline,scaling,rate}` wraps both steps.

```{eval-rst}
.. currentmodule:: mtl5.bench.analyze

.. autosummary::
   :toctree: generated
   :nosignatures:

   capability
   roofline
   thread_scaling
   rate_vs_size

.. currentmodule:: mtl5.bench.plot

.. autosummary::
   :toctree: generated
   :nosignatures:

   BenchFigure
   capability_card
   roofline
   thread_scaling
   rate_vs_size
   from_csv
   provenance

.. currentmodule:: mtl5.bench.style

.. autosummary::
   :toctree: generated
   :nosignatures:

   house_style
   save
```

### Across machines

Each machine contributes its comparable runs: its latest commit's sessions
(MTL5 takes two per commit) at one thread count, by default the most it has
a run of. Tables and figures show each machine's median over those runs, with
the spread beside it. `mtl5-bench table {machines,compare}` writes the tables
as Markdown or CSV, and `mtl5-bench figure {balance,compare}` draws the
figures.

```{eval-rst}
.. currentmodule:: mtl5.bench.analyze

.. autosummary::
   :toctree: generated
   :nosignatures:

   machines
   cross_machine

.. currentmodule:: mtl5.bench.plot

.. autosummary::
   :toctree: generated
   :nosignatures:

   machine_balance
   cross_machine

.. currentmodule:: mtl5.bench.tables

.. autosummary::
   :toctree: generated
   :nosignatures:

   machine_table
   compare_table
   to_markdown
```

### Trends

A trend is one entry's history on one machine, at one label and thread
count, read from the `trend_results` view only: complete, checked runs.
{mod}`mtl5.bench.trends` groups the runs into series by their comparability
key, so a change of build or machine state is a **series break**, not a
change, and into points by commit. Each point gets a 95% confidence interval
from the run-to-run noise of its series. A **change point** is a step within
a series of at least `k` standard errors of that noise (default 4) and at
least `min_effect` in size (default 2%). `mtl5-bench trend ENTRY --machine M`
prints the summary and the change points and draws the figure;
`--fail-on-regression` makes it a check.

```{eval-rst}
.. currentmodule:: mtl5.bench.analyze

.. autosummary::
   :toctree: generated
   :nosignatures:

   trend

.. currentmodule:: mtl5.bench.trends

.. autosummary::
   :toctree: generated
   :nosignatures:

   points
   noise
   change_points
   summary
   t975

.. currentmodule:: mtl5.bench.plot

.. autosummary::
   :toctree: generated
   :nosignatures:

   trend
```
