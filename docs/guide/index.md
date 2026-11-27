# User guide

The guides are executed when the site is built, so their numbers come from the
current release. Some are pages of this site, and some are Jupyter notebooks
from the repository's `notebooks/` directory. To run a notebook yourself,
`pip install -e ".[notebooks]"` and open it with `jupyter lab notebooks/`.

```{toctree}
:maxdepth: 1
:caption: Sparse direct solvers

direct-solvers
../notebooks/sparse_direct_orderings
how-orderings-work
spd-example
mixed-precision
diagnostics
scipy-interop
```

```{toctree}
:maxdepth: 1
:caption: Test matrices

../notebooks/generators
../notebooks/conditioning
../notebooks/spectra_inertia
```

```{toctree}
:maxdepth: 1
:caption: Benchmark data

../notebooks/benchmark_analysis
```
