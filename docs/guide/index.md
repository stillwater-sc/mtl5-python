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
scipy-interop
```

```{toctree}
:maxdepth: 1
:caption: Test matrices

../notebooks/generators
```
