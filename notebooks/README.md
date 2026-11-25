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
