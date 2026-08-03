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
