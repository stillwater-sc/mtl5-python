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

## `adder_dcop_01.mtx`, `adder_dcop_02.mtx`: circuit matrices

Read by `examples/circuit_klu.ipynb`, and `adder_dcop_01.mtx` also by
`examples/sparse_formats.ipynb`.

| File | Source | Description |
|---|---|---|
| `adder_dcop_01.mtx` | SuiteSparse Matrix Collection, [`Sandia/adder_dcop_01`](https://sparse.tamu.edu/Sandia/adder_dcop_01) (id 984) | Order 1813, 11,156 entries. The Jacobian of a DC operating point solve for an adder circuit, from Sandia National Laboratories' circuit simulations (R. Hoekstra, 2003). Unsymmetric, cond ≈ 3 × 10⁷, 469 diagonal blocks in block triangular form. |
| `adder_dcop_02.mtx` | SuiteSparse Matrix Collection, [`Sandia/adder_dcop_02`](https://sparse.tamu.edu/Sandia/adder_dcop_02) (id 985) | The next matrix of the same sequence: order 1813, 11,246 entries, cond ≈ 3 × 10¹⁴. |

Both are byte-for-byte the `.mtx` files in the collection's Matrix Market
archives (`adder_dcop_01.tar.gz`, `adder_dcop_02.tar.gz`), with SHA-256:

- `adder_dcop_01.mtx`: `ea3ffaae3876647f7c341faca0edd8d6918a0568e8240edf047a2fa211831fcd`
- `adder_dcop_02.mtx`: `f6c655f752786d9d61f12f33b574c65f720ee544e82aa457128ef3c020698275`

The SuiteSparse Matrix Collection is licensed CC BY 4.0. Reference: T. A. Davis
and Y. Hu, "The University of Florida Sparse Matrix Collection", *ACM
Transactions on Mathematical Software* 38(1), 2011, 1:1–1:25.

## `mtl5-bench/`: a sample of MTL5's benchmark data

Read by `benchmark_analysis.ipynb`. It's a subset of MTL5's committed
`benchmarks/data`, in MTL5's own layout, and the files are byte-for-byte copies
from stillwater-sc/mtl5 at `1dcc9a0`. The files are unchanged there since
`20feb69`. Each result file comes with its `.sysinfo` sidecar. The sample is
chosen so every figure of the notebook has its data:

| Files | For |
|---|---|
| `i7-12700k/mtl5_bench/20261112T004852Z/native-t{1,8}.json`, `…/20261111T190041Z/native-t8.json` | the capability card, the roofline and thread scaling; two 8-thread sessions at one commit for the cross-machine part |
| `jetson-orin/mtl5_bench/2026111{1T190610Z,1T193624Z}/native-t6.json` | two 6-thread sessions at `bfa1c28` |
| `arduino-ventuno-q/mtl5_bench/20261111T{185721Z,191603Z}/native-t4.json` | two 4-thread sessions at `bfa1c28` |
| `arduino-ventuno-q/mtl5_bench/*/native-fast-t1.json` (six sessions) | a trend over three commits (`bfa1c28`, `967d20a`, `bf315e4`), two sessions each, with a real series break: from `bf315e4` the build names its target (`-march=armv8.2-a+dotprod…`) |
| `blas_sweep_blis.csv` | rate against problem size; a `bench_all` sweep whose sidecar predates the provenance rules, so it's in the incomplete tier |

The notebook reads a full MTL5 checkout instead when `MTL5_BENCH_DATA` names its
`benchmarks/data` directory. To refresh the sample, copy the same paths from a
newer MTL5 checkout and update the commit above.
