---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Diagnostics: reading a factorization

A factorization that returns without an exception has not necessarily
succeeded. This page covers what a factor object can tell you, and which of
its numbers to trust:

- what `__repr__` shows, and what it leaves out;
- zero pivots, `pivot_perturb` and `num_perturbed`, and a singular matrix that
  factors without complaint;
- threshold pivoting, which trades stability for fill;
- the `accumulator=` policy, and when it changes the answer.

The page is executed when the documentation is built, and the `assert`
statements check the claims the prose makes.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import mtl5
import mtl5.sparse as ms

iterative_refine = mtl5.mixed.iterative_refine

SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#d8d7d2",
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "figure.dpi": 120,
    }
)


def laplacian_2d(k):
    """The k x k five-point Laplacian with Dirichlet boundaries, unscaled."""
    eye = sp.eye(k, format="csr")
    tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k), format="csr")
    return (sp.kron(eye, tri) + sp.kron(tri, eye)).tocsr()


def relres(A, x, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


def fwd_err(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
```

## Reading `__repr__`

Every factor object prints as its class name and a few numbers that describe
the factorization. Here is one of each, on a small Poisson problem:

```{code-cell} python
P = laplacian_2d(8)
factors = {
    "splu": ms.splu(P),
    "klu": ms.klu(P),
    "supernodal_lu": ms.supernodal_lu(P),
    "cholesky": ms.cholesky(P),
    "ldlt": ms.ldlt(P),
    "supernodal_ldlt": ms.supernodal_ldlt(P),
    "qr": ms.qr(P),
}
for name, F in factors.items():
    print(f"{name:>16}: {F!r}")
```

The fields mean the following:

| Field | Printed by | Meaning |
|---|---|---|
| `n` | all square factors | the order of the matrix |
| `shape` | `SparseQR` | rows and columns; QR also takes tall matrices |
| `ordering` | `SparseLU`, `SparseCholesky`, `SparseLDLT`, `SparseQR` | the fill-reducing ordering used |
| `nnz` | `SparseLU` | entries of L plus U, with L's unit diagonal stored |
| `nnz` | `SparseCholesky` | entries of L, diagonal included |
| `nnz` | `SparseLDLT` | entries of L **without** its unit diagonal, so it is n less than Cholesky's count on the same matrix |
| `nblocks` | `KLU` | diagonal blocks in the block triangular form; many small blocks mean BTF found structure |
| `nsuper` | `SupernodalLU` | supernodes formed; n of them would mean no dense blocks at all |

`SupernodalLDLT` prints only `n`. Its fill is available as `.nnz`, and the
other properties are listed below.

```{code-cell} python
r = {name: repr(F) for name, F in factors.items()}
n = P.shape[0]
assert f"nnz={factors['splu'].nnz}" in r["splu"]
assert f"nblocks={factors['klu'].nblocks}" in r["klu"]
assert f"nsuper={factors['supernodal_lu'].nsuper}" in r["supernodal_lu"]
assert r["supernodal_ldlt"].endswith(f"(n={n})")
assert factors["ldlt"].nnz == factors["cholesky"].nnz - n
assert factors["supernodal_ldlt"].nnz == factors["cholesky"].nnz - n
```

### What the repr leaves out

Three diagnostics are properties only, and none of them appears in the repr:

- `num_perturbed`, on `splu`, `klu` and `supernodal_lu`: how many pivots
  `pivot_perturb` replaced. It is 0 for a clean factor.
- `accumulator`, on the same three and on `supernodal_ldlt`: the
  accumulator policy the factor was built with.
- `dtype`: `"f32"` or `"f64"`, the precision of the stored factor. It follows
  the dtype of the matrix you passed in.

A short helper prints whichever of these a factor has:

```{code-cell} python
DIAGNOSTICS = ("dtype", "ordering", "nnz", "nblocks", "nsuper", "num_perturbed", "accumulator")


def report(F):
    fields = {k: getattr(F, k) for k in DIAGNOSTICS if hasattr(F, k)}
    return type(F).__name__ + ": " + ", ".join(f"{k}={v!r}" for k, v in fields.items())


for F in factors.values():
    print(report(F))
```

## Did the factorization succeed?

### A zero pivot raises

In LU with partial pivoting, a pivot is zero only when every remaining
candidate in its column is zero. That means the columns factored so far are
linearly dependent in the factor's precision. By default the factorization
raises. Here is a 2×2 matrix that is nonsingular in float64 and exactly
singular in float32, because 1 + 10⁻⁸ rounds to 1:

```{code-cell} python
A2 = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0 + 1e-8]]))
x2_true = np.array([1.0, 2.0])
b2 = A2 @ x2_true

raised = None
try:
    ms.splu(ms.from_scipy(A2.astype(np.float32)))
except RuntimeError as err:
    raised = str(err)
print(raised)
assert raised is not None and "zero pivot" in raised
assert ms.splu(A2).num_perturbed == 0  # float64 factors it cleanly
```

### `pivot_perturb` finishes the factorization, and `num_perturbed` says so

`pivot_perturb=p` replaces a pivot whose magnitude is below
`p * max|column|` with that floor, keeping its sign, and counts the
replacement in `num_perturbed`. A completely zero column still raises,
because there is nothing to scale the floor by. The point is to let a narrow
factor finish, and then rely on iterative refinement against a float64
residual to clean up the answer.

Whether refinement can actually do that depends mostly on the matrix. The
usual sufficient condition for convergence is κ·u well below 1, where u is
the factor's unit roundoff. It is conservative: refinement can also converge
when κ·u > 1, for instance when the narrow factor represents the matrix
exactly. Here it is far from met. This matrix has κ ≈ 4·10⁸, and float32's
unit roundoff is 6·10⁻⁸, so κ·u ≈ 24:

```{code-cell} python
A2_64 = ms.from_scipy(A2)
A2_32 = ms.from_scipy(A2.astype(np.float32))
print(f"{'pivot_perturb':>13}  num_perturbed  converged  rel. residual  forward error")
perturb_rows = []
for p in (1e-7, 1e-5, 1e-3):
    F = ms.splu(A2_32, pivot_perturb=p)
    x, info = iterative_refine(A2_64, F, b2, max_iter=100, rel_tol=1e-14)
    err = fwd_err(np.asarray(x), x2_true)
    perturb_rows.append((p, F.num_perturbed, info["converged"], err, info["rel_residual"]))
    print(
        f"{p:13.0e}  {F.num_perturbed:13d}  {info['converged']!s:>9}  "
        f"{info['rel_residual']:13.1e}  {err:13.1e}"
    )

assert all(row[1] == 1 and not row[2] for row in perturb_rows)
assert perturb_rows[0][3] < 1e-3 < perturb_rows[-1][3]  # a larger floor gets less far
kappa2 = 4.0 / 1e-8  # the 2x2's condition number, to leading order
assert all(row[3] <= 10 * kappa2 * row[4] for row in perturb_rows)  # error <~ kappa * residual
```

Every run completes and reports one perturbed pivot. None of them converges
within 100 iterations, and `converged=False` is how you find out. Refinement
computes its residual with the original A, so if it did converge, the result
would solve the original system. A perturbed factor only slows it. The larger
the floor, the worse the factor approximates A, and the slower the iteration
contracts, until it stops making progress. The smallest floor, about
float32's unit roundoff, gets furthest.

So read a nonzero `num_perturbed` as a warning: the factor is of a different
matrix, and the answer is only as good as the refinement made it. Check
`converged` and `rel_residual`, and remember that a small residual bounds
the error only through the conditioning. The relative forward error can be
as large as κ times the relative residual. Here that is 4·10⁸ × 2·10⁻¹³ ≈
10⁻⁴, close to the measured 4·10⁻⁵.

### No error does not mean nonsingular

The opposite case is the more dangerous one. The Laplacian with Neumann
boundaries, where every row sums to zero, is exactly singular: the constant
vector is in its null space. In floating point, though, the last pivot comes
out as rounding noise rather than an exact zero, so the default factorization
raises nothing:

```{code-cell} python
L = laplacian_2d(16)
L = (L - sp.diags(np.asarray(L.sum(axis=1)).ravel())).tocsr()  # zero row sums
L.eliminate_zeros()
n = L.shape[0]

F = ms.splu(L)
pivots = np.abs(ms.to_scipy(F.U).diagonal())
pivot_ratio = pivots.min() / pivots.max()
x = F.solve(np.ones(n)).to_numpy()
print(report(F))
print(f"smallest / largest pivot: {pivot_ratio:.1e}")
print(f"max |x| for b = ones:     {np.abs(x).max():.1e}")

assert F.num_perturbed == 0
assert pivot_ratio < 1e-12
assert np.abs(x).max() > 1e12
```

The factor reports `num_perturbed=0`, and `solve` returns a vector of size
10¹⁶ for a right-hand side of ones. Two cheap checks catch it:

- **The pivot ratio.** `U.diagonal()` holds the pivots. A smallest-to-largest
  ratio near the unit roundoff means the matrix is singular to working
  precision. The ratio is not a condition number: a small ratio always
  deserves a look, but a large one does not prove the matrix well
  conditioned.
- **A small `pivot_perturb` as a detector.** With `pivot_perturb=1e-10`, the
  same factorization reports the tiny pivot:

```{code-cell} python
detected = {
    "splu": ms.splu(L, pivot_perturb=1e-10).num_perturbed,
    "klu": ms.klu(L, pivot_perturb=1e-10).num_perturbed,
    "supernodal_lu": ms.supernodal_lu(L, pivot_perturb=1e-10).num_perturbed,
}
print(detected)
assert all(v == 1 for v in detected.values())

chol_error = None
try:
    ms.cholesky(L)
except RuntimeError as err:
    chol_error = str(err)
print(chol_error)
assert chol_error is not None and "not positive definite" in chol_error

D = np.abs(ms.ldlt(L).diagonal())
print(f"ldlt: smallest / largest |D|: {D.min() / D.max():.1e}")
assert D.min() / D.max() < 1e-12
```

The symmetric factorizations differ. Cholesky refuses the matrix: positive
semidefinite is not positive definite, and the rounding-level last pivot
fails its check. `ldlt` does not pivot and has no `pivot_perturb`, so it
factors the matrix, and its `diagonal()` shows the same near-zero entry. Its
diagonal is where to look for the inertia too: the counts of positive and
negative entries are the numbers of positive and negative eigenvalues.

## Threshold pivoting: fill against stability

`splu`, `klu` and `supernodal_lu` take `threshold` in (0, 1]. At each column,
the diagonal entry is kept as the pivot if its magnitude is at least
`threshold` times the largest candidate. Otherwise the largest candidate is
used. With `threshold=1.0`, the default, this is classic partial pivoting.
A smaller threshold keeps more pivots on the diagonal, which keeps the
factorization close to the fill-reducing ordering. The price is a looser
bound on growth: every entry of L is at most `1/threshold` in magnitude.

The test matrix is a 40×40 grid with random unsymmetric weights and a
diagonal weakened by a random factor in [0.05, 1]. Full pivoting leaves the
diagonal at about half the columns:

```{code-cell} python
def weak_diagonal(k, seed=0):
    rng = np.random.default_rng(seed)
    A = laplacian_2d(k) / 4.0
    A.data = A.data * rng.uniform(0.5, 1.5, A.nnz)
    d = A.diagonal()
    return (A - sp.diags(d) + sp.diags(d * rng.uniform(0.05, 1.0, k * k))).tocsr()


W = weak_diagonal(40)
n = W.shape[0]
x_true = np.random.default_rng(3).standard_normal(n)
b = W @ x_true
W64, W32 = ms.from_scipy(W), ms.from_scipy(W.astype(np.float32))

THRESHOLDS = (1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4)
sweep = []
for t in THRESHOLDS:
    F = ms.splu(W64, ordering="amd", threshold=t)
    x = F.solve(b).to_numpy()
    F32 = ms.splu(W32, ordering="amd", threshold=t)
    _, info = iterative_refine(W64, F32, b, max_iter=30, rel_tol=1e-14)
    sweep.append(
        {
            "t": t,
            "nnz": F.nnz,
            "diagonal": np.mean(np.asarray(F.row_perm) == np.asarray(F.col_perm)),
            "max_L": np.abs(ms.to_scipy(F.L)).max(),
            "relres": relres(W, x, b),
            "fwd": fwd_err(x, x_true),
            "f32_iters": info["iters"],
            "f32_converged": info["converged"],
        }
    )

print(" threshold   nnz(L+U)  on diagonal  max|L|  rel. residual  fwd error  f32 refine")
for s in sweep:
    print(
        f"{s['t']:10g} {s['nnz']:10d} {s['diagonal']:12.0%} {s['max_L']:7.1e} "
        f"{s['relres']:14.1e} {s['fwd']:10.1e}  {s['f32_iters']} its"
    )
```

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
ts = [s["t"] for s in sweep]
ax1.semilogx(ts, [s["nnz"] / 1e3 for s in sweep], "o-", color=SERIES[0])
ax1.set_xlabel("threshold")
ax1.set_ylabel("nnz(L+U), thousands")
ax1.set_title("Fill")
ax1.invert_xaxis()
ax2.loglog(ts, [s["max_L"] for s in sweep], "o-", color=SERIES[1], label="max |L|")
ax2.loglog(ts, [1 / t for t in ts], "--", color=INK_2, lw=1, label="bound 1/threshold")
ax2.loglog(ts, [s["relres"] / 1e-16 for s in sweep], "s-", color=SERIES[2], label="rel. residual / 1e-16")
ax2.set_xlabel("threshold")
ax2.set_title("Stability")
ax2.invert_xaxis()
ax2.legend(frameon=False, fontsize=9)
fig.tight_layout()
```

```{code-cell} python
nnz = {s["t"]: s["nnz"] for s in sweep}
assert all(s["max_L"] <= (1 / s["t"]) * (1 + 1e-12) for s in sweep)  # the growth bound
assert nnz[1e-3] < 0.35 * nnz[1.0]  # about 3.5x less fill
assert sweep[0]["diagonal"] < 0.6 and sweep[-1]["diagonal"] > 0.95
assert all(s["relres"] < 1e-12 for s in sweep)
assert sweep[-1]["relres"] > 10 * sweep[0]["relres"]  # stability is what it costs
assert all(s["f32_converged"] for s in sweep)
assert sweep[0]["f32_iters"] < sweep[-1]["f32_iters"]
```

Going from 1 to 0.001 cuts the fill by about 3.5×. Entries of L grow up to the
`1/threshold` bound, and the residual grows by roughly two orders of
magnitude, though it stays near rounding error here. A float32 factor still
refines to full accuracy, but needs five iterations instead of three. Below 0.001 almost every pivot is already on the diagonal, so the
fill stops improving and only the growth bound keeps loosening.

SuperLU's default is 1.0 for general matrices, and UMFPACK uses 0.1 for
unsymmetric strategies. A threshold between 0.1 and 0.01 is a common middle
ground when fill matters and the matrix is not badly scaled. KLU equilibrates
the rows first by default (`scale=True`), which makes the comparison against
the column maximum fairer on a badly scaled matrix.

```{code-cell} python
others = {"klu": ms.klu, "supernodal_lu, amd": lambda A, **kw: ms.supernodal_lu(A, ordering="amd", **kw)}
for name, f in others.items():
    counts = [f(W, threshold=t).nnz for t in (1.0, 0.1, 0.01, 1e-3)]
    print(f"{name:>18}: nnz at threshold 1, 0.1, 0.01, 0.001 = {counts}")
    assert counts[-1] < 0.35 * counts[0]
```

How much the threshold matters depends on the matrix. On `west0479`, under
COLAMD, it changes the fill by two entries, because full partial pivoting
already keeps close to the ordering there:

```{code-cell} python
S = ms.to_scipy(mtl5.io.mm_read("../notebooks/data/west0479.mtx"))
S.eliminate_zeros()
west = {t: ms.splu(S, threshold=t).nnz for t in (1.0, 0.1, 0.01)}
print(west)
assert max(west.values()) < 1.05 * min(west.values())
```

```{note}
Before MTL5 5.24.1 the threshold preferred the wrong row under every ordering
except `natural`, so it barely changed fill
([stillwater-sc/mtl5#916](https://github.com/stillwater-sc/mtl5/issues/916)).
mtl5-python requires 5.24.1 or later.
```

## The accumulator

`splu`, `klu`, `supernodal_lu` and `supernodal_ldlt` take
`accumulator=`, which sets the type of the dense workspace in which each
column's updates are accumulated. A float32 factor built with
`accumulator="f64"` is still stored in float32: each entry of L and U is
rounded once, on the way out. What gets more precise is the arithmetic that
produced it, because the intermediate roundings inside each column's update
chain disappear.

The choices are `None` (the element type), `"f32"`, `"f64"`, `"fma32"` and
`"fma64"`. The effect is visible on a badly scaled matrix, with condition
number about 2.8·10¹⁰. It is the matrix the test suite uses for this:

```{code-cell} python
rng = np.random.default_rng(1)
H = sp.random(400, 400, density=0.02, random_state=2, format="csr") + sp.eye(400) * 1e-3
H = (sp.diags(10.0 ** rng.uniform(-4, 4, 400)) @ H).tocsr()
xh_true = rng.standard_normal(400)
bh = H @ xh_true
H64, H32 = ms.from_scipy(H), ms.from_scipy(H.astype(np.float32))

print("ordering  accumulator  direct fwd error  refine iters  refined fwd error")
acc_rows = {}
for o in ("colamd", "amd", "natural"):
    for acc in (None, "f64"):
        F = ms.splu(H32, ordering=o, accumulator=acc)
        direct = fwd_err(F.solve(bh).to_numpy(), xh_true)
        x, info = iterative_refine(H64, F, bh, max_iter=30, rel_tol=1e-15)
        refined = fwd_err(np.asarray(x), xh_true)
        acc_rows[o, F.accumulator] = (direct, info["iters"], refined)
        print(f"{o:>8}  {F.accumulator:>11}  {direct:16.2e}  {info['iters']:12d}  {refined:17.2e}")

for o in ("colamd", "amd", "natural"):
    assert acc_rows[o, "f64"][0] < acc_rows[o, "default"][0]
    assert acc_rows[o, "f64"][2] < acc_rows[o, "default"][2]
    assert acc_rows[o, "f64"][1] <= acc_rows[o, "default"][1]
```

The direct error improves by a modest 1.3–2.5×. The larger effect is on
refinement: a float64 accumulator takes one or two fewer iterations and ends
6 to 300 times closer, from a factor that occupies the same memory.
The direction holds on every platform the test suite runs on, but the size of
the gain varies between platforms. `"fma32"` is not consistently better than
the default, so it is not shown.

On a well-scaled matrix there is little to gain. When each column's update
chain is short and free of cancellation, rounding the stored factor
dominates the error, and the accumulator cannot change that. Here it is
even slightly worse:

```{code-cell} python
Pn = laplacian_2d(32) / 4.0 + 1e-4 * sp.eye(32 * 32)
xp_true = np.random.default_rng(1).standard_normal(Pn.shape[0])
bp = Pn @ xp_true
P32 = ms.from_scipy(Pn.astype(np.float32))
e_default = fwd_err(ms.splu(P32).solve(bp).to_numpy(), xp_true)
e_f64 = fwd_err(ms.splu(P32, accumulator="f64").solve(bp).to_numpy(), xp_true)
print(f"default {e_default:.2e}, f64 {e_f64:.2e}")
assert 0.5 < e_f64 / e_default < 2
```

Three combinations are refused rather than silently accepted:

```{code-cell} python
refusals = {}
try:
    ms.splu(H64, accumulator="f32")
except ValueError as err:
    refusals["narrower than float64"] = str(err)
try:
    ms.splu(H32, accumulator="quire")
except ValueError as err:
    refusals["quire"] = str(err)
try:
    ms.splu(H32, accumulator="f64").refactor(H32)
except ValueError as err:
    refusals["refactor"] = str(err)
for k, v in refusals.items():
    print(f"{k}: {v}\n")
assert len(refusals) == 3

F = ms.supernodal_ldlt(ms.from_scipy(Pn.astype(np.float32)), accumulator="f64")
F.refactor(ms.from_scipy(Pn.astype(np.float32)))
assert F.accumulator == "f64"
```

- An accumulator narrower than the element type would lose precision rather
  than gain it.
- The quire is refused, because the workspace holds one accumulator per row
  and a 4226-bit quire there needs its own design.
- `refactor()` on `splu`, `klu` and `supernodal_lu` replays the stored pivot
  sequence through an entry point that has no accumulator. Accepting it would
  quietly drop back to element precision, so it raises, and you construct a
  new factorization instead. `supernodal_ldlt` re-runs the full numeric
  factorization on `refactor()`, so it keeps the policy.

## A checklist

After a factorization, and before trusting its solve:

1. **`num_perturbed`** (LU family). If it is nonzero, the factor is of a
   perturbed matrix: check the refinement's `converged` and `rel_residual`.
2. **The pivot ratio**, from `U.diagonal()` or LDLᵀ's `diagonal()`. A ratio
   near the unit roundoff means the matrix is singular to working precision,
   whether or not anything raised. For a quick detector, pass a small
   `pivot_perturb` and read `num_perturbed`.
3. **The residual** of the solve you care about, in float64. It is cheap,
   and it is what refinement minimizes. It bounds the forward error only
   through κ (error ≲ κ × residual), so weigh it against the conditioning and
   the accuracy you need.
4. **`nnz`** against `A.nnz`, and **`nblocks`** or **`nsuper`** for KLU and
   supernodal LU. These tell you whether the ordering, the threshold and the
   block structure did what you expected. The
   [orderings guide](../notebooks/sparse_direct_orderings) shows what good
   and bad fill look like.
5. **`dtype` and `accumulator`**, which the repr does not show, when the
   factor came from code you did not write.

API reference: {py:func}`mtl5.sparse.splu`, {py:func}`mtl5.sparse.klu`,
{py:func}`mtl5.sparse.supernodal_lu`, {py:func}`mtl5.sparse.cholesky`,
{py:func}`mtl5.sparse.ldlt`, {py:func}`mtl5.sparse.supernodal_ldlt`,
{py:func}`mtl5.sparse.qr`, {py:func}`mtl5.mixed.iterative_refine`.
