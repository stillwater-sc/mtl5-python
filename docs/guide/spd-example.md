---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A symmetric positive definite example

The [orderings guide](../notebooks/sparse_direct_orderings) works through an
unsymmetric matrix with LU. Many of the matrices people actually factor are
symmetric positive definite (SPD), provided they are nonsingular:

- stiffness matrices, once enough boundary conditions remove the rigid-body
  modes;
- graph Laplacians, once a node is grounded (a plain Laplacian is singular,
  since the constant vector is in its nullspace);
- normal equations $A^TA$, when $A$ has full column rank;
- covariance matrices, when the centred variables are linearly independent
  (so no variable is an affine function of the others, not even $X$ and
  $X + 1$).

For SPD matrices, `mtl5.sparse` has two factorizations that exploit the
symmetry and need no pivoting. `cholesky` raises `RuntimeError` on a matrix
that turns out not to be positive definite. `supernodal_ldlt` also accepts a
symmetric indefinite matrix, and raises only on a zero pivot.

- `ms.cholesky(A)`, $A = LL^T$, an up-looking (column-at-a-time,
  non-supernodal) factorization;
- `ms.supernodal_ldlt(A)`, $A = LDL^T$, which groups columns with the same
  pattern into *supernodes* and updates them with dense block operations.

This page factors the 2-D Poisson matrix with both, for every ordering, and
measures how the fill and the factor time grow with the problem size. The page
is executed when the documentation is built, and the `assert` statements check
what the prose claims.

```{code-cell} python
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import mtl5
import mtl5.sparse as ms

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


SPY_DPI = 200  # about two pixels per matrix row in the spy figures


def spy(ax, M, title, color=SERIES[0]):
    # One image cell per matrix entry, so two nonzeros never overlap.
    S = ms.to_scipy(M).tocoo()
    pattern = np.zeros(S.shape, dtype=bool)
    pattern[S.row, S.col] = True
    ax.imshow(pattern, cmap=ListedColormap([SURFACE, color]), interpolation="none")
    ax.set_title(f"{title}\nnz = {S.nnz:,}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def best_ms(fn, repeats=3):
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out
```

## The matrix

`mtl5.generators.poisson2d(k, k)` is the five-point Laplacian on a $k \times k$
grid with Dirichlet boundaries: $n = k^2$ unknowns, numbered row by row, with
$4/h^2$ on the diagonal and $-1/h^2$ for each grid neighbour, where
$h = 1/(k+1)$ is the grid spacing. It is SPD, and it is the standard model
problem for sparse Cholesky.

```{code-cell} python
k = 24
A = mtl5.generators.poisson2d(k, k)
n = A.shape[0]
S = ms.to_scipy(A)
print(f"n = {n}, nnz(A) = {A.nnz}")

assert (S != S.T).nnz == 0                      # symmetric
assert np.linalg.eigvalsh(S.toarray()).min() > 0  # positive definite

# Both factorizations solve it to rounding.
b = S @ np.ones(n)
for f in (ms.cholesky(A), ms.supernodal_ldlt(A)):
    x = f.solve(b).to_numpy()
    print(f"{type(f).__name__:>20}: max |x - 1| = {np.abs(x - 1).max():.1e}")
    assert np.abs(x - 1).max() < 1e-12
```

## What the ordering does to the factor

A symmetric factorization must permute rows and columns together, $PAP^T$, to
keep the symmetry, so it takes one permutation. `amd` is the default for both.
It orders the graph of $A$ directly. `colamd` is offered too, but it orders
$A^TA$, which for a symmetric $A$ is the pattern of $A^2$, a denser graph than
the one being factored. `rcm` is a heuristic that reduces the bandwidth,
without guaranteeing the minimum, and `natural` keeps the grid's row-by-row
numbering.

```{code-cell} python
ORDERINGS = ["natural", "rcm", "colamd", "amd"]
chol = {o: ms.cholesky(A, ordering=o) for o in ORDERINGS}

fig, axes = plt.subplots(2, 3, figsize=(13, 9.4), dpi=SPY_DPI)
spy(axes[0, 0], A, "A", color=INK_2)
for ax, o, c in zip(axes.flat[1:], ORDERINGS, SERIES):
    spy(ax, chol[o].L, f"L  ({o})", color=c)
axes[1, 2].axis("off")
plt.tight_layout()
plt.show()

# The factors reconstruct the permuted matrix, to rounding.
for o, f in chol.items():
    L = ms.to_scipy(f.L)
    err = abs(S[f.perm][:, f.perm] - L @ L.T).max() / abs(S).max()
    assert err < 1e-14, (o, err)
```

The natural ordering leaves $A$ banded, with half-bandwidth $k$. A Cholesky
factor never fills outside the *envelope* of $A$, the region between each
row's first nonzero and the diagonal, and for this matrix it fills that
envelope completely. So its fill can be counted without factoring: the first
grid row couples each unknown only to its left neighbour, giving $2k - 1$
entries, and each of the remaining $n - k$ rows of $L$ carries $k + 1$.
`rcm` keeps a band of about the same width, but numbers the grid along its
diagonals, which shortens most rows of the envelope: it saves about a quarter
of the fill. `amd` does not try to keep a band at all. At each step it
eliminates an unknown of smallest *approximate degree*: few neighbours in the
graph that remains, estimated cheaply instead of counted exactly. Low degree is
a proxy for low fill, not a guarantee of it, but on this grid it leaves a
factor with well under half the entries.

```{code-cell} python
envelope = (2 * k - 1) + (n - k) * (k + 1)
print(f"natural: nnz(L) = {chol['natural'].nnz:,}, envelope of A = {envelope:,}")
assert chol["natural"].nnz == envelope

for o in ORDERINGS:
    print(f"{o:>8}: nnz(L) = {chol[o].nnz:7,}  ({chol[o].nnz / A.nnz:4.1f}x nnz(A))")
assert chol["amd"].nnz < chol["colamd"].nnz < chol["rcm"].nnz <= chol["natural"].nnz
```

## How fill and time grow with the problem size

Now repeat that for grids from $16 \times 16$ to $128 \times 128$, with both
factorizations.

`nnz` counts what each object stores. `cholesky` stores $L$ with its diagonal.
`supernodal_ldlt` stores $L$ without its unit diagonal and keeps $D$ apart, so
its count is smaller by exactly $n$: the two factors have the same pattern.

```{code-cell} python
SIZES = [16, 32, 64, 128]
results = {}  # (ordering, k) -> (n, nnz(A), nnz(L), cholesky ms, supernodal ms)
for k_ in SIZES:
    Ak = mtl5.generators.poisson2d(k_, k_)
    nk = Ak.shape[0]
    for o in ORDERINGS:
        t_c, c = best_ms(lambda: ms.cholesky(Ak, ordering=o))
        t_s, s = best_ms(lambda: ms.supernodal_ldlt(Ak, ordering=o))
        assert s.nnz == c.nnz - nk  # same pattern; LDL^T leaves out the unit diagonal
        results[o, k_] = (nk, Ak.nnz, c.nnz, t_c, t_s)

print(f"{'grid':>9} {'n':>7} {'ordering':>8} {'nnz(L)':>11} {'fill':>6} {'cholesky':>9} {'supernodal':>11}")
for k_ in SIZES:
    for o in ORDERINGS:
        nk, nnz_a, nnz_l, t_c, t_s = results[o, k_]
        print(f"{k_:>4}x{k_:<4} {nk:7,} {o:>8} {nnz_l:11,} {nnz_l / nnz_a:5.1f}x {t_c:7.1f}ms {t_s:9.1f}ms")
```

On a log-log plot the slope is the growth exponent. The envelope argument gives
the natural ordering's exactly: $nnz(L) \approx n \cdot k = n^{3/2}$, and `rcm`,
a band of the same width, grows at nearly the same rate. The best possible
growth on a 2-D grid is $O(n \log n)$, which nested dissection achieves
(George, 1973), and which shows up on this range of sizes as a slope of about
1.1. Minimum degree is a greedy heuristic with no such guarantee: here it lands
in between, well below $3/2$ but above the nested dissection rate. mtl5 has no
nested dissection ordering yet.

```{code-cell} python
def slope(o):
    # Least-squares slope of log nnz(L) against log n over the measured sizes.
    ns = np.array([results[o, k_][0] for k_ in SIZES], dtype=float)
    nz = np.array([results[o, k_][2] for k_ in SIZES], dtype=float)
    return np.polyfit(np.log(ns), np.log(nz), 1)[0]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
for o, c in zip(ORDERINGS, SERIES):
    ns = [results[o, k_][0] for k_ in SIZES]
    ax1.loglog(ns, [results[o, k_][2] for k_ in SIZES], "o-", color=c, label=f"{o} (slope {slope(o):.2f})")
    ax2.loglog(ns, [results[o, k_][3] for k_ in SIZES], "o-", color=c, label=f"{o}, cholesky")
    ax2.loglog(ns, [results[o, k_][4] for k_ in SIZES], "s--", color=c, alpha=0.7, label=f"{o}, supernodal")
ax1.set(xlabel="n", ylabel="nnz(L)", title="fill")
ax2.set(xlabel="n", ylabel="factor time (ms)", title="time (best of 3)")
ax1.legend(fontsize=8, frameon=False)
ax2.legend(fontsize=7, frameon=False, ncol=2)
for ax in (ax1, ax2):
    ax.grid(True, which="major", color="#ecebe7", linewidth=0.6)
plt.tight_layout()
plt.show()

for o in ORDERINGS:
    print(f"{o:>8}: nnz(L) grows like n^{slope(o):.2f}")
assert abs(slope("natural") - 1.5) < 0.05
assert abs(slope("rcm") - 1.5) < 0.1
assert slope("amd") < 1.35

# The O(n log n) rate, fitted over the same sizes, for comparison.
ns = np.array([k_ * k_ for k_ in SIZES], dtype=float)
nd_slope = np.polyfit(np.log(ns), np.log(ns * np.log(ns)), 1)[0]
print(f"   n log n: grows like n^{nd_slope:.2f}")
assert nd_slope < slope("amd")
```

Three things to take from the measurements:

- **The ordering decides the exponent, not just the constant.** At $128 \times
  128$ the `amd` factor is several times smaller than the `natural` one, and
  the gap widens with every doubling of the grid. Work grows faster still:
  $O(n^2)$ for a band of width $\sqrt n$ against $O(n^{3/2})$ for nested
  dissection orderings.
- **`colamd` is the wrong tool for a symmetric matrix.** It is the right
  default for `splu`, but it orders $A^TA$, so on this problem its factor is
  roughly half as large again as `amd`'s.
- **Supernodes pay off once the factor is large enough.** With `amd`, the
  columns near the end of the elimination form dense blocks, and
  `supernodal_ldlt` updates them with dense kernels instead of one column at a
  time. On small grids the two are about even, and the supernodal variant
  pulls ahead as the grid grows. The times above are measured on whatever
  machine built this page, so the exact ratio varies; the fill counts do not.

Symmetric indefinite matrices are outside `cholesky`'s reach, but
`supernodal_ldlt` and `ldlt` factor them as long as no pivot is zero. For those,
and for unsymmetric matrices, see
[choosing a factorization](direct-solvers.md#choosing-a-factorization).
