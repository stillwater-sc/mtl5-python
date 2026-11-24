---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sparse direct solvers

A direct solver factors the matrix into triangular factors once, then solves
by forward and back substitution. This page covers three things:

- when that beats an iterative method, and when it does not;
- where the algorithms came from;
- how to choose among the factorizations in `mtl5.sparse`.

The page is executed when the documentation is built. Every number on it is
measured, and the `assert` statements check the claims the prose makes.

```{code-cell} python
import time

import numpy as np
import scipy.sparse as sp

import mtl5
import mtl5.sparse as ms


def best_ms(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out


def relres(A, x, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)
```

## Why a direct solver?

### Robustness

An iterative method converges only as well as the spectrum and the
preconditioner allow. A direct factorization with pivoting succeeds on any
nonsingular matrix, up to rounding. `west0479`, the chemical-process matrix
from the [orderings guide](../notebooks/sparse_direct_orderings), is a good
example. It is unsymmetric, badly conditioned, and has zeros on its diagonal:

```{code-cell} python
S = ms.to_scipy(mtl5.io.mm_read("../notebooks/data/west0479.mtx"))
S.eliminate_zeros()
W = ms.from_scipy(S)
b = S @ np.ones(S.shape[0])

for name in ("gmres", "bicgstab"):
    x, info = getattr(ms, name)(W, b, rtol=1e-10, maxiter=2000)
    print(f"{name:>9}: info = {info}, relative residual {relres(S, np.asarray(x), b):.1e}")

try:
    ms.ilu0(W)
except RuntimeError as err:
    print(f"     ilu0: {err}")

x = ms.splu(W).solve(b).to_numpy()
print(f"     splu: relative residual {relres(S, x, b):.1e}")
assert relres(S, x, b) < 1e-14
```

GMRES stalls and BiCGSTAB diverges. The standard cheap preconditioner,
ILU(0), cannot even be built, because it needs a nonzero diagonal. The sparse
LU solves the system to rounding error.

### Many right-hand sides

Once the factors exist, each further solve costs two triangular substitutions,
and their cost is proportional to the nonzeros in the factors. An iterative
method starts over for every right-hand side. Here are twenty right-hand sides
on a 2-D Poisson problem:

```{code-cell} python
P = mtl5.generators.poisson2d(150, 150)
PS = ms.to_scipy(P)
n = P.shape[0]
B = np.random.default_rng(0).standard_normal((n, 20))

t = time.perf_counter()
chol = ms.cholesky(P)
X = [chol.solve(B[:, k]).to_numpy() for k in range(20)]
t_direct = (time.perf_counter() - t) * 1e3

t = time.perf_counter()
Y = [ms.cg(P, B[:, k], rtol=1e-10, maxiter=5000) for k in range(20)]
t_cg = (time.perf_counter() - t) * 1e3

print(f"n = {n:,}: Cholesky factor + 20 solves {t_direct:6.0f} ms")
print(f"           20 CG solves                {t_cg:6.0f} ms")
assert all(info == 0 for _, info in Y)
assert max(relres(PS, x, B[:, k]) for k, x in enumerate(X)) < 1e-12
```

The factorization also pays off across systems that share a pattern. When
only the values change, `refactor()` reuses the ordering and the analysis;
the [orderings guide](../notebooks/sparse_direct_orderings) measures that.

### When not to: fill in three dimensions

The price of a direct solver is fill-in, and fill depends on the geometry of
the problem. Compare two Poisson problems with the same number of unknowns,
one on a 2-D grid and one on a 3-D grid:

```{code-cell} python
def poisson(k, dim):
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k), format="csr")
    eye = sp.identity(k, format="csr")
    terms = []
    for axis in range(dim):
        factors = [T if d == axis else eye for d in range(dim)]
        term = factors[0]
        for f in factors[1:]:
            term = sp.kron(term, f, format="csr")
        terms.append(term)
    return sum(terms).tocsr()


rows = []
for label, A in (("2-D 125 x 125", poisson(125, 2)), ("3-D 25 x 25 x 25", poisson(25, 3))):
    M = ms.from_scipy(A)
    rhs = np.ones(A.shape[0])
    t_chol, c = best_ms(lambda M=M: ms.cholesky(M), repeats=1)
    t_cg, (x, info) = best_ms(lambda M=M, rhs=rhs: ms.cg(M, rhs, rtol=1e-10, maxiter=10000), repeats=1)
    rows.append((label, A.shape[0], c.nnz / A.nnz, t_chol, t_cg))

print(f"{'grid':>16} {'n':>7} {'fill':>6} {'Cholesky':>9} {'one CG':>8}")
for label, size, fill, t_chol, t_cg in rows:
    print(f"{label:>16} {size:7,} {fill:5.1f}x {t_chol:7.0f}ms {t_cg:6.0f}ms")
assert rows[0][1] == rows[1][1] and rows[1][2] > 3 * rows[0][2]
```

With the same number of unknowns, the 3-D factor carries several times the
fill of the 2-D one, and it takes much longer to compute. The reason is
geometric. A good ordering of a 2-D grid of $n$ points produces factors with
$O(n \log n)$ nonzeros. On a 3-D grid it produces $O(n^{4/3})$ nonzeros and
needs $O(n^2)$ work, because the separators that split the grid are planes,
not lines. For large 3-D problems, and for operators you can apply but not
store, a preconditioned iterative method is usually the better tool.

The rule of thumb:

- **Prefer a direct solver** when robustness matters (ill-conditioned,
  indefinite or unsymmetric systems with no good preconditioner), when there
  are many right-hand sides, when a fixed pattern is refactored repeatedly,
  and when fill stays moderate. That covers 2-D meshes, circuits, chemical
  process models, and most problems under a few hundred thousand unknowns.
- **Prefer an iterative solver** for large 3-D problems, when memory is the
  constraint, and when the matrix is only available as a product.

## A short history

Sparse elimination is almost as old as electronic computing. Each idea below
survives in `mtl5.sparse`:

- **1957: fill-in as a choice.** Markowitz observed that the order of
  elimination decides how many new nonzeros appear. He chose each pivot to
  minimize the product of its row and column counts. Most sparse LU pivoting
  heuristics descend from that rule.
- **1967: minimum degree.** Tinney and Walker, solving power-network
  equations, eliminated at each step the node with the fewest connections.
  Through the 1970s and 80s, George and Liu made minimum degree fast and
  precise, and it became the standard symmetric ordering.
- **1969–1971: bandwidth.** Cuthill and McKee ordered nodes level by level to
  keep the nonzeros near the diagonal. George observed that reversing the
  order (reverse Cuthill–McKee) gives a smaller envelope.
- **1973: nested dissection.** George showed that recursively splitting a
  grid with small separators gives asymptotically optimal fill on regular 2-D
  meshes. Today's large-scale orderings (METIS, SCOTCH) build on the idea.
- **1987–1988: pivoting without waste.** George and Ng proved that the fill
  of LU with partial pivoting is bounded by the Cholesky factor of $A^TA$.
  That is why a column ordering for LU is computed on $A^TA$. Gilbert and
  Peierls gave a left-looking LU with partial pivoting whose time is
  proportional to the arithmetic performed. That is the algorithm behind
  `splu`.
- **1987–1999: supernodes.** Ashcraft, Grimes, Lewis, Peyton and Simon
  grouped columns with identical structure into supernodes, so most of the
  work becomes dense matrix products. SuperLU (Demmel, Eisenstat, Gilbert,
  Li and Liu) carried supernodes into unsymmetric LU. Hence
  `supernodal_lu` and `supernodal_ldlt`.
- **1996: approximate minimum degree.** Amestoy, Davis and Duff replaced
  exact degrees with cheap, tight bounds. The result, AMD, is much faster and
  just as good, and it is the `amd` ordering here.
- **2004: COLAMD.** Davis, Gilbert, Larimore and Ng applied approximate
  minimum degree to the columns of $A$, which orders $A^TA$ without ever
  forming it. It is the `colamd` ordering, and MATLAB's.
- **2010: KLU.** Davis and Palamadai Natarajan combined a block triangular
  form with per-block LU. It was designed for circuit simulation, whose
  matrices are very sparse and decompose into many small blocks. It is
  `klu` here.

Further reading: T. A. Davis, *Direct Methods for Sparse Linear Systems*,
SIAM, 2006; I. S. Duff, A. M. Erisman and J. K. Reid, *Direct Methods for
Sparse Matrices*, 2nd ed., Oxford University Press, 2017; T. A. Davis,
S. Rajamanickam and W. M. Sid-Lakhdar, "A survey of direct methods for sparse
linear systems", *Acta Numerica* 25 (2016), 383–566.

## Choosing a factorization

| Factorization | Matrix | Pivoting | Default ordering | Strengths | Limits |
|---|---|---|---|---|---|
| `cholesky` | symmetric positive definite | none needed | `amd` | stores only `L`; half the work of LU; no pivoting to go wrong | raises `RuntimeError` if `A` is not positive definite |
| `supernodal_ldlt` | symmetric, also indefinite | none | `amd` | dense block updates, the fastest symmetric option below; `.diagonal()` gives the inertia | raises on a zero pivot |
| `ldlt` | symmetric, also indefinite | none | `amd` | simple column-by-column factor; `.diagonal()` gives the inertia | raises on a zero pivot |
| `splu` | square, unsymmetric | threshold partial | `colamd` | the general-purpose choice; `threshold=`, `pivot_perturb=`, `accumulator=`; exposes `L`, `U` and both permutations | `refactor()` reuses the old pivot sequence |
| `supernodal_lu` | square, unsymmetric | threshold partial | `colamd` | dense block updates; `scale=True` row-equilibrates, which helps most in low precision | supernodes pay off on denser factors |
| `klu` | square, unsymmetric, reducible | per block | AMD per block | block triangular form: factors only the diagonal blocks; strong on circuits and flowsheets | no single `L`, `U` to inspect |
| `qr` | rectangular $m \ge n$ | none (orthogonal) | `colamd` | least squares without forming $A^TA$; exposes `R` | `Q` stays implicit |

Every factorization takes `ordering=` (see the
[orderings guide](../notebooks/sparse_direct_orderings)), solves with
`.solve(b)`, and has `.refactor(A2)` for a matrix with the same pattern. One
exception applies to `refactor()`. `splu`, `supernodal_lu` and `klu` replay
their stored pivot sequence through a path that has no accumulator. So if the
factorization was built with a non-default `accumulator=`, `refactor()`
raises `ValueError` rather than silently drop back to element precision.
Build a new factorization instead. `supernodal_ldlt` reruns its full numeric
factorization, so it keeps the accumulator.

A decision path:

1. **Overdetermined ($m \ge n$), or least squares?** Use `qr`. Underdetermined
   systems ($m < n$) are not supported by these factorizations.
2. **Symmetric positive definite?** Use `cholesky`, or `supernodal_ldlt`
   when the factors are large.
3. **Symmetric indefinite?** Use `supernodal_ldlt` or `ldlt`. Neither pivots;
   if one raises on a zero pivot, fall back to `splu`.
4. **Unsymmetric?** Use `splu`. If the matrix comes from a circuit, a
   flowsheet or another network, also try `klu` and compare `.nnz`.

### The table, measured

The symmetric factorizations on the 2-D Poisson matrix from above:

```{code-cell} python
print(f"{'factorization':>16} {'factor':>9} {'nnz':>10} {'max residual':>13}")
rhs = np.ones(n)
for name in ("cholesky", "supernodal_ldlt", "ldlt", "splu", "supernodal_lu"):
    t_f, f = best_ms(lambda name=name: getattr(ms, name)(P, ordering="amd"))
    r = relres(PS, f.solve(rhs).to_numpy(), rhs)
    print(f"{name:>16} {t_f:7.1f}ms {f.nnz:10,} {r:13.1e}")
    assert r < 1e-12
```

`cholesky` and the LDLᵀ variants store one triangle, so they hold about half
of LU's nonzeros. (The LDLᵀ counts exclude the unit diagonal, which is not
stored.) Supernodes help most when the factors are dense enough to form large
blocks. Measure on your own matrices before choosing a supernodal variant for
speed alone.

On `west0479`, block triangular form finds structure that a single ordering
does not:

```{code-cell} python
k = ms.klu(W)
lu = ms.splu(W)
print(f"klu:  {k.nblocks} diagonal blocks, nnz = {k.nnz:,}")
print(f"splu: one block,              nnz = {lu.nnz:,}  (colamd)")
assert k.nblocks > 1 and k.nnz < lu.nnz
```

The symmetric factorizations differ in what they accept. Shift the Poisson
matrix into the middle of its spectrum, and it becomes indefinite:

```{code-cell} python
small = ms.to_scipy(mtl5.generators.poisson2d(10, 10))
eigs = np.linalg.eigvalsh(small.toarray())
shift = 500.3  # inside the spectrum, and not equal to any diagonal entry
A_ind = (small - shift * sp.identity(100)).tocsr()

try:
    ms.cholesky(A_ind)
except RuntimeError as err:
    print("cholesky:", err)

for name in ("ldlt", "supernodal_ldlt"):
    f = getattr(ms, name)(A_ind)
    d = f.diagonal()
    print(f"{name}: {int((d < 0).sum())} negative and {int((d > 0).sum())} positive pivots")

print(f"eigenvalues below the shift: {int((eigs < shift).sum())}")
assert int((ms.ldlt(A_ind).diagonal() < 0).sum()) == int((eigs < shift).sum())
```

Cholesky refuses the matrix. Both LDLᵀ factorizations accept it, and the
signs of `D` count the eigenvalues on each side of zero. By Sylvester's law
of inertia, $LDL^T$ preserves those counts. The negative pivots of
$A - \sigma I$ therefore count the eigenvalues of $A$ below $\sigma$. To count
the eigenvalues in an interval $[a, b)$, factor at both ends and subtract:

```{code-cell} python
def count_below(A, sigma):
    shifted = (A - sigma * sp.identity(A.shape[0])).tocsr()
    return int((ms.ldlt(shifted).diagonal() < 0).sum())


a, b = 200.3, 400.3  # endpoints chosen off the diagonal value, so no pivot is zero
inside = count_below(small, b) - count_below(small, a)
print(f"eigenvalues in [{a}, {b}): {inside} by two LDLT factorizations, "
      f"{int(((eigs >= a) & (eigs < b)).sum())} by eigvalsh")
assert inside == int(((eigs >= a) & (eigs < b)).sum())
```

What LDLᵀ without pivoting cannot do is pass a zero pivot. LU's row
interchanges handle that case:

```{code-cell} python
swap = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
try:
    ms.ldlt(swap, ordering="natural")
except RuntimeError as err:
    print("ldlt:", err)
print("splu:", ms.splu(swap, ordering="natural").solve(np.array([1.0, 2.0])).to_numpy())
```
