---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# How fill-reducing orderings work

The [orderings guide](../notebooks/sparse_direct_orderings) shows *that*
the ordering decides how much a factorization fills in: on `west0479` the
same matrix gives three times more nonzeros in L and U under one ordering
than under another. This page explains *why*, and how each ordering
algorithm turns "reduce the fill" into a rule it can actually follow.

In brief:

- every ordering reads the same information, the sparsity pattern, and never
  the values;
- the pattern becomes a graph, and Gaussian elimination becomes a game played
  on that graph, in which every new edge is a fill-in entry;
- finding the best order is NP-hard, so each algorithm is a heuristic, and
  they differ in what property of the graph they use as the signal.

The page is executed when the documentation is built, and the `assert`
statements check the claims the prose makes.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import ListedColormap

import mtl5
import mtl5.sparse as ms

SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
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
    """The k x k five-point Laplacian: one vertex per grid point."""
    eye = sp.eye(k, format="csr")
    tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k), format="csr")
    A = (sp.kron(eye, tri) + sp.kron(tri, eye)).tocsr()
    A.eliminate_zeros()  # for small k, scipy's kron stores whole blocks, zeros included
    return A


def permuted(A, p):
    return A[p][:, p].tocsr()


def cholesky_nnz(A, p):
    """nnz(L) of the Cholesky factor of A[p][:, p], taking the order as given."""
    return ms.cholesky(ms.from_scipy(permuted(A, p)), ordering="natural").nnz


def spy(ax, M, title, color=SERIES[0]):
    """One image cell per stored entry, so neighbouring nonzeros never merge."""
    S = sp.coo_matrix(M)
    pattern = np.zeros(S.shape, dtype=bool)
    pattern[S.row, S.col] = True
    ax.imshow(pattern, cmap=ListedColormap([SURFACE, color]), interpolation="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
```

## A sparse matrix is a graph

Give the symmetric matrix A one vertex per row and column, and an edge i–j
whenever $a_{ij} \neq 0$. For the five-point Laplacian on a 4×4 grid, the
graph is the grid itself:

```{code-cell} python
K = 4
A4 = laplacian_2d(K)
XY = np.array([(i % K, -(i // K)) for i in range(K * K)], dtype=float)


def edges(A):
    S = sp.triu(A, k=1).tocoo()
    return list(zip(S.row.tolist(), S.col.tolist()))


def draw_graph(ax, xy, base, fill=(), eliminated=(), title=""):
    for i, j in base:
        ax.plot(*xy[[i, j]].T, color=GRID, lw=1.5, zorder=1)
    for i, j in fill:
        ax.plot(*xy[[i, j]].T, color=SERIES[1], lw=1.8, zorder=2)
    eliminated = [int(v) for v in eliminated]
    alive = [v for v in range(len(xy)) if v not in set(eliminated)]
    ax.scatter(*xy[alive].T, s=160, color=SERIES[0], zorder=3)
    if eliminated:
        ax.scatter(*xy[list(eliminated)].T, s=160, facecolor=SURFACE, edgecolor=GRID, zorder=3)
    for v, (x, y) in enumerate(xy):
        ax.text(x, y, str(v), ha="center", va="center", fontsize=7,
                color=SURFACE if v in alive else INK_2, zorder=4)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


assert A4.nnz == 5 * K * K - 4 * K  # five-point stencil, minus the boundary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
spy(ax1, A4, "pattern of A (16 x 16)")
draw_graph(ax2, XY, edges(A4), title="its graph")
fig.tight_layout()
```

## Elimination is a game on the graph

Eliminating variable $v$ subtracts multiples of row $v$ from every row
that has a nonzero in column $v$. In the graph, those rows are the
neighbours of $v$. Each of them acquires the whole of row $v$'s pattern,
so after the step every pair of $v$'s neighbours is connected.

> **Eliminating a vertex turns its remaining neighbours into a clique.
> Every edge the clique adds is a fill-in entry of the factor.**

This is the *elimination game* (Parter 1961, Rose 1970). It needs nothing
but the pattern, so the fill of an ordering can be counted before any
arithmetic is done. A direct implementation fits in a few lines:

```{code-cell} python
def adjacency(A):
    S = (abs(A) + abs(A).T).tocsr()
    return [set(S.indices[S.indptr[i] : S.indptr[i + 1]].tolist()) - {i} for i in range(S.shape[0])]


def elimination_game(A, order, record=None):
    """Eliminate vertices in `order`. Returns (fill edges, strictly-lower nnz of L)."""
    g = adjacency(A)
    fill, lower = [], 0
    for v in order:
        nbrs = sorted(g[v])
        lower += len(nbrs)  # column v of L: one entry per remaining neighbour
        for i, a in enumerate(nbrs):
            for b in nbrs[i + 1 :]:
                if b not in g[a]:
                    g[a].add(b)
                    g[b].add(a)
                    fill.append((a, b))
        for a in nbrs:
            g[a].discard(v)
        g[v] = set()
        if record is not None:
            record.append(list(fill))
    return fill, lower
```

Eliminating the 4×4 grid row by row, in the natural order, looks like this.
The orange edges are fill:

```{code-cell} python
steps = []
order = np.arange(K * K)
elimination_game(A4, order, record=steps)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, s in zip(axes, (1, 2, 5, 8)):
    draw_graph(ax, XY, edges(A4), fill=steps[s - 1], eliminated=order[:s],
               title=f"after {s} eliminations: {len(steps[s - 1])} fill")
fig.tight_layout()
```

Eliminating vertex 0 joins its two neighbours, 1 and 4. Eliminating 1 then
joins 2, 4 and 5, and so on. The fill builds a band that sweeps down the
grid, one row wide.

The game is exact, not a model. The strictly lower count it returns, plus
the n diagonal entries, is precisely the `nnz` of mtl5's Cholesky factor,
for any ordering:

```{code-cell} python
A12 = laplacian_2d(12)
n12 = A12.shape[0]
for name in ("natural", "rcm", "amd"):
    p = np.asarray(ms.ordering(ms.from_scipy(A12), name)) if name != "natural" else np.arange(n12)
    fill, lower = elimination_game(A12, p)
    nnz = cholesky_nnz(A12, p)
    print(f"{name:>8}: {len(fill):4d} fill edges, game predicts nnz(L) = {lower + n12}, cholesky: {nnz}")
    assert lower + n12 == nnz
```

## The order is everything: the arrow matrix

A star graph, one hub joined to n − 1 leaves, makes the point in its
starkest form. As a matrix it is an arrow: a full first row and column,
and a diagonal.

- **Hub first:** the hub's n − 1 neighbours become a clique, and the factor
  is completely dense.
- **Hub last:** each leaf has a single neighbour, the hub. A one-vertex
  clique adds nothing, so there is no fill at all.

```{code-cell} python
n = 40
arrow = sp.lil_matrix((n, n))
arrow.setdiag(float(n))
arrow[0, 1:] = 1.0
arrow[1:, 0] = 1.0
arrow = arrow.tocsr()

hub_first = np.arange(n)
hub_last = np.r_[1:n, 0]
L_first = ms.to_scipy(ms.cholesky(ms.from_scipy(permuted(arrow, hub_first)), ordering="natural").L)
L_last = ms.to_scipy(ms.cholesky(ms.from_scipy(permuted(arrow, hub_last)), ordering="natural").L)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
spy(axes[0], arrow, f"A: nnz = {arrow.nnz}")
spy(axes[1], L_first, f"L, hub first: nnz = {L_first.nnz}", SERIES[1])
spy(axes[2], L_last, f"L, hub last: nnz = {L_last.nnz}", SERIES[2])
fig.tight_layout()

assert L_first.nnz == n * (n + 1) // 2  # dense lower triangle
assert L_last.nnz == 2 * n - 1  # exactly the lower half of A: no fill
```

Same matrix, same nonzeros: from no fill at all to a dense factor. Every
ordering algorithm is a systematic way to eliminate leaves before hubs.

## Where the information comes from: the pattern alone

Because the game never looks at a value, neither does an ordering. Replace
every value with a random one, keeping the matrix positive definite, and the
ordering and the fill are unchanged:

```{code-cell} python
rng = np.random.default_rng(0)
R = A12.copy()
R.data = -rng.uniform(0.1, 1.0, R.nnz)
R = (R + R.T) / 2
R = (R - sp.diags(R.diagonal()) + sp.diags(np.asarray(abs(R).sum(axis=1)).ravel() + 1.0)).tocsr()

p_a = np.asarray(ms.amd(ms.from_scipy(A12)))
p_r = np.asarray(ms.amd(ms.from_scipy(R)))
print("same AMD ordering:", np.array_equal(p_a, p_r))
print("same fill:", cholesky_nnz(A12, p_a), cholesky_nnz(R, p_r))
assert np.array_equal(p_a, p_r) and cholesky_nnz(A12, p_a) == cholesky_nnz(R, p_r)
```

That is what makes the problem tractable: the ordering is a pure graph
problem, solved once, before the numeric factorization. It also means that
values can only make things worse. With Cholesky, every pivot is on the
diagonal and the game is exact. With LU and partial pivoting, the values
decide the row interchanges, and the ordering can only bound the fill (see
[unsymmetric matrices](#unsymmetric-matrices-colamd-and-the-pattern-of-a-a)
below).

Finding the order that minimizes fill is NP-hard (Yannakakis, 1981). The
practical algorithms are heuristics, and the interesting question about
each is what property of the graph it treats as the signal.

## Minimum degree: a local, greedy signal

If the vertex being eliminated has degree d, its clique adds at most
$d(d-1)/2$ edges. So eliminate a vertex of smallest degree, update the
graph, and repeat. This is Markowitz's rule (1957) in its symmetric form,
due to Tinney and Walker (1967).

```{code-cell} python
def minimum_degree(A):
    """Textbook minimum degree: always eliminate a vertex of smallest current degree."""
    g = adjacency(A)
    alive, order = set(range(len(g))), []
    while alive:
        v = min(alive, key=lambda u: (len(g[u]), u))
        order.append(v)
        alive.remove(v)
        nbrs = list(g[v])
        for i, a in enumerate(nbrs):
            for b in nbrs[i + 1 :]:
                g[a].add(b)
                g[b].add(a)
        for a in nbrs:
            g[a].discard(v)
        g[v] = set()
    return np.array(order)


p_md = minimum_degree(A12)
md, amd = cholesky_nnz(A12, p_md), cholesky_nnz(A12, p_a)
print(f"textbook minimum degree: nnz(L) = {md}; mtl5 AMD: {amd}; natural: {cholesky_nnz(A12, np.arange(n12))}")
assert abs(md - amd) < 0.1 * amd
```

The signal is local: the current degree of each vertex in the *evolving*
graph, which changes after every step as fill appears. The textbook version
above is far too slow for real matrices, because it forms every clique
explicitly. Forty years of work went into making it fast without changing
what it computes:

- **The quotient graph** (George and Liu). An eliminated vertex is kept as an
  *element* that stands for its clique, instead of adding the clique's
  edges. Memory never grows beyond the original graph.
- **Indistinguishable vertices.** Vertices with identical neighbourhoods are
  merged into supervariables and eliminated together. Grids and meshes have
  many of them.
- **Multiple elimination** (Liu's MMD). An independent set of minimum-degree
  vertices is eliminated before the degrees are updated.
- **Approximate degrees** (AMD; Amestoy, Davis and Duff, 1996). An upper
  bound on each degree, computed cheaply from the element structure, replaces
  the exact degree. This is what `mtl5.sparse.amd` implements.

Minimum degree's weakness is that it is myopic. It has no view of the
graph's global shape, and on large meshes its fill grows faster than the
best achievable.

## Nested dissection: a global, divide-and-conquer signal

Find a small set of vertices S, a *separator*, whose removal splits the
graph into two parts A and B with no edge between them. Order A first, then
B, then S last.

Eliminating a vertex of A creates fill only among its neighbours, and those
lie in A or S, never in B. So fill can never cross the separator: the
factor's off-diagonal blocks between A and B stay empty, and only S's own
block becomes dense. Apply the same idea to A and B recursively, and you
have nested dissection (George, 1973).

On a grid, the middle row or column is a separator. Here is the recursion on
a 15×15 grid, coloured by depth: the first separator is the darkest, and
each part is split in turn.

```{code-cell} python
def grid_nested_dissection(k, A=None, leaf=1):
    """Geometric nested dissection of a k x k grid: split the longer side at its
    middle line, order both halves, then the separator. Leaves of at most `leaf`
    vertices are ordered with AMD (a hybrid, as production codes use)."""
    order, depth = [], np.full(k * k, -1)

    def finish(cells):
        cells = np.array(cells)
        if A is not None and len(cells) > 1:
            q = np.asarray(ms.amd(ms.from_scipy(A[cells][:, cells].tocsr())))
            cells = cells[q]
        order.extend(cells.tolist())

    def rec(r0, r1, c0, c1, d):
        if r1 <= r0 or c1 <= c0:
            return
        if (r1 - r0) * (c1 - c0) <= leaf:
            finish([r * k + c for r in range(r0, r1) for c in range(c0, c1)])
            return
        if c1 - c0 >= r1 - r0:
            m = (c0 + c1) // 2
            rec(r0, r1, c0, m, d + 1)
            rec(r0, r1, m + 1, c1, d + 1)
            sep = [r * k + m for r in range(r0, r1)]
        else:
            m = (r0 + r1) // 2
            rec(r0, m, c0, c1, d + 1)
            rec(m + 1, r1, c0, c1, d + 1)
            sep = [m * k + c for c in range(c0, c1)]
        depth[sep] = d
        order.extend(sep)

    rec(0, k, 0, k, 0)
    return np.array(order), depth


k = 15
p_nd, depth = grid_nested_dissection(k)
A15 = laplacian_2d(k)
L_nd = ms.to_scipy(ms.cholesky(ms.from_scipy(permuted(A15, p_nd)), ordering="natural").L)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
shade = np.where(depth >= 0, depth, depth.max() + 1).reshape(k, k)
ax1.imshow(shade, cmap="Blues_r", vmin=-1, vmax=depth.max() + 2)
ax1.set_title("separators by depth (darkest first)", fontsize=10)
ax1.set_xticks([])
ax1.set_yticks([])
spy(ax2, L_nd, f"L under nested dissection: nnz = {L_nd.nnz}")
fig.tight_layout()

assert sorted(p_nd.tolist()) == list(range(k * k))
```

The factor shows the structure: dense triangles at the bottom right for the
separators, and empty blocks wherever two parts never meet.

The payoff is asymptotic. A planar graph has separators of $O(\sqrt{n})$
vertices (Lipton and Tarjan, 1979), which makes nested dissection's fill
$O(n \log n)$ and its work $O(n^{3/2})$. That is optimal for planar
graphs: no ordering does better than a constant factor (Hoffman, Martin
and Rose, 1973; Lipton, Rose and Tarjan, 1979). The independent parts A and
B can also be factored in parallel, which is why parallel solvers use it.

Optimal asymptotics do not mean a smaller factor at every size. Here is this
simple geometric version, with AMD on leaves of up to 256 vertices, against
AMD:

```{code-cell} python
def flops(A, p):
    """Cholesky work: the sum over columns of (column count)^2."""
    L = ms.to_scipy(ms.cholesky(ms.from_scipy(permuted(A, p)), ordering="natural").L).tocsc()
    c = np.diff(L.indptr).astype(float)
    return float((c * c).sum()), L.nnz


rows = []
for k in (32, 64, 128, 256):
    A = laplacian_2d(k)
    fa, na = flops(A, np.asarray(ms.amd(ms.from_scipy(A))))
    fn, nn = flops(A, grid_nested_dissection(k, A, leaf=256)[0])
    rows.append((k * k, na, nn, fa, fn))
    print(f"n = {k * k:6d}: nnz(L) AMD {na:9d}, ND {nn:9d} ({nn / na:.2f}x);   "
          f"flops AMD {fa:.2e}, ND {fn:.2e} ({fn / fa:.2f}x)")

ns = np.array([r[0] for r in rows], dtype=float)
slope = lambda y: np.polyfit(np.log(ns), np.log(np.array(y, dtype=float)), 1)[0]
exp = {
    "fill AMD": slope([r[1] for r in rows]),
    "fill ND": slope([r[2] for r in rows]),
    "flops AMD": slope([r[3] for r in rows]),
    "flops ND": slope([r[4] for r in rows]),
}
print({k_: round(v, 2) for k_, v in exp.items()})
assert exp["flops ND"] < exp["flops AMD"]  # ND's work grows more slowly...
assert abs(exp["flops ND"] - 1.62) < 0.05 and abs(exp["flops AMD"] - 1.70) < 0.05  # the quoted exponents
assert rows[-1][4] / rows[-1][3] < rows[0][4] / rows[0][3]  # ...so its disadvantage shrinks
```

At these sizes AMD still produces the smaller factor and does less work. But
nested dissection's work grows more slowly, $n^{1.62}$ against AMD's
$n^{1.70}$ over this range, heading toward its asymptotic $n^{1.5}$, and the
gap closes as n grows. On a 512×512 grid, too large to rebuild on every
documentation build, the two need nearly the same work (ratio 1.02). The
constant matters too: production nested dissection (METIS, Karypis and Kumar
1998) finds better separators than straight lines, by multilevel graph
partitioning, and orders the pieces more carefully. That is why it wins
clearly on large 3-D meshes, where separators are $O(n^{2/3})$ and minimum
degree degrades.

mtl5 does not have a nested dissection ordering yet. It is in progress
upstream as
[stillwater-sc/mtl5#919](https://github.com/stillwater-sc/mtl5/issues/919).

## Reverse Cuthill–McKee: a distance signal

RCM aims at a different target: bandwidth. Start a breadth-first search at a
vertex far from the rest of the graph, a *pseudo-peripheral* vertex, and
number the graph level by level. Every edge then joins vertices in the same
or adjacent levels, so every nonzero lies close to the diagonal
(Cuthill and McKee, 1969).

Why that limits fill: in row i, fill can only appear between the row's first
nonzero and the diagonal. That region is the *envelope*. Nothing outside it
ever fills, so a small envelope bounds the fill. George (1971) observed that
numbering in reverse shrinks the envelope further, hence *reverse*
Cuthill–McKee.

```{code-cell} python
def envelope_size(A):
    """Entries between each row's first nonzero and the diagonal, diagonal included."""
    A = sp.csr_matrix(A)
    first = np.array([A.indices[A.indptr[i] : A.indptr[i + 1]].min() for i in range(A.shape[0])])
    return int((np.arange(A.shape[0]) - first + 1).sum())


k = 12
A = laplacian_2d(k)
p_rcm = np.asarray(ms.rcm(ms.from_scipy(A)))
levels = np.empty(k * k, dtype=int)
levels[p_rcm[::-1]] = np.arange(k * k)  # RCM reverses a BFS numbering

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
ax1.imshow(levels.reshape(k, k), cmap="viridis")
ax1.set_title("Cuthill-McKee numbering: level by level", fontsize=10)
ax1.set_xticks([])
ax1.set_yticks([])
L_rcm = ms.to_scipy(ms.cholesky(ms.from_scipy(permuted(A, p_rcm)), ordering="natural").L)
spy(ax2, L_rcm, f"L under RCM: nnz = {L_rcm.nnz}", SERIES[3])
fig.tight_layout()

env = envelope_size(permuted(A, p_rcm))
print(f"envelope of A under RCM: {env};  nnz(L): {L_rcm.nnz}")
assert L_rcm.nnz <= env  # all fill stays inside the envelope
assert L_rcm.nnz == env  # on a grid, it fills the envelope completely
```

The numbering sweeps the grid in diagonal bands, and L is a band. The signal
is distance in the graph, and the guarantee is containment: fill stays
inside the envelope. On a grid it fills the envelope completely, as the
assert checks. RCM bounds fill rather than minimizing it. That suits
band and profile solvers, and memory locality, but on 2-D and 3-D meshes
minimum degree and nested dissection do much better: the
[SPD example](spd-example.md) measures RCM's fill growing like
$n^{1.5}$, against AMD's $n^{1.25}$.

## Unsymmetric matrices: COLAMD and the pattern of AᵀA

With LU and partial pivoting, the row interchanges depend on the values, and
the values are not known until the numeric factorization runs. So the
ordering cannot choose a row order in advance; it chooses only the column
order Q, and has to cope with whatever rows pivoting picks later.

George and Ng (1985) gave the bound that makes this possible. Whatever the
row interchanges, the factors of $AQ$ fit inside the Cholesky factor $L_c$
of $(AQ)^T(AQ)$: the pattern of U is contained in that of $L_c^T$, and L's in
that of $L_c$. So an ordering that keeps the Cholesky factor of $A^TA$
small also keeps LU's fill small, *for every pivot sequence*.

That is what COLAMD does (Davis, Gilbert, Larimore and Ng, 2004): minimum
degree on the graph of $A^TA$, without ever forming $A^TA$. Each row of A
becomes a clique of the columns it touches, which is exactly the quotient
graph's element structure. On `west0479`:

```{code-cell} python
S = ms.to_scipy(mtl5.io.mm_read("../notebooks/data/west0479.mtx")).tocsr()
S.eliminate_zeros()
pattern = S.copy()
pattern.data[:] = 1.0

print("ordering   nnz(L+U)   2 nnz(Lc)   pattern(U) within pattern(Lc^T)")
bounds = {}
for name in ("colamd", "amd", "natural"):
    F = ms.splu(S, ordering=name)
    AQ = pattern[:, np.asarray(F.col_perm)]
    C = (AQ.T @ AQ).tocsr()
    C = (C + sp.diags(np.asarray(C.sum(axis=1)).ravel() + 1.0)).tocsr()  # same pattern, SPD
    Lc = ms.to_scipy(ms.cholesky(ms.from_scipy(C), ordering="natural").L)
    U = ms.to_scipy(F.U)
    inside = (U != 0).astype(int) - (U != 0).multiply(Lc.T != 0).astype(int)
    contained = inside.count_nonzero() == 0
    bounds[name] = (F.nnz, 2 * Lc.nnz)
    print(f"{name:>8}  {F.nnz:9d}  {2 * Lc.nnz:10d}   {contained}")
    assert contained and F.nnz <= 2 * Lc.nnz

assert bounds["colamd"][1] < bounds["amd"][1] < bounds["natural"][1]
assert bounds["colamd"][0] < bounds["amd"][0] < bounds["natural"][0]
```

The bound is loose, two to six times the actual fill here, but it ranks the
orderings the same way the actual fill does. Minimizing it is minimizing the
right thing.

There is an alternative for LU: bet that pivots will stay on or near the
diagonal, and order the symmetric pattern of $A + A^T$ with AMD instead.
KLU does this for each diagonal block, and so does `splu(A, ordering="amd")`.
Small `threshold` values exist to make the bet pay: they keep the pivot on
the diagonal whenever it is within that factor of the column's largest
entry, as the [diagnostics page](diagnostics.md) measures. When pivoting
does leave the diagonal, the AMD ordering's fill prediction no longer
applies.

## After the ordering: the elimination tree

Once the order is fixed, the pattern of L can be computed exactly, still
without arithmetic, in nearly linear time. The tool is the *elimination
tree*: the parent of column j is the row of the first off-diagonal nonzero
in column j of L. Column j's pattern is then the union of A's column j with
the patterns of its children in the tree (Liu, 1990).

This symbolic factorization is how a factor object knows its `nnz` before
the numeric phase runs, how memory is allocated exactly once, and how the
supernodal methods find columns with identical patterns to process as dense
blocks. Independent subtrees can be factored in parallel, which is the
structural reason nested dissection, whose tree is bushy, parallelizes well,
and RCM, whose tree is close to a path, does not.

## Summary

| ordering | signal from the pattern | scope | best for |
|---|---|---|---|
| minimum degree / `amd` | current degree in the elimination graph | local, greedy, adaptive | general sparse symmetric matrices; the default for `cholesky`, `ldlt` |
| nested dissection (mtl5#919) | small separators | global, divide and conquer | large 2-D and 3-D meshes; parallel factorization |
| `rcm` | graph distance (BFS levels) | global, but aims at bandwidth | banded and profile solvers, locality |
| `colamd` | degrees in the graph of AᵀA (rows as cliques) | local, greedy | LU with partial pivoting; the default for `splu`, `qr` |
| `natural` | none | — | matrices already ordered well, and baselines |

They all read the same pattern. They differ in which feature of it they
trust: local degree, global separators, distance, or the column
intersections that bound fill under any pivoting.

API reference: {py:func}`mtl5.sparse.amd`, {py:func}`mtl5.sparse.colamd`,
{py:func}`mtl5.sparse.rcm`, {py:func}`mtl5.sparse.ordering`,
{py:func}`mtl5.sparse.orderings`, {py:func}`mtl5.sparse.cholesky`,
{py:func}`mtl5.sparse.splu`.
