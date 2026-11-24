---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Coming from SciPy

`mtl5.sparse` accepts SciPy sparse matrices directly, and its API follows
`scipy.sparse.linalg` where it can. This page maps one onto the other, and
spells out the places where they differ. This page is executed when the site
is built, so each mapping below is checked against SciPy itself.

```{code-cell} python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mtl5
import mtl5.sparse as ms

S = ms.to_scipy(mtl5.io.mm_read("../notebooks/data/west0479.mtx"))
S.eliminate_zeros()
n = S.shape[0]
b = S @ np.ones(n)
```

## Matrices

The solvers in `mtl5.sparse` accept two kinds of matrix:

- a SciPy sparse matrix of `float32` or `float64`, in any format (CSR, CSC,
  COO, …). Other dtypes raise `TypeError`; convert first, for example with
  `A.astype(np.float64)`.
- an MTL5 CSR matrix, `SparseMatrix_f32` or `SparseMatrix_f64`. MTL5's COO
  and ELL containers are not accepted by the solvers.

`from_scipy` and `to_scipy` convert explicitly, and both **copy**. MTL5 stores
CSR in its own containers, so a SciPy matrix and its MTL5 counterpart never
share memory. When the same matrix is used repeatedly, convert it once and
reuse the result.

```{code-cell} python
A = ms.from_scipy(S)
print(type(A).__name__, A.shape, A.nnz)

# from_scipy copies: changing the SciPy matrix afterwards leaves A unchanged.
src = S.copy()
copied = ms.from_scipy(src)
src.data[:] = 0.0
assert np.array_equal(ms.to_scipy(copied).data, S.data)

# to_scipy copies: changing the result leaves the MTL5 matrix unchanged.
out = ms.to_scipy(copied)
out.data[:] = 0.0
assert np.array_equal(ms.to_scipy(copied).data, S.data)
print("from_scipy and to_scipy both copy")
```

## Direct solvers

| SciPy | mtl5 |
|---|---|
| `spsolve(A, b)` | `ms.splu(A).solve(b)`, for a 1-D `b` only: `spsolve` also takes a matrix of right-hand sides, `solve` does not. Solve one column at a time; the factorization is reused, so each extra column costs only the triangular solves. |
| `splu(A, permc_spec="COLAMD")` | `ms.splu(A, ordering="colamd")`, the default |
| `permc_spec="NATURAL"` | `ordering="natural"` |
| `permc_spec="MMD_AT_PLUS_A"` | `ordering="amd"` (approximate minimum degree on $A + A^T$) |
| `permc_spec="MMD_ATA"` | no direct equivalent: `colamd` orders $A^TA$ without forming it |
| `factorized(A)` | `ms.splu(A).solve` |
| `SuperLU.L`, `SuperLU.U` | `.L`, `.U` (MTL5 sparse matrices; `ms.to_scipy` converts) |
| `SuperLU.perm_r`, `SuperLU.perm_c` | `.row_perm`, `.col_perm`, in the inverse convention: see below |
| `SuperLU.solve(b, trans="T")` | not available: only $Ax = b$ |
| no equivalent | `.refactor(A2)`, numeric refactorization with the analysis reused |
| no equivalent (scikit-sparse's CHOLMOD) | `ms.cholesky`, `ms.ldlt`, `ms.supernodal_ldlt` |
| `lsqr(A, b)`, `lsmr(A, b)` (iterative) | `ms.qr(A).solve(b)` (direct sparse QR), for $m \ge n$ only: `lsqr` and `lsmr` also handle underdetermined systems, `ms.qr` does not |

Both libraries solve to the same accuracy:

```{code-cell} python
x_scipy = spla.spsolve(S.tocsc(), b)
x_mtl5 = ms.splu(S).solve(b).to_numpy()
print(f"max |x_scipy - x_mtl5| = {np.abs(x_scipy - x_mtl5).max():.1e}")
assert np.allclose(x_scipy, x_mtl5, rtol=0, atol=1e-8)
```

### The permutations point the other way

SciPy documents its LU as $P_r A P_c = LU$, with
`Pr = csc_matrix((ones, (perm_r, arange(n))))`. Read as an index, its
`perm_r[old] = new`. mtl5 uses `p[new] = old`, which indexes the original
matrix directly:

$$A[\text{row\_perm}][:,\ \text{col\_perm}] = L\,U$$

The two conventions are inverses of each other. For a given factorization,
`argsort` converts SciPy's permutations into mtl5's convention. The two
libraries still choose different pivots and orderings for the same matrix,
so their permutation arrays generally differ even after conversion:

```{code-cell} python
f = spla.splu(S.tocsc(), permc_spec="COLAMD")
row_perm, col_perm = np.argsort(f.perm_r), np.argsort(f.perm_c)

err = spla.norm(S[row_perm][:, col_perm] - f.L @ f.U) / spla.norm(S)
print(f"SciPy's factors, mtl5's convention: ||A[row_perm][:, col_perm] - LU|| / ||A|| = {err:.1e}")
assert err < 1e-14

lu = ms.splu(S)
L, U = ms.to_scipy(lu.L), ms.to_scipy(lu.U)
err = spla.norm(S[lu.row_perm][:, lu.col_perm] - L @ U) / spla.norm(S)
print(f"mtl5's factors:                     ||A[row_perm][:, col_perm] - LU|| / ||A|| = {err:.1e}")
assert err < 1e-14
```

The factors themselves differ, because the two libraries choose different
pivots and orderings. Their size is comparable; the
[orderings guide](../notebooks/sparse_direct_orderings) compares them in
detail.

## Iterative solvers

`ms.cg`, `ms.gmres`, `ms.bicgstab`, `ms.minres` and the other Krylov methods
take `rtol=`, `maxiter=` and a preconditioner `M=`, and return `(x, info)`
the way SciPy's do: `info == 0` means converged. The preconditioner is one of
mtl5's own (`ms.ilu0`, `ms.ic0`, `ms.ilut`, …), applied without a round trip
through Python on every iteration.

The bridge also works the other way, so SciPy's solvers can use MTL5 pieces:

- `ms.as_linear_operator(A)` wraps an MTL5 matrix as a
  `scipy.sparse.linalg.LinearOperator`, and its products run in MTL5.
- `ms.as_preconditioner_lo(pc, n)` wraps an ILU(0) or IC(0) preconditioner as
  SciPy's `M=`.
- A direct factorization can serve as a preconditioner too, wrapped in a
  `LinearOperator` by hand. That is most useful when the factor is cheaper
  than the operator. Here a float32 factor preconditions a float64 GMRES:

```{code-cell} python
lu32 = ms.splu(S.astype(np.float32))
# SciPy may hand matvec an (n, 1) column; solve() takes 1-D, so flatten it.
# SciPy restores the shape of the result itself.
M = spla.LinearOperator(
    (n, n), matvec=lambda r: lu32.solve(np.ravel(r)).to_numpy().astype(np.float64)
)
# The column form works too, which a matrix product with M relies on.
assert M.matvec(np.ones((n, 1))).shape == (n, 1)

x, info = spla.gmres(S, b, M=M, rtol=1e-12, restart=20, maxiter=50)
print(f"SciPy GMRES with an MTL5 float32 LU preconditioner: info = {info}, "
      f"relative residual {np.linalg.norm(S @ x - b) / np.linalg.norm(b):.1e}")
assert info == 0
```

`mtl5.mixed.iterative_refine` does the same job, refining a narrow factor
against a float64 residual, without leaving MTL5.

API reference: {py:func}`mtl5.sparse.from_scipy`, {py:func}`mtl5.sparse.to_scipy`, {py:func}`mtl5.sparse.splu`, {py:func}`mtl5.sparse.qr`, {py:func}`mtl5.mixed.iterative_refine`, {py:func}`mtl5.io.mm_read`.
