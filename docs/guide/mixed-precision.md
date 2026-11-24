---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Mixed precision: a narrow factor, refined

The factorization is the expensive part of a direct solve: it holds the most
memory and does almost all of the arithmetic. Iterative refinement lets that
part run in a narrow precision while the answer keeps the accuracy of a wide
one. Factor in `float32`, then correct the solution with residuals computed in
`float64`:

$$
r = b - Ax \;\;(\texttt{float64}), \qquad
\delta = \hat{L}^{-T}\hat{L}^{-1} r \;\;(\texttt{float32 factor}), \qquad
x \leftarrow x + \delta \;\;(\texttt{float64}).
$$

Each pass removes most of the remaining error, as long as the narrow factor is
a good enough approximation of $A$. The idea is old: Wilkinson analysed it in
1963 and Moler in 1967. It came back when hardware made single precision much
faster than double (Langou et al., SC'06), and Carson and Higham (SIAM J. Sci.
Comput. 40(2), 2018) extended it to three precisions and to GMRES-based
refinement, which this page uses at the end.

In mtl5 it is one call, `mtl5.mixed.iterative_refine(A, M, b)`: `A` and `b`
are `float64`, and `M` is any sparse direct factorization (or an ILU(0) or
IC(0) preconditioner), in either precision. This page measures what it buys
and where it stops working. The page is executed when the documentation is
built, and the `assert` statements check the claims the prose makes.

```{code-cell} python
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mtl5
import mtl5.sparse as ms

iterative_refine = mtl5.mixed.iterative_refine


def best_ms(fn, repeats=3):
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3, out


def fwd_err(x, x_true):
    return np.abs(x - x_true).max() / np.abs(x_true).max()
```

## Getting float64 accuracy from a float32 factor

The test problem is the 2-D Poisson matrix from the
[SPD example](spd-example.md), with a random solution. The precision of a
factorization follows the dtype of the matrix it is given, so a `float32` copy
of $A$ gives a `float32` factor.

```{code-cell} python
k = 64
A = mtl5.generators.poisson2d(k, k)
S = ms.to_scipy(A)
n = S.shape[0]
x_true = np.random.default_rng(0).standard_normal(n)
b = S @ x_true

chol64 = ms.cholesky(A)
chol32 = ms.cholesky(ms.from_scipy(S.astype(np.float32)))
print(chol64)
print(chol32)

x64 = chol64.solve(b).to_numpy()
x32 = chol32.solve(b.astype(np.float32)).to_numpy().astype(np.float64)
x_ir, info = iterative_refine(A, chol32, b, rel_tol=1e-15)

print(f"float64 factor, direct solve:      forward error {fwd_err(x64, x_true):.1e}")
print(f"float32 factor, direct solve:      forward error {fwd_err(x32, x_true):.1e}")
print(f"float32 factor + float64 refining: forward error {fwd_err(x_ir, x_true):.1e}")
print(info)

assert fwd_err(x32, x_true) > 1e-7  # float32's unit roundoff is 6e-8
assert info["converged"] and info["iters"] <= 5
assert fwd_err(x_ir, x_true) < 2 * fwd_err(x64, x_true)
```

A few refinement steps take the `float32` answer from about six correct digits
to the accuracy of the `float64` solve.

`info` reports what happened:

- `iters` is the number of corrections applied.
- `rel_residual` is $\|b - Ax\|_\infty / \|b\|_\infty$ for the returned $x$.
- `converged` is true when `rel_tol` was met. `rel_tol` defaults to 0, which
  disables the test: refinement then runs until the residual stops improving
  for `patience` consecutive steps (default 3), or for `max_iter` steps
  (default 20), and reports `converged=False` even when it reached the
  `float64` floor. Pass a `rel_tol` when you want the flag to mean something.

The returned $x$ is always the best iterate seen, so a generous `max_iter`
never makes the answer worse.

## What it buys

The `float32` factor stores its values in half the bytes. Both factors have
the same pattern, so their index arrays are the same size, and the total
saving is somewhat less than half:

```{code-cell} python
assert chol64.nnz == chol32.nnz  # same ordering, same pattern
print(f"factor values: float64 {chol64.nnz * 8 / 1e6:.2f} MB, float32 {chol32.nnz * 4 / 1e6:.2f} MB")
```

Time is a different matter. Here are both factorizations, and the solve
phases, on a $128 \times 128$ grid:

```{code-cell} python
k2 = 128
A2 = mtl5.generators.poisson2d(k2, k2)
S2 = ms.to_scipy(A2)
b2 = S2 @ np.ones(S2.shape[0])
A2_32 = ms.from_scipy(S2.astype(np.float32))

print(f"{'':>16} {'factor f64':>11} {'solve f64':>10} {'factor f32':>11} {'refine':>8}")
for name, factor in (("cholesky", ms.cholesky), ("supernodal_ldlt", ms.supernodal_ldlt)):
    t64, f64 = best_ms(lambda: factor(A2))
    t32, f32 = best_ms(lambda: factor(A2_32))
    ts, _ = best_ms(lambda: f64.solve(b2))
    tr, (x, inf) = best_ms(lambda: iterative_refine(A2, f32, b2, rel_tol=1e-15))
    assert inf["converged"]
    print(f"{name:>16} {t64:9.1f}ms {ts:8.1f}ms {t32:9.1f}ms {tr:6.1f}ms  ({inf['iters']} corrections)")
```

On the CPU that built this page, the `float32` factorization is not much
faster than the `float64` one, and refinement adds a few solves. The sparse
factorizations here spend their time on index handling and scattered memory
access, not on wide arithmetic, so halving the width of the values does not
halve the time. Mixed precision pays off when:

- **memory is the limit.** The factor is the largest object in a direct solve.
  Half the bytes of values can be the difference between a problem that fits
  and one that does not.
- **bandwidth is the limit.** Each solve streams the whole factor from memory.
  With many right-hand sides, or repeated solves inside a Newton or
  time-stepping loop, the narrow factor moves half the bytes.
- **narrow arithmetic is much faster than wide.** That is true of SIMD units,
  of GPUs, and of accelerators built around narrow number formats, such as
  Stillwater's KPU. Splitting the work so that the factorization runs in the
  narrow format and the residual stays in `float64` on the host is the
  pattern `iterative_refine` is designed for.

## When it fails: conditioning

Refinement converges when the narrow factor is close enough to $A$ that each
correction removes more error than it introduces. The rule of thumb is
$\kappa(A)\,u_{\text{factor}} < 1$, where $u$ is the unit roundoff of the
factor's precision: $6 \times 10^{-8}$ for `float32`, so $\kappa(A)$ up to
about $10^7$.

To see the limit, take the grid Laplacian with *Neumann* boundaries. It is
singular, because the constant vector is in its nullspace, and adding
$\varepsilon I$ makes it SPD with smallest eigenvalue exactly $\varepsilon$.
Its largest is below 8 (no row sum of absolute values exceeds 8), so
$\kappa \approx 8/\varepsilon$, and $\varepsilon$ dials in the conditioning.

```{code-cell} python
T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k)).tolil()
T[0, 0] = T[-1, -1] = 1.0  # Neumann ends
T = T.tocsr()
eye = sp.identity(k, format="csr")
neumann = (sp.kron(T, eye) + sp.kron(eye, T)).tocsr()
assert abs(neumann @ np.ones(n)).max() == 0  # singular: constants are in the nullspace


def shifted(eps):
    return (neumann + eps * sp.identity(n, format="csr")).tocsr()


print(f"{'kappa':>8} {'corrections':>11} {'converged':>9} {'fwd err, refined':>17} {'fwd err, f64 direct':>20}")
sweep = {}
for eps in [1e-1, 1e-3, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]:
    M = shifted(eps)
    rhs = M @ x_true
    f32 = ms.cholesky(ms.from_scipy(M.astype(np.float32)))
    x, inf = iterative_refine(ms.from_scipy(M), f32, rhs, rel_tol=1e-14, max_iter=50)
    x_d = ms.cholesky(ms.from_scipy(M)).solve(rhs).to_numpy()
    sweep[eps] = (inf, fwd_err(x, x_true), fwd_err(x_d, x_true))
    print(f"{8 / eps:8.0e} {inf['iters']:11d} {str(inf['converged']):>9} "
          f"{fwd_err(x, x_true):17.1e} {fwd_err(x_d, x_true):20.1e}")

assert all(sweep[e][0]["converged"] for e in (1e-1, 1e-3, 1e-5, 1e-6, 1e-7))
assert not any(sweep[e][0]["converged"] for e in (1e-8, 1e-10))

# Where it converges, the forward error is within the kappa * u64 bound
# that any backward-stable float64 solve is held to (with a factor 10 of slack).
u64 = np.finfo(np.float64).eps / 2
for eps, (inf, err_refined, _) in sweep.items():
    if inf["converged"]:
        assert err_refined < 10 * (8 / eps) * u64, (eps, err_refined)
```

Two things to read off the table:

- **Up to $\kappa \approx 10^8$ here, refinement converges.** The rule of
  thumb is conservative by an order of magnitude on this matrix. Refinement
  needs more steps as $\kappa$ grows, because each step removes a smaller
  fraction of the error. The forward error grows with $\kappa$ too, and
  stays within the $\kappa\,u_{64}$ bound that any backward-stable `float64`
  solve is held to. Within that bound it is not always as small as the
  `float64` direct solve's: refinement drives the *residual* to the
  `float64` level, and a small residual guarantees no more than the bound.
- **Above it, refinement stalls.** The `float32` factor is no longer accurate
  enough for the corrections to contract, `converged` stays false, and the
  answer can be several orders of magnitude worse than the `float64` solve.
  Always pass a `rel_tol` and check `info["converged"]`: a stalled refinement
  raises no error.

### What to do instead: GMRES with the narrow factor

The `float32` factor is still a very good *preconditioner* at those condition
numbers, even when it is too poor to drive plain refinement. Carson and
Higham's GMRES-based refinement uses it that way: GMRES, run in `float64`,
preconditioned by the narrow factor. The same factor rescues both stalled
cases:

```{code-cell} python
for eps in (1e-8, 1e-10):
    M = shifted(eps)
    rhs = M @ x_true
    f32 = ms.cholesky(ms.from_scipy(M.astype(np.float32)))
    P = spla.LinearOperator(
        (n, n), matvec=lambda r, f=f32: f.solve(np.ravel(r).astype(np.float32)).to_numpy().astype(np.float64)
    )
    x, gm_info = spla.gmres(M, rhs, M=P, rtol=1e-14, restart=50, maxiter=20)
    x_d = sweep[eps][2]
    print(f"kappa {8 / eps:.0e}: GMRES info = {gm_info}, forward error {fwd_err(x, x_true):.1e} "
          f"(refinement alone {sweep[eps][1]:.1e}, float64 direct {x_d:.1e})")
    assert gm_info == 0
    assert fwd_err(x, x_true) < 10 * x_d
```

GMRES costs more per iteration than a refinement step, so use plain refinement
while it converges and fall back to GMRES when it does not.

## When it fails: underflow, and `scaled=True`

A second failure has nothing to do with conditioning. `float32` cannot
represent magnitudes much below $10^{-38}$, and the residuals of a problem
whose data is that small underflow when they are cast into the `float32`
correction solve. Refinement then stalls at `float32` accuracy, even on a
well-conditioned matrix.

`scaled=True` divides each residual by its norm before the correction solve,
and multiplies the correction back in `float64`. Only the shape of the
correction passes through the narrow factor; its size stays in `float64`.

```{code-cell} python
tiny = 1e-40
b_tiny = b * tiny
for scaled in (False, True):
    x, inf = iterative_refine(A, chol32, b_tiny, rel_tol=1e-14, scaled=scaled)
    print(f"scaled={scaled!s:>5}: converged={inf['converged']!s:>5}, "
          f"forward error {fwd_err(x, x_true * tiny):.1e}")
    if scaled:
        assert inf["converged"] and fwd_err(x, x_true * tiny) < 1e-13
    else:
        assert not inf["converged"] and fwd_err(x, x_true * tiny) > 1e-7
```

Scaling costs one extra pass over the residual per step. It matters most for
number formats with a much narrower exponent range than `float32`, such as
`fp16` or the 8-bit formats. Those are available for dense matrices through
`mtl5.mixed.lu_iterative_refine(A, b, working="fp16", scaled=True)`.

## Summary

| Situation | What to use |
|---|---|
| $\kappa(A) \lesssim 10^7$ (often more), memory or bandwidth bound | `float32` factor + `iterative_refine(..., rel_tol=...)` |
| $\kappa(A)$ larger, or refinement does not converge | `float32` factor as the preconditioner for `float64` GMRES |
| data near the bottom of the narrow format's range | `iterative_refine(..., scaled=True)` |
| none of the above, and time is the only concern on a CPU | a `float64` factor |

`mtl5.mixed.lu_iterative_refine(A, b, working=...)` does the same for dense
matrices. Its factor can be in `f32`, `fp16`, `fp8`, a posit, a fixed-point or
a logarithmic format, which makes it the place to explore how narrow a factor
can get before refinement gives up.
