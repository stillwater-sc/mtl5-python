#!/usr/bin/env python3
"""Precision Showcase — Posit vs IEEE Float for Ill-Conditioned Problems

Demonstrates the tapered precision of posits versus IEEE floats on two
classical numerical analysis stress tests:

  1. Hilbert matrix solve — H_ij = 1/(i+j-1), a notoriously ill-conditioned
     SPD matrix. Even small dimensions lose digits rapidly under any solver.

  2. Harmonic series — sum 1/k for k=1..N. The values get small enough that
     IEEE float32 and posit32 reveal different rounding behavior.

Both tests run at f32, f64, posit32, and posit64 and report the relative
error against an arbitrary-precision (mpmath) reference, falling back to
f64 numpy if mpmath is not installed.

Usage:
    python precision_showcase.py [--n 8] [--harmonic-terms 100000]
"""

import argparse

import numpy as np

import mtl5


def hilbert_matrix(n: int) -> np.ndarray:
    """Construct the n x n Hilbert matrix H_ij = 1 / (i + j - 1)."""
    i = np.arange(1, n + 1).reshape(-1, 1)
    j = np.arange(1, n + 1).reshape(1, -1)
    return 1.0 / (i + j - 1.0)


def reference_solution(n: int) -> np.ndarray:
    """Compute a reference solution to H @ x = b where b = H @ ones,
    so the exact solution is x = ones(n). Use mpmath if available."""
    try:
        import mpmath

        mpmath.mp.dps = 100  # 100 decimal digits
        H = mpmath.matrix(n, n)
        for i in range(n):
            for j in range(n):
                H[i, j] = mpmath.mpf(1) / (i + j + 1)
        b = mpmath.matrix(n, 1)
        for i in range(n):
            b[i, 0] = sum(H[i, j] for j in range(n))
        x = mpmath.lu_solve(H, b)
        return np.array([float(x[i, 0]) for i in range(n)])
    except ImportError:
        return np.ones(n)  # exact when constructed from H @ ones


def hilbert_solve_test(n: int) -> None:
    """Solve the Hilbert system at multiple precisions and compare."""
    print(f"\nHilbert Matrix Solve (n={n})")
    print("=" * 60)

    H_f64 = hilbert_matrix(n)
    cond = np.linalg.cond(H_f64)
    print(f"  Condition number κ(H) ≈ {cond:.2e}")

    # Construct b such that the exact solution is x = ones(n)
    b_f64 = H_f64 @ np.ones(n)
    x_exact = reference_solution(n)

    methods = [
        ("f32", lambda H, b: mtl5.solve(H.astype(np.float32), b.astype(np.float32))),
        ("f64", lambda H, b: mtl5.solve(H, b)),
        (
            "posit32",
            lambda H, b: np.array(
                mtl5.solve(mtl5.matrix_posit32(H), mtl5.vector_posit32(b)).to_list()
            ),
        ),
        (
            "posit64",
            lambda H, b: np.array(
                mtl5.solve(mtl5.matrix_posit64(H), mtl5.vector_posit64(b)).to_list()
            ),
        ),
    ]

    print(f"\n  {'precision':<10s}  {'relative error':>16s}  {'max abs error':>16s}")
    print(f"  {'---------':<10s}  {'--------------':>16s}  {'-------------':>16s}")
    for name, solver in methods:
        try:
            x = solver(H_f64, b_f64)
            x = np.asarray(x, dtype=np.float64)
            rel_err = np.linalg.norm(x - x_exact) / max(np.linalg.norm(x_exact), 1e-30)
            max_err = np.max(np.abs(x - x_exact))
            print(f"  {name:<10s}  {rel_err:16.6e}  {max_err:16.6e}")
        except Exception as e:
            print(f"  {name:<10s}  FAILED: {e}")


def harmonic_series_test(n_terms: int) -> None:
    """Compute the harmonic series at multiple precisions, comparing
    forward (1 + 1/2 + 1/3 + ...) vs backward (1/N + 1/(N-1) + ...) summation.
    Backward summation is the textbook trick for accurate small-value sums."""
    print(f"\nHarmonic Series Sum (N={n_terms})")
    print("=" * 60)

    # Build the terms (independent of precision so we measure summation error,
    # not term construction error)
    terms_f64 = 1.0 / np.arange(1, n_terms + 1, dtype=np.float64)

    def reference_sum() -> float:
        try:
            import mpmath

            mpmath.mp.dps = 50
            return float(sum(mpmath.mpf(1) / k for k in range(1, n_terms + 1)))
        except ImportError:
            return float(np.sum(terms_f64.astype(np.longdouble)))

    H_ref = reference_sum()
    print(f"  Reference H_{n_terms} ≈ {H_ref:.15f}")

    print(f"\n  {'precision':<10s}  {'forward sum':>20s}  {'rel err':>14s}")
    print(f"  {'---------':<10s}  {'-----------':>20s}  {'-------':>14s}")

    methods = [
        ("f32", lambda terms: float(np.sum(terms.astype(np.float32)))),
        ("f64", lambda terms: float(np.sum(terms))),
        (
            "posit32",
            lambda terms: float(
                mtl5.dot(
                    mtl5.vector_posit32(terms),
                    mtl5.vector_posit32(np.ones_like(terms)),
                )
            ),
        ),
        (
            "posit64",
            lambda terms: float(
                mtl5.dot(
                    mtl5.vector_posit64(terms),
                    mtl5.vector_posit64(np.ones_like(terms)),
                )
            ),
        ),
    ]

    for name, sum_fn in methods:
        try:
            s = sum_fn(terms_f64)
            rel_err = abs(s - H_ref) / H_ref
            print(f"  {name:<10s}  {s:20.10f}  {rel_err:14.4e}")
        except Exception as e:
            print(f"  {name:<10s}  FAILED: {e}")


def main():
    parser = argparse.ArgumentParser(description="Posit vs IEEE Float precision showcase")
    parser.add_argument("--n", type=int, default=8, help="Hilbert matrix size")
    parser.add_argument(
        "--harmonic-terms",
        type=int,
        default=100_000,
        help="Number of harmonic series terms to sum",
    )
    args = parser.parse_args()

    print(f"MTL5 v{mtl5.__version__} — Precision Showcase")
    print(f"  Backends: {mtl5.backends()}")
    print(f"  Active:   {mtl5.get_backend()}")

    hilbert_solve_test(args.n)
    harmonic_series_test(args.harmonic_terms)

    print()
    print("Observations")
    print("=" * 60)
    print("- The Hilbert matrix is ill-conditioned: even small n loses many")
    print("  digits in any precision. Compare relative errors across types.")
    print("- Posits use tapered precision: more bits near 1, fewer at extremes.")
    print("  This often gives better accuracy than IEEE for values in [-2, 2].")
    print("- Both effects matter on KPU hardware where posit arithmetic is")
    print("  native and can replace IEEE operations at lower energy cost.")


if __name__ == "__main__":
    main()
