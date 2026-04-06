#!/usr/bin/env python3
"""Iterative Refinement with Mixed Precision — fp16 + f64

Demonstrates MTL5's multi-precision capabilities by solving a linear system
using iterative refinement:

    1. Factorize and solve in fp16 (low precision, fast)
    2. Compute residual r = b - Ax in f64 (high precision)
    3. Solve correction Ae = r in fp16
    4. Update x += e
    5. Repeat until convergence

This is the core algorithm motivating Stillwater's KPU hardware: the
factorization and triangular solves run on energy-efficient fp16 hardware,
while the residual computation uses full f64 to maintain accuracy.

Usage:
    python iterative_refinement.py [--size 20] [--max-iter 20]
"""

import argparse

import numpy as np

import mtl5


def create_test_system(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Create a well-conditioned n×n system Ax = b with known solution.

    Returns (A, b) where A is diagonally dominant and the exact solution is x = [1, 2, ..., n].
    """
    A = rng.standard_normal((n, n))
    # Make diagonally dominant for stability
    A += n * np.eye(n)
    # Known solution
    x_exact = np.arange(1.0, n + 1.0)
    b = A @ x_exact
    return A, b


def iterative_refinement(
    A: np.ndarray,
    b: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    """Solve Ax = b using iterative refinement with fp16 factorization.

    The algorithm:
        1. x₀ = solve_fp16(A, b)         — initial solve in fp16
        2. for k = 1, 2, ...:
             r  = b - A @ x              — residual in f64
             e  = solve_fp16(A, r)        — correction in fp16
             x  = x + e                  — update in f64
             if ||r|| / ||b|| < tol: break

    Returns (x, residual_history).
    """
    n = len(b)
    norm_b = mtl5.norm(b)

    print(f"  System size: {n}×{n}")
    print(f"  ||b||₂ = {norm_b:.6e}")
    print(f"  Target relative residual: {tol:.0e}")
    print()

    # Create fp16 versions of A and b for the low-precision solves
    A_fp16 = mtl5.matrix_fp16(A)
    b_fp16 = mtl5.vector_fp16(b)

    print(f"  A_fp16: {A_fp16}")
    print(f"  b_fp16: dtype={b_fp16.dtype}, size={len(b_fp16)}")
    print()

    # Step 1: Initial solve in fp16
    x_fp16 = mtl5.solve(A_fp16, b_fp16)

    # Convert to f64 for the refinement loop
    x = np.array(x_fp16.to_list())

    # Compute initial residual in f64
    r = b - A @ x
    rel_res = mtl5.norm(r) / norm_b
    residual_history = [rel_res]

    print(f"  {'Iter':>4s}  {'||r||/||b||':>14s}  {'||correction||':>14s}  {'Status':>10s}")
    print(f"  {'----':>4s}  {'-----------':>14s}  {'--------------':>14s}  {'------':>10s}")
    print(f"  {'init':>4s}  {rel_res:14.6e}  {'—':>14s}  {'fp16 solve':>10s}")

    # Step 2: Iterative refinement loop
    for iteration in range(1, max_iter + 1):
        if rel_res < tol:
            print(f"\n  Converged after {iteration - 1} refinement steps.")
            break

        # Solve the correction equation in fp16: A·e ≈ r
        r_fp16 = mtl5.vector_fp16(r)
        e_fp16 = mtl5.solve(A_fp16, r_fp16)
        e = np.array(e_fp16.to_list())

        norm_e = np.linalg.norm(e)

        # Update solution in f64
        x += e

        # Recompute residual in f64
        r = b - A @ x
        rel_res = mtl5.norm(r) / norm_b
        residual_history.append(rel_res)

        status = "converged" if rel_res < tol else "refining"
        print(f"  {iteration:4d}  {rel_res:14.6e}  {norm_e:14.6e}  {status:>10s}")
    else:
        print(f"\n  Did not converge after {max_iter} iterations (||r||/||b|| = {rel_res:.2e})")

    return x, residual_history


def compare_with_full_precision(A: np.ndarray, b: np.ndarray, x_refined: np.ndarray):
    """Compare iterative refinement result with full f64 solve."""
    n = len(b)
    x_exact = np.arange(1.0, n + 1.0)

    # Full f64 solve via MTL5
    x_f64 = mtl5.solve(A, b)

    # Full f64 solve via NumPy
    x_numpy = np.linalg.solve(A, b)

    # Pure fp16 solve (no refinement)
    A_fp16 = mtl5.matrix_fp16(A)
    b_fp16 = mtl5.vector_fp16(b)
    x_fp16_only = np.array(mtl5.solve(A_fp16, b_fp16).to_list())

    print("\nAccuracy Comparison:")
    print(f"  {'Method':<30s}  {'||x - x_exact||/||x_exact||':>26s}")
    print(f"  {'------':<30s}  {'--------------------------':>26s}")

    for name, x_sol in [
        ("fp16 only (no refinement)", x_fp16_only),
        ("fp16 + iterative refinement", x_refined),
        ("MTL5 f64 solve", x_f64),
        ("NumPy f64 solve", x_numpy),
    ]:
        rel_err = np.linalg.norm(x_sol - x_exact) / np.linalg.norm(x_exact)
        print(f"  {name:<30s}  {rel_err:26.6e}")

    # Summarize: compare absolute errors to avoid division-by-zero artifacts
    eps = np.finfo(np.float64).eps * np.linalg.norm(x_exact)
    abs_err_fp16 = np.linalg.norm(x_fp16_only - x_exact)
    abs_err_refined = np.linalg.norm(x_refined - x_exact)
    abs_err_f64 = np.linalg.norm(x_f64 - x_exact)

    print(f"\n  fp16-only absolute error:   {abs_err_fp16:.6e}")
    print(f"  Refined absolute error:     {abs_err_refined:.6e}")
    print(f"  f64 solve absolute error:   {abs_err_f64:.6e}")
    print(f"  Machine epsilon × ||x||:    {eps:.6e}")

    if abs_err_refined < eps:
        print("\n  Refinement recovered full f64 machine precision from fp16.")
    elif abs_err_fp16 > 0:
        improvement = abs_err_fp16 / abs_err_refined
        digits = np.log10(improvement)
        print(
            f"\n  Refinement improved accuracy by {improvement:.1f}x ({digits:.1f} decimal digits)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Iterative Refinement — Mixed Precision fp16/f64 with MTL5"
    )
    parser.add_argument("--size", type=int, default=20, help="System size n")
    parser.add_argument("--max-iter", type=int, default=20, help="Max refinement iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Show convergence plot")
    args = parser.parse_args()

    print(f"MTL5 v{mtl5.__version__} — Iterative Refinement (fp16 + f64)")
    print(f"  Available devices: {mtl5.devices()}")
    print(f"  System size: {args.size}")
    print()

    rng = np.random.default_rng(args.seed)
    A, b = create_test_system(args.size, rng)

    print("Phase 1: Iterative Refinement")
    print("=" * 60)
    x_refined, residuals = iterative_refinement(A, b, max_iter=args.max_iter)

    print()
    print("Phase 2: Accuracy Comparison")
    print("=" * 60)
    compare_with_full_precision(A, b, x_refined)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.semilogy(range(len(residuals)), residuals, "bo-", markersize=6)
            ax.axhline(y=1e-10, color="r", linestyle="--", label="Target tolerance")
            ax.set_xlabel("Refinement Iteration")
            ax.set_ylabel("Relative Residual ||r||/||b||")
            ax.set_title("Iterative Refinement Convergence — fp16 factorization + f64 residual")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, len(residuals) - 0.5)
            plt.tight_layout()
            plt.savefig("iterative_refinement.png", dpi=150)
            print("\nPlot saved to iterative_refinement.png")
            plt.show()
        except ImportError:
            print("\nmatplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
