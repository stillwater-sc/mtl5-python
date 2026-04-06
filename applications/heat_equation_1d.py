#!/usr/bin/env python3
"""Finite Difference Heat Equation (1D) — Implicit Euler Scheme

Solves the transient heat equation:

    ∂T/∂t = α ∂²T/∂x²

on a 1D rod of length L with boundary conditions T(0,t) = T(L,t) = 0
and an initial Gaussian hot spot at the center.

Each timestep solves a tridiagonal system Ax = b via mtl5.solve(),
demonstrating zero-copy NumPy ↔ MTL5 interop for a real PDE solver.

Usage:
    python heat_equation_1d.py [--nx 50] [--nt 200] [--plot]
"""

import argparse

import numpy as np

import mtl5


def assemble_implicit_matrix(nx: int, alpha: float, dt: float, dx: float) -> np.ndarray:
    """Assemble the (nx-2) × (nx-2) tridiagonal matrix for implicit Euler.

    The implicit scheme: (I - α·dt/dx² · D²) T^{n+1} = T^n
    where D² is the second-difference operator.
    """
    r = alpha * dt / dx**2
    n = nx - 2  # interior nodes only

    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1.0 + 2.0 * r
        if i > 0:
            A[i, i - 1] = -r
        if i < n - 1:
            A[i, i + 1] = -r
    return A


def initial_condition(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """Gaussian pulse centered at x0 with width sigma."""
    return np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))


def solve_heat_equation(
    L: float = 1.0,
    nx: int = 50,
    nt: int = 200,
    alpha: float = 0.01,
    t_final: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 1D heat equation and return (x, time_snapshots, T_snapshots)."""
    dx = L / (nx - 1)
    dt = t_final / nt
    x = np.linspace(0, L, nx)

    # Initial condition: Gaussian hot spot at center
    T = initial_condition(x, L / 2, L / 10)
    T[0] = T[-1] = 0.0  # Dirichlet BCs

    # Assemble the implicit matrix (constant across timesteps)
    A = assemble_implicit_matrix(nx, alpha, dt, dx)

    # Create MTL5 matrix view (zero-copy)
    A_mtl5 = mtl5.matrix(A)
    print(f"System matrix: {A_mtl5}")
    print(f"  dtype={A_mtl5.dtype}, device={A_mtl5.device}, is_view={A_mtl5.is_view}")

    # Store snapshots for visualization
    snapshot_interval = max(1, nt // 10)
    snapshots = [T.copy()]
    times = [0.0]

    # Time-stepping loop
    for step in range(1, nt + 1):
        # RHS is the interior temperatures from previous timestep
        b = T[1:-1].copy()  # copy needed — solve may read while we write

        # Solve the tridiagonal system using MTL5
        T_interior = mtl5.solve(A, b)

        # Update interior nodes
        T[1:-1] = T_interior

        # Monitor convergence via L2 norm of the solution
        if step % snapshot_interval == 0 or step == nt:
            norm_T = mtl5.norm(T)
            t = step * dt
            print(f"  step {step:4d}/{nt}, t={t:.4f}, ||T||₂ = {norm_T:.6e}")
            snapshots.append(T.copy())
            times.append(t)

    return x, np.array(times), np.array(snapshots)


def main():
    parser = argparse.ArgumentParser(description="1D Heat Equation — Implicit Euler with MTL5")
    parser.add_argument("--nx", type=int, default=50, help="Number of spatial nodes")
    parser.add_argument("--nt", type=int, default=200, help="Number of timesteps")
    parser.add_argument("--alpha", type=float, default=0.01, help="Thermal diffusivity")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    args = parser.parse_args()

    print(f"MTL5 v{mtl5.__version__} — 1D Heat Equation Solver")
    print(f"  Grid: {args.nx} nodes, {args.nt} timesteps, α={args.alpha}")
    print(f"  Available devices: {mtl5.devices()}")
    print()

    x, times, snapshots = solve_heat_equation(nx=args.nx, nt=args.nt, alpha=args.alpha)

    print(f"\nFinal ||T||₂ = {mtl5.norm(snapshots[-1]):.6e}")
    print(f"Max temperature: {snapshots[-1].max():.6e} (started at {snapshots[0].max():.6e})")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Temperature profiles at different times
            for i, (t, T) in enumerate(zip(times, snapshots)):
                color = plt.cm.coolwarm(i / len(times))
                ax1.plot(x, T, color=color, label=f"t={t:.3f}")
            ax1.set_xlabel("x")
            ax1.set_ylabel("T(x, t)")
            ax1.set_title("Temperature Evolution")
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)

            # Space-time heatmap
            im = ax2.imshow(
                snapshots,
                aspect="auto",
                extent=[0, 1, times[-1], 0],
                cmap="hot",
            )
            ax2.set_xlabel("x")
            ax2.set_ylabel("t")
            ax2.set_title("Temperature (space-time)")
            plt.colorbar(im, ax=ax2, label="T")

            plt.suptitle("1D Heat Equation — Implicit Euler with MTL5", fontsize=14)
            plt.tight_layout()
            plt.savefig("heat_equation_1d.png", dpi=150)
            print("Plot saved to heat_equation_1d.png")
            plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
