#!/usr/bin/env python3
"""2D Truss Structural Analysis — Direct Stiffness Method

Analyzes a simple 2D truss (Warren bridge) using the direct stiffness method:

    1. Define nodes and elements (bars)
    2. Assemble global stiffness matrix K from element contributions
    3. Apply boundary conditions (pin and roller supports)
    4. Solve Ku = F for nodal displacements using mtl5.solve()
    5. Compute member forces and stresses

Demonstrates MTL5 zero-copy matrix assembly and linear solve for a
classical structural engineering problem.

Usage:
    python truss_analysis_2d.py [--plot]
"""

import argparse

import numpy as np

import mtl5


def element_stiffness(
    x1: float, y1: float, x2: float, y2: float, E: float, A: float
) -> tuple[np.ndarray, float, float, float]:
    """Compute 4×4 element stiffness matrix in global coordinates.

    Returns (k_global, length, cos_theta, sin_theta).
    """
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    c = dx / L  # cos(θ)
    s = dy / L  # sin(θ)

    # Element stiffness in global coordinates
    k = (E * A / L) * np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ]
    )
    return k, L, c, s


def assemble_global_stiffness(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    A: float,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Assemble global stiffness matrix from element contributions.

    Returns (K, element_props) where element_props is [(L, cos, sin), ...].
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))
    element_props = []

    for elem in elements:
        i, j = int(elem[0]), int(elem[1])
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]

        k_elem, L, c, s = element_stiffness(x1, y1, x2, y2, E, A)
        element_props.append((L, c, s))

        # DOF indices: node i → [2i, 2i+1], node j → [2j, 2j+1]
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += k_elem[a, b]

    return K, element_props


def apply_boundary_conditions(
    K: np.ndarray, F: np.ndarray, fixed_dofs: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply boundary conditions by zeroing rows/cols and setting diagonal to 1."""
    K_bc = K.copy()
    F_bc = F.copy()
    for dof in fixed_dofs:
        K_bc[dof, :] = 0.0
        K_bc[:, dof] = 0.0
        K_bc[dof, dof] = 1.0
        F_bc[dof] = 0.0
    return K_bc, F_bc


def compute_member_forces(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
    E: float,
    A: float,
    element_props: list[tuple[float, float, float]],
) -> np.ndarray:
    """Compute axial force in each member from nodal displacements."""
    forces = np.zeros(len(elements))
    for idx, elem in enumerate(elements):
        i, j = int(elem[0]), int(elem[1])
        L, c, s = element_props[idx]

        # Extract element displacements
        u = np.array(
            [
                displacements[2 * i],
                displacements[2 * i + 1],
                displacements[2 * j],
                displacements[2 * j + 1],
            ]
        )

        # Axial deformation in local coordinates
        delta = -c * u[0] - s * u[1] + c * u[2] + s * u[3]
        forces[idx] = E * A * delta / L

    return forces


def warren_bridge_truss() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], float, float]:
    """Define a Warren bridge truss with 9 nodes and 15 members.

    Geometry:
        Bottom chord: nodes 0-1-2-3-4 (y=0)
        Top chord:    nodes 5-6-7 (y=h)
        Diagonals connect bottom to top in a zigzag pattern.

    Returns: (nodes, elements, forces, fixed_dofs, E, A)
    """
    height = 3.0  # meters

    # Node coordinates [x, y]
    nodes = np.array(
        [
            [0.0, 0.0],  # 0: left support (bottom)
            [3.0, 0.0],  # 1: bottom
            [6.0, 0.0],  # 2: bottom center
            [9.0, 0.0],  # 3: bottom
            [12.0, 0.0],  # 4: right support (bottom)
            [1.5, height],  # 5: top left
            [6.0, height],  # 6: top center
            [10.5, height],  # 7: top right
        ]
    )

    # Element connectivity [node_i, node_j]
    elements = np.array(
        [
            # Bottom chord
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            # Top chord
            [5, 6],
            [6, 7],
            # Diagonals (Warren pattern)
            [0, 5],
            [5, 1],
            [1, 6],
            [6, 2],
            [2, 6],  # note: same as [6,2] but different direction check
            [6, 3],
            [3, 7],
            [7, 4],
        ]
    )

    # Remove duplicate element [2,6] / [6,2]
    elements = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],  # bottom chord
            [5, 6],
            [6, 7],  # top chord
            [0, 5],
            [5, 1],
            [1, 6],
            [6, 3],
            [3, 7],
            [7, 4],  # diagonals
            [5, 2],
            [2, 7],  # inner diagonals
        ]
    )

    # Material properties (steel)
    E = 200e9  # Young's modulus [Pa]
    A = 0.001  # Cross-sectional area [m²]

    # Applied loads: 10 kN downward at each bottom interior node
    n_dof = 2 * len(nodes)
    F = np.zeros(n_dof)
    load = -10e3  # 10 kN downward
    for node in [1, 2, 3]:
        F[2 * node + 1] = load  # y-direction

    # Boundary conditions: pin at node 0 (fix x,y), roller at node 4 (fix y)
    fixed_dofs = [0, 1, 9]  # node 0: dof 0,1; node 4: dof 9

    return nodes, elements, F, fixed_dofs, E, A


def main():
    parser = argparse.ArgumentParser(description="2D Truss Analysis with MTL5")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    args = parser.parse_args()

    print(f"MTL5 v{mtl5.__version__} — 2D Truss Structural Analysis")
    print(f"  Available devices: {mtl5.devices()}")
    print()

    # Define the truss
    nodes, elements, F, fixed_dofs, E, A = warren_bridge_truss()
    n_nodes = len(nodes)
    n_elements = len(elements)
    n_dof = 2 * n_nodes

    print("Warren Bridge Truss:")
    print(f"  {n_nodes} nodes, {n_elements} members, {n_dof} DOFs")
    print(f"  E = {E:.0e} Pa, A = {A:.4f} m²")
    print(f"  Fixed DOFs: {fixed_dofs}")
    print()

    # Assemble global stiffness matrix
    K, element_props = assemble_global_stiffness(nodes, elements, E, A)
    print(f"Global stiffness matrix K: {K.shape}")

    # Apply boundary conditions
    K_bc, F_bc = apply_boundary_conditions(K, F, fixed_dofs)

    # Solve using MTL5
    K_mtl5 = mtl5.matrix(K_bc)
    print(f"  MTL5 matrix: {K_mtl5}")
    print("  Solving Ku = F with mtl5.solve()...")

    u = mtl5.solve(K_bc, F_bc)

    # Verify: compute residual
    residual = mtl5.norm(K_bc @ u - F_bc)
    print(f"  Residual ||Ku - F||₂ = {residual:.2e}")
    print()

    # Display displacements
    print("Nodal Displacements:")
    print(f"  {'Node':>4s}  {'ux [mm]':>10s}  {'uy [mm]':>10s}")
    print(f"  {'----':>4s}  {'-------':>10s}  {'-------':>10s}")
    for i in range(n_nodes):
        ux = u[2 * i] * 1000  # convert to mm
        uy = u[2 * i + 1] * 1000
        print(f"  {i:4d}  {ux:10.4f}  {uy:10.4f}")
    print()

    # Compute member forces
    forces = compute_member_forces(nodes, elements, u, E, A, element_props)
    stresses = forces / A

    print("Member Forces and Stresses:")
    print(f"  {'Elem':>4s}  {'Nodes':>7s}  {'Force[kN]':>10s}  {'Stress[MPa]':>12s}  {'Type':>8s}")
    print(f"  {'----':>4s}  {'-----':>7s}  {'---------':>10s}  {'-----------':>12s}  {'----':>8s}")
    for idx in range(n_elements):
        i, j = int(elements[idx, 0]), int(elements[idx, 1])
        f_kn = forces[idx] / 1000
        s_mpa = stresses[idx] / 1e6
        typ = "Tension" if forces[idx] > 0 else "Compress."
        print(f"  {idx:4d}  {i:3d}-{j:<3d}  {f_kn:10.3f}  {s_mpa:12.2f}  {typ:>10s}")

    max_stress = np.max(np.abs(stresses))
    max_disp = np.max(np.abs(u)) * 1000
    print(f"\nMax |stress| = {max_stress / 1e6:.2f} MPa")
    print(f"Max |displacement| = {max_disp:.4f} mm")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            scale = 500  # displacement magnification

            for ax, title, show_deformed in [
                (ax1, "Undeformed + Loads", False),
                (ax2, f"Deformed (×{scale}) + Member Forces", True),
            ]:
                # Draw elements
                for idx in range(n_elements):
                    i, j = int(elements[idx, 0]), int(elements[idx, 1])
                    if show_deformed:
                        x = [
                            nodes[i, 0] + scale * u[2 * i],
                            nodes[j, 0] + scale * u[2 * j],
                        ]
                        y = [
                            nodes[i, 1] + scale * u[2 * i + 1],
                            nodes[j, 1] + scale * u[2 * j + 1],
                        ]
                        color = "blue" if forces[idx] > 0 else "red"
                        lw = 1 + 3 * abs(forces[idx]) / max(abs(forces))
                    else:
                        x = [nodes[i, 0], nodes[j, 0]]
                        y = [nodes[i, 1], nodes[j, 1]]
                        color = "gray"
                        lw = 1.5
                    ax.plot(x, y, color=color, linewidth=lw)

                # Draw nodes
                for i in range(n_nodes):
                    if show_deformed:
                        px = nodes[i, 0] + scale * u[2 * i]
                        py = nodes[i, 1] + scale * u[2 * i + 1]
                    else:
                        px, py = nodes[i]
                    ax.plot(px, py, "ko", markersize=6)
                    ax.annotate(
                        str(i), (px, py), textcoords="offset points", xytext=(5, 5), fontsize=8
                    )

                # Draw loads
                if not show_deformed:
                    for node in [1, 2, 3]:
                        ax.annotate(
                            "",
                            xy=(nodes[node, 0], nodes[node, 1]),
                            xytext=(nodes[node, 0], nodes[node, 1] + 1.5),
                            arrowprops=dict(arrowstyle="->", color="green", lw=2),
                        )
                        ax.text(
                            nodes[node, 0],
                            nodes[node, 1] + 1.7,
                            "10 kN",
                            ha="center",
                            fontsize=8,
                            color="green",
                        )

                ax.set_xlim(-1, 13)
                ax.set_ylim(-2, 6)
                ax.set_aspect("equal")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")

            # Legend for deformed plot
            ax2.plot([], [], "b-", linewidth=2, label="Tension")
            ax2.plot([], [], "r-", linewidth=2, label="Compression")
            ax2.legend()

            plt.suptitle("Warren Bridge Truss — MTL5 Analysis", fontsize=14)
            plt.tight_layout()
            plt.savefig("truss_analysis_2d.png", dpi=150)
            print("\nPlot saved to truss_analysis_2d.png")
            plt.show()
        except ImportError:
            print("\nmatplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
