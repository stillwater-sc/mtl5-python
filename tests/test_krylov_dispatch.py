"""Krylov solver / preconditioner dispatch.

Ten solvers and eight preconditioners compose freely, because the
preconditioner is type-erased rather than being a template parameter of each
solver. The alternative — instantiating the cross product — would be 160 copies
of a full iterative solver.

Two classes here carry most of the weight:

  * `TestEveryPairingConverges` walks the grid against a known solution. A
    solver that returns `info = 0` and the wrong vector is the failure this
    whole file exists to catch, so every case checks the answer, not the status.
  * `TestBicgQmrNeedSymmetricPreconditioners` pins the one guard. Those two are
    the only solvers that apply Mᵀ, and MTL5 implements a preconditioner's
    adjoint as its forward solve — exact when M is symmetric, wrong otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
import mtl5.sparse as ms  # noqa: E402

# Non-symmetric solvers, and the ones needing symmetry.
SYMMETRIC_SOLVERS = ["cg", "minres"]
GENERAL_SOLVERS = ["bicgstab", "bicgstab_ell", "cgs", "gmres", "idr_s", "tfqmr"]
ADJOINT_SOLVERS = ["bicg", "qmr"]  # apply M^T
ALWAYS_SYMMETRIC_PC = ["identity", "diagonal", "ic0", "ildl"]
A_DEPENDENT_PC = ["ilu0", "ssor", "ilut"]


def laplacian(k=10, nonsym=False):
    """2-D Laplacian; `nonsym` skews the forward couplings so A != A^T."""
    n = k * k
    A = sp.lil_matrix((n, n))
    for i in range(k):
        for j in range(k):
            r = i * k + j
            A[r, r] = 4.0
            if i:
                A[r, r - k] = -1.0
            if i + 1 < k:
                A[r, r + k] = -0.5 if nonsym else -1.0
            if j:
                A[r, r - 1] = -1.0
            if j + 1 < k:
                A[r, r + 1] = -0.5 if nonsym else -1.0
    return A.tocsr()


def system(nonsym=False, k=10):
    A = laplacian(k, nonsym)
    rng = np.random.default_rng(0)
    x_true = rng.standard_normal(A.shape[0])
    return A, x_true, A @ x_true


def make_pc(name, A):
    if name == "block_diagonal":
        return ms.block_diagonal(A, 4)
    return getattr(ms, name)(A)


class TestEveryPairingConverges:
    """The grid. Each case asserts the answer, not just `info == 0`."""

    @pytest.mark.parametrize("solver", SYMMETRIC_SOLVERS)
    @pytest.mark.parametrize("pc", ALWAYS_SYMMETRIC_PC + A_DEPENDENT_PC)
    def test_symmetric_system(self, solver, pc):
        A, x_true, b = system()
        x, info = getattr(ms, solver)(A, b, M=make_pc(pc, A), rtol=1e-12, maxiter=5000)
        assert info == 0, f"{solver}+{pc} did not converge"
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("solver", GENERAL_SOLVERS)
    @pytest.mark.parametrize("pc", A_DEPENDENT_PC)
    def test_non_symmetric_system(self, solver, pc):
        A, x_true, b = system(nonsym=True)
        x, info = getattr(ms, solver)(A, b, M=make_pc(pc, A), rtol=1e-12, maxiter=5000)
        assert info == 0, f"{solver}+{pc} did not converge"
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("solver", [s for s in GENERAL_SOLVERS if s != "tfqmr"])
    @pytest.mark.parametrize("pc", ALWAYS_SYMMETRIC_PC[:2])
    def test_non_symmetric_system_weak_preconditioners(self, solver, pc):
        """tfqmr is excluded — see TestTfqmrStagnatesUnpreconditioned."""
        A, x_true, b = system(nonsym=True)
        x, info = getattr(ms, solver)(A, b, M=make_pc(pc, A), rtol=1e-12, maxiter=5000)
        assert info == 0, f"{solver}+{pc} did not converge"
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    def test_block_diagonal(self):
        A, x_true, b = system()
        x, info = ms.cg(A, b, M=ms.block_diagonal(A, 4), rtol=1e-12)
        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    def test_a_preconditioner_actually_helps(self):
        """If ILU(0) did not reduce the iteration count, the preconditioner
        would not be reaching the solver at all."""
        A, _, b = system(k=14)
        _, _, plain, _ = ms._krylov(
            "cg", ms.from_scipy(A), ms.identity(A), _vec(b), 1e-12, 5000, 30, 2, 4
        )
        _, _, precond, _ = ms._krylov(
            "cg", ms.from_scipy(A), ms.ilu0(A), _vec(b), 1e-12, 5000, 30, 2, 4
        )
        assert precond < plain, (precond, plain)


def _vec(b):
    import mtl5

    return mtl5.vector(np.ascontiguousarray(b, dtype=np.float64))


class TestTfqmrStagnatesUnpreconditioned:
    """TFQMR can stagnate, and does on this system with identity or diagonal —
    it plateaus rather than converging slowly, unchanged from 2 000 to 20 000
    iterations. That is a known characteristic of the method rather than a
    defect in the dispatch: the same solver converges in single-digit
    iterations once given a real preconditioner. Recorded here so the exclusion
    above is deliberate and visible rather than a quietly narrowed grid.
    """

    @pytest.mark.parametrize("pc", ["identity", "diagonal"])
    def test_it_stagnates(self, pc):
        A, _, b = system(nonsym=True)
        _, info = ms.tfqmr(A, b, M=make_pc(pc, A), rtol=1e-12, maxiter=5000)
        assert info != 0, "tfqmr unexpectedly converged — revisit the exclusion"

    @pytest.mark.parametrize("pc", A_DEPENDENT_PC)
    def test_a_preconditioner_rescues_it(self, pc):
        A, x_true, b = system(nonsym=True)
        x, info = ms.tfqmr(A, b, M=make_pc(pc, A), rtol=1e-12, maxiter=5000)
        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)


class TestBicgQmrNeedSymmetricPreconditioners:
    """bicg and qmr apply M^T, and MTL5's adjoint_solve is an alias for solve.
    That is exact for a symmetric M and wrong otherwise, so the non-symmetric
    pairings are refused rather than allowed to break down."""

    @pytest.mark.parametrize("solver", ADJOINT_SOLVERS)
    @pytest.mark.parametrize("pc", ALWAYS_SYMMETRIC_PC[:2])
    def test_symmetric_preconditioners_are_allowed(self, solver, pc):
        A, x_true, b = system(nonsym=True)
        x, info = getattr(ms, solver)(A, b, M=make_pc(pc, A), rtol=1e-12, maxiter=5000)
        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("solver", ADJOINT_SOLVERS)
    @pytest.mark.parametrize("pc", A_DEPENDENT_PC)
    def test_non_symmetric_preconditioners_are_refused(self, solver, pc):
        A, _, b = system(nonsym=True)
        with pytest.raises(ValueError, match="symmetric preconditioner"):
            getattr(ms, solver)(A, b, M=make_pc(pc, A))

    @pytest.mark.parametrize("solver", ADJOINT_SOLVERS)
    @pytest.mark.parametrize("pc", A_DEPENDENT_PC)
    def test_the_same_pairing_is_allowed_when_a_is_symmetric(self, solver, pc):
        """The guard is on the preconditioner's symmetry, not on its kind —
        ILU(0) of a symmetric matrix is symmetric, and works."""
        A, x_true, b = system(nonsym=False)
        M = make_pc(pc, A)
        assert M.is_symmetric
        x, info = getattr(ms, solver)(A, b, M=M, rtol=1e-12, maxiter=5000)
        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    def test_the_message_names_a_way_forward(self):
        A, _, b = system(nonsym=True)
        with pytest.raises(ValueError, match="bicgstab"):
            ms.bicg(A, b, M=ms.ilu0(A))


class TestSymmetryDetection:
    @pytest.mark.parametrize("pc", ALWAYS_SYMMETRIC_PC)
    def test_structurally_symmetric_kinds(self, pc):
        """These are symmetric whatever A is."""
        A = laplacian(6, nonsym=True)
        assert make_pc(pc, A).is_symmetric

    @pytest.mark.parametrize("pc", A_DEPENDENT_PC)
    def test_a_dependent_kinds_follow_the_matrix(self, pc):
        assert make_pc(pc, laplacian(6, nonsym=False)).is_symmetric
        assert not make_pc(pc, laplacian(6, nonsym=True)).is_symmetric

    def test_kind_and_n_are_reported(self):
        A = laplacian(6)
        M = ms.ilu0(A)
        assert M.kind == "ilu0"
        assert M.n == 36
        assert "ILU0_f64" in repr(M)


class TestPreconditionerFactories:
    @pytest.mark.parametrize("pc", ALWAYS_SYMMETRIC_PC + A_DEPENDENT_PC)
    def test_each_applies(self, pc):
        A = laplacian(6)
        M = make_pc(pc, A)
        r = np.ones(A.shape[0])
        out = M.solve(r).to_numpy()
        assert out.shape == r.shape
        assert np.all(np.isfinite(out))

    def test_float32(self):
        A = laplacian(6).astype(np.float32)
        assert "f32" in repr(ms.ilu0(A))

    def test_ssor_omega_is_validated(self):
        A = laplacian(6)
        with pytest.raises(ValueError, match="omega"):
            ms.ssor(A, omega=2.5)

    def test_block_size_is_validated(self):
        A = laplacian(6)
        with pytest.raises(ValueError, match="block_size"):
            ms.block_diagonal(A, 0)

    def test_ilut_parameters_change_the_factor(self):
        A = laplacian(8)
        loose = ms.ilut(A, fill=1, tau=1e-1)
        tight = ms.ilut(A, fill=40, tau=1e-12)
        r = np.ones(A.shape[0])
        assert not np.allclose(loose.solve(r).to_numpy(), tight.solve(r).to_numpy())


class TestSolverSurface:
    def test_solvers_and_preconditioners_are_listed(self):
        assert len(ms.solvers()) == 10
        assert len(ms.preconditioners()) == 8
        for name in ms.solvers():
            assert hasattr(ms, name), f"{name} not exposed"
            assert name in ms.__all__

    def test_iterative_solve_dispatches_by_name(self):
        A, x_true, b = system(nonsym=True)
        for name in GENERAL_SOLVERS:  # all converge with a real preconditioner
            x, info = ms.iterative_solve(A, b, solver=name, M=ms.ilu0(A), rtol=1e-12)
            assert info == 0
            np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    def test_unknown_solver(self):
        A, _, b = system()
        with pytest.raises(ValueError, match="unknown solver"):
            ms.iterative_solve(A, b, solver="jacobi")

    def test_m_defaults_to_identity(self):
        A, x_true, b = system()
        x, info = ms.cg(A, b, rtol=1e-12)
        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    def test_a_non_preconditioner_m_is_refused(self):
        A, _, b = system()
        with pytest.raises(TypeError, match="mtl5.sparse preconditioner"):
            ms.cg(A, b, M="ilu0")

    def test_solver_specific_parameters(self):
        A, x_true, b = system(nonsym=True)
        for kwargs in ({"restart": 10}, {"restart": 50}):
            x, info = ms.gmres(A, b, M=ms.ilu0(A), rtol=1e-12, **kwargs)
            assert info == 0
            np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)
        for ell in (1, 2, 4):
            x, info = ms.bicgstab_ell(A, b, M=ms.ilu0(A), rtol=1e-12, ell=ell)
            np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)
        for s in (1, 4, 8):
            x, info = ms.idr_s(A, b, M=ms.ilu0(A), rtol=1e-12, s=s)
            np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    def test_size_mismatch_is_refused(self):
        A, _, b = system()
        with pytest.raises(ValueError, match="does not match"):
            ms.cg(A, b, M=ms.ilu0(laplacian(4)))

    def test_callback_and_device_still_reserved(self):
        A, _, b = system()
        with pytest.raises(NotImplementedError):
            ms.cg(A, b, callback=lambda *_: None)
        with pytest.raises(NotImplementedError):
            ms.cg(A, b, device="kpu")


class TestBackwardCompatibility:
    """cg/gmres/bicgstab predate this and must keep working unchanged."""

    @pytest.mark.parametrize("solver", ["cg", "gmres", "bicgstab"])
    def test_positional_call_without_m(self, solver):
        A, x_true, b = system()
        x, info = getattr(ms, solver)(A, b)
        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-6)

    def test_scipy_matrix_input(self):
        A, x_true, b = system()
        x, info = ms.cg(sp.csr_matrix(A), b, M=ms.ilu0(A), rtol=1e-12)
        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-6, atol=1e-8)

    def test_preconditioners_still_refine(self):
        """mtl5.mixed.iterative_refine drives a factorization through
        .factor().solve(); ILU(0) and IC(0) moved into the new hierarchy, so
        this checks the overloads followed them."""
        import mtl5

        A, _, b = system()
        for M in (ms.ilu0(A), ms.ic0(A)):
            x, info = mtl5.mixed.iterative_refine(ms.from_scipy(A), M, b, max_iter=30)
            assert x.shape == b.shape
            assert "iters" in info
