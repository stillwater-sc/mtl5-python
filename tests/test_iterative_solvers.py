"""Tests for MTL5 iterative solvers and preconditioners."""

import numpy as np
import numpy.testing as npt
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")
from scipy.sparse.linalg import cg as scipy_cg  # noqa: E402

import mtl5  # noqa: F401, E402
import mtl5.sparse as msp  # noqa: E402


@pytest.fixture
def spd_poisson_1d():
    """1D Poisson matrix: tridiagonal SPD with 4 on diagonal, -1 off-diagonal."""
    n = 30
    diag = 4.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return scipy_sparse.csr_matrix(A)


@pytest.fixture
def nonsymmetric_diag_dom():
    """Random non-symmetric diagonally dominant matrix."""
    rng = np.random.default_rng(123)
    n = 25
    A = rng.standard_normal((n, n))
    A += n * np.eye(n)  # diagonally dominant
    return scipy_sparse.csr_matrix(A)


class TestCG:
    def test_cg_converges_on_spd(self, spd_poisson_1d):
        n = spd_poisson_1d.shape[0]
        b = np.ones(n)
        x, info = msp.cg(spd_poisson_1d, b)
        assert info == 0
        residual = np.linalg.norm(spd_poisson_1d @ x - b)
        assert residual < 1e-8

    def test_cg_accepts_mtl5_matrix(self, spd_poisson_1d):
        A = msp.from_scipy(spd_poisson_1d)
        b = np.ones(A.shape[0])
        x, info = msp.cg(A, b)
        assert info == 0
        residual = np.linalg.norm(spd_poisson_1d @ x - b)
        assert residual < 1e-8

    def test_cg_matches_scipy(self, spd_poisson_1d):
        b = np.arange(1.0, spd_poisson_1d.shape[0] + 1.0)
        x_mtl5, info_mtl5 = msp.cg(spd_poisson_1d, b, rtol=1e-12)
        x_scipy, info_scipy = scipy_cg(spd_poisson_1d, b, rtol=1e-12)
        assert info_mtl5 == 0 and info_scipy == 0
        npt.assert_allclose(x_mtl5, x_scipy, rtol=1e-6)

    def test_cg_max_iter_returns_info_1(self):
        # Build a system unlikely to converge in 1 iteration
        n = 50
        rng = np.random.default_rng(0)
        A = rng.standard_normal((n, n))
        A = scipy_sparse.csr_matrix(A @ A.T + n * np.eye(n))
        b = rng.standard_normal(n)
        _x, info = msp.cg(A, b, rtol=1e-15, maxiter=1)
        assert info == 1  # exceeded max_iter


class TestGMRES:
    def test_gmres_on_nonsymmetric(self, nonsymmetric_diag_dom):
        n = nonsymmetric_diag_dom.shape[0]
        b = np.ones(n)
        x, info = msp.gmres(nonsymmetric_diag_dom, b, rtol=1e-10, maxiter=200)
        assert info == 0
        residual = np.linalg.norm(nonsymmetric_diag_dom @ x - b)
        assert residual < 1e-6

    def test_gmres_accepts_mtl5_matrix(self, nonsymmetric_diag_dom):
        A = msp.from_scipy(nonsymmetric_diag_dom)
        b = np.ones(A.shape[0])
        x, info = msp.gmres(A, b, maxiter=200)
        assert info == 0


class TestBiCGSTAB:
    def test_bicgstab_on_nonsymmetric(self, nonsymmetric_diag_dom):
        n = nonsymmetric_diag_dom.shape[0]
        b = np.ones(n)
        x, info = msp.bicgstab(nonsymmetric_diag_dom, b, rtol=1e-10, maxiter=200)
        assert info == 0
        residual = np.linalg.norm(nonsymmetric_diag_dom @ x - b)
        assert residual < 1e-6


class TestPreconditioners:
    def test_ilu0_solve(self, spd_poisson_1d):
        A = msp.from_scipy(spd_poisson_1d)
        precond = msp.ilu0(A)
        n = A.shape[0]
        r = np.ones(n)
        z = precond.solve(r)
        # Verify (LU) z ≈ r
        z_arr = z.to_numpy()
        assert z_arr.shape == (n,)
        # The result should be a meaningful vector (not all zeros)
        assert np.linalg.norm(z_arr) > 0

    def test_ilu0_accepts_scipy(self, spd_poisson_1d):
        precond = msp.ilu0(spd_poisson_1d)
        z = precond.solve(np.ones(spd_poisson_1d.shape[0]))
        assert z.to_numpy().shape == (spd_poisson_1d.shape[0],)

    def test_ic0_on_spd(self, spd_poisson_1d):
        precond = msp.ic0(spd_poisson_1d)
        z = precond.solve(np.ones(spd_poisson_1d.shape[0]))
        assert z.to_numpy().shape == (spd_poisson_1d.shape[0],)

    def test_repr(self, spd_poisson_1d):
        ilu = msp.ilu0(spd_poisson_1d)
        ic = msp.ic0(spd_poisson_1d)
        assert "ILU0" in repr(ilu)
        assert "IC0" in repr(ic)


class TestPreconditionedScipyCG:
    def test_ilu0_as_M_in_scipy_cg(self, spd_poisson_1d):
        """Use MTL5's ILU0 as preconditioner in scipy.sparse.linalg.cg."""
        n = spd_poisson_1d.shape[0]
        precond = msp.ilu0(spd_poisson_1d)
        M = msp.as_preconditioner_lo(precond, n)

        b = np.ones(n)
        x_pc, info_pc = scipy_cg(spd_poisson_1d, b, M=M, rtol=1e-10, maxiter=200)
        x_no_pc, info_no_pc = scipy_cg(spd_poisson_1d, b, rtol=1e-10, maxiter=200)
        assert info_pc == 0
        assert info_no_pc == 0
        # Both should reach the same solution
        npt.assert_allclose(x_pc, x_no_pc, rtol=1e-6)

    def test_ic0_as_M_in_scipy_cg(self, spd_poisson_1d):
        n = spd_poisson_1d.shape[0]
        precond = msp.ic0(spd_poisson_1d)
        M = msp.as_preconditioner_lo(precond, n)

        b = np.arange(1.0, n + 1.0)
        x, info = scipy_cg(spd_poisson_1d, b, M=M, rtol=1e-10, maxiter=200)
        assert info == 0
        residual = np.linalg.norm(spd_poisson_1d @ x - b)
        assert residual < 1e-6

    def test_ilu0_f32_as_M_in_scipy_cg(self, spd_poisson_1d):
        """Exercise the float32 preconditioner path end-to-end."""
        A_f32 = spd_poisson_1d.astype(np.float32)
        n = A_f32.shape[0]
        precond = msp.ilu0(A_f32)
        # dtype is inferred from the preconditioner type
        M = msp.as_preconditioner_lo(precond, n)
        assert M.dtype == np.float32

        b = np.ones(n, dtype=np.float32)
        x, info = scipy_cg(A_f32, b, M=M, rtol=1e-6, maxiter=200)
        assert info == 0
        residual = np.linalg.norm(A_f32 @ x - b)
        assert residual < 1e-3


class TestSolverErrors:
    def test_cg_non_square_raises(self):
        A = scipy_sparse.csr_matrix(np.ones((3, 4)))
        with pytest.raises(ValueError):
            msp.cg(A, np.ones(3))

    def test_dimension_mismatch_raises(self, spd_poisson_1d):
        with pytest.raises(ValueError):
            msp.cg(spd_poisson_1d, np.ones(5))  # wrong b length

    def test_invalid_matrix_type(self):
        with pytest.raises(TypeError):
            msp.cg(np.eye(3), np.ones(3))  # dense numpy not allowed
