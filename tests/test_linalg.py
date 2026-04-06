"""Tests for dense linear algebra operations: matmul, transpose, det, inv, lu, cholesky."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


class TestMatmul:
    def test_matmul_views(self, rng):
        A_np = rng.standard_normal((5, 4))
        B_np = rng.standard_normal((4, 6))
        A = mtl5.matrix(A_np)
        B = mtl5.matrix(B_np)
        C = mtl5.matmul(A, B)
        npt.assert_allclose(C.to_numpy(), A_np @ B_np, rtol=1e-12)

    def test_matmul_ndarray(self, rng):
        A = rng.standard_normal((5, 4))
        B = rng.standard_normal((4, 6))
        C = mtl5.matmul(A, B)
        npt.assert_allclose(C.to_numpy(), A @ B, rtol=1e-12)

    def test_matmul_operator(self, rng):
        A_np = rng.standard_normal((4, 4))
        B_np = rng.standard_normal((4, 4))
        A = mtl5.matrix(A_np)
        B = mtl5.matrix(B_np)
        C = A @ B
        npt.assert_allclose(C.to_numpy(), A_np @ B_np, rtol=1e-12)

    def test_matmul_dimension_mismatch(self):
        A = np.ones((3, 4))
        B = np.ones((5, 6))
        with pytest.raises(ValueError):
            mtl5.matmul(A, B)


class TestMatvec:
    def test_matvec(self, rng):
        A_np = rng.standard_normal((5, 4))
        x_np = rng.standard_normal(4)
        A = mtl5.matrix(A_np)
        x = mtl5.vector(x_np)
        y = mtl5.matvec(A, x)
        npt.assert_allclose(y.to_numpy(), A_np @ x_np, rtol=1e-12)

    def test_matrix_at_vector_operator(self, rng):
        A_np = rng.standard_normal((5, 4))
        x_np = rng.standard_normal(4)
        A = mtl5.matrix(A_np)
        x = mtl5.vector(x_np)
        y = A @ x
        npt.assert_allclose(y.to_numpy(), A_np @ x_np, rtol=1e-12)


class TestTranspose:
    def test_transpose_function(self, rng):
        A_np = rng.standard_normal((3, 5))
        A = mtl5.matrix(A_np)
        AT = mtl5.transpose(A)
        assert AT.shape == (5, 3)
        npt.assert_allclose(AT.to_numpy(), A_np.T, rtol=1e-14)

    def test_T_property(self, rng):
        A_np = rng.standard_normal((4, 6))
        A = mtl5.matrix(A_np)
        AT = A.T
        assert AT.shape == (6, 4)
        npt.assert_allclose(AT.to_numpy(), A_np.T, rtol=1e-14)


class TestDet:
    def test_det_identity(self):
        A = np.eye(5)
        assert mtl5.det(A) == pytest.approx(1.0)

    def test_det_2x2(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        # det = 1*4 - 2*3 = -2
        assert mtl5.det(A) == pytest.approx(-2.0, rel=1e-12)

    def test_det_matches_numpy(self, rng):
        for n in [3, 5, 10]:
            A = rng.standard_normal((n, n)) + n * np.eye(n)
            d_mtl5 = mtl5.det(A)
            d_numpy = np.linalg.det(A)
            npt.assert_allclose(d_mtl5, d_numpy, rtol=1e-10)

    def test_det_singular(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        assert mtl5.det(A) == pytest.approx(0.0, abs=1e-12)

    def test_det_non_square_raises(self):
        with pytest.raises(ValueError):
            mtl5.det(mtl5.matrix(np.ones((3, 4))))


class TestInv:
    def test_inv_identity(self):
        A = np.eye(4)
        Ainv = mtl5.inv(A)
        npt.assert_allclose(Ainv.to_numpy(), np.eye(4), atol=1e-14)

    def test_inv_round_trip(self, rng):
        n = 6
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        Ainv = mtl5.inv(A).to_numpy()
        npt.assert_allclose(A @ Ainv, np.eye(n), atol=1e-10)
        npt.assert_allclose(Ainv @ A, np.eye(n), atol=1e-10)

    def test_inv_matches_numpy(self, rng):
        n = 8
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        Ainv_mtl5 = mtl5.inv(A).to_numpy()
        Ainv_numpy = np.linalg.inv(A)
        npt.assert_allclose(Ainv_mtl5, Ainv_numpy, rtol=1e-10)


class TestLUFactor:
    def test_lu_object(self, rng):
        n = 6
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        b = rng.standard_normal(n)

        lu_factor = mtl5.lu(A)
        assert lu_factor.n == n

        x = lu_factor.solve(b)
        npt.assert_allclose(A @ x.to_numpy(), b, atol=1e-10)

    def test_lu_repeated_solve(self, rng):
        """Verify factorization can be reused for multiple RHS."""
        n = 5
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        lu_factor = mtl5.lu(A)

        for _ in range(3):
            b = rng.standard_normal(n)
            x = lu_factor.solve(b)
            npt.assert_allclose(A @ x.to_numpy(), b, atol=1e-10)

    def test_lu_singular_raises(self):
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(RuntimeError, match="singular"):
            mtl5.lu(A)


class TestCholeskyFactor:
    @staticmethod
    def make_spd(n: int, rng: np.random.Generator) -> np.ndarray:
        """Construct a symmetric positive-definite matrix."""
        L = np.tril(rng.standard_normal((n, n)))
        # Ensure positive diagonal
        np.fill_diagonal(L, np.abs(np.diag(L)) + n)
        return L @ L.T

    def test_cholesky_spd(self, rng):
        n = 6
        A = self.make_spd(n, rng)
        b = rng.standard_normal(n)

        chol = mtl5.cholesky(A)
        assert chol.n == n

        x = chol.solve(b)
        npt.assert_allclose(A @ x.to_numpy(), b, atol=1e-10)

    def test_cholesky_repeated_solve(self, rng):
        n = 5
        A = self.make_spd(n, rng)
        chol = mtl5.cholesky(A)

        for _ in range(3):
            b = rng.standard_normal(n)
            x = chol.solve(b)
            npt.assert_allclose(A @ x.to_numpy(), b, atol=1e-10)

    def test_cholesky_non_spd_raises(self):
        A = np.array([[1.0, 2.0], [2.0, 1.0]])  # not SPD (negative eigenvalue)
        with pytest.raises(RuntimeError, match="positive definite"):
            mtl5.cholesky(A)

    def test_cholesky_matches_numpy(self, rng):
        n = 10
        A = self.make_spd(n, rng)
        b = rng.standard_normal(n)
        chol = mtl5.cholesky(A)
        x_mtl5 = chol.solve(b).to_numpy()
        x_numpy = np.linalg.solve(A, b)
        npt.assert_allclose(x_mtl5, x_numpy, rtol=1e-10)


class TestBackendAPI:
    def test_backends(self):
        bs = mtl5.backends()
        assert "reference" in bs

    def test_get_backend(self):
        backend = mtl5.get_backend()
        assert isinstance(backend, str)

    def test_set_backend_valid(self):
        mtl5.set_backend("cpu")  # no-op, just shouldn't raise

    def test_set_backend_unknown(self):
        with pytest.raises(RuntimeError, match="Unknown backend"):
            mtl5.set_backend("nonexistent")

    def test_set_backend_kpu_not_available(self):
        with pytest.raises(RuntimeError, match="not yet available"):
            mtl5.set_backend("kpu")
