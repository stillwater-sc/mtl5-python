"""Tests for dense matrix operations and linear solve across precisions."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


class TestZeroCopyMatrixF64:
    def test_zero_copy_from_numpy(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_f64)
        assert M.is_view
        assert M.shape == (2, 2)

    def test_shares_memory(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        M = mtl5.matrix(a)
        M[0, 1] = 99.0
        assert a[0, 1] == 99.0

    def test_numpy_to_mtl5_to_numpy_shares_memory(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        M = mtl5.matrix(a)
        b = M.to_numpy()
        b[1, 0] = 42.0
        assert a[1, 0] == 42.0

    def test_copy_is_independent(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = mtl5.matrix_copy(a)
        assert not M.is_view
        M[0, 0] = 99.0
        assert a[0, 0] == 1.0

    def test_device(self):
        M = mtl5.matrix(np.eye(2))
        assert M.device == "cpu"

    def test_repr(self):
        M = mtl5.matrix(np.eye(3))
        r = repr(M)
        assert "DenseMatrix_f64" in r
        assert "view" in r


class TestZeroCopyMatrixF32:
    def test_zero_copy_f32(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_f32)
        assert M.dtype == "f32"

    def test_to_numpy_preserves_dtype(self):
        a = np.eye(3, dtype=np.float32)
        M = mtl5.matrix(a)
        result = M.to_numpy()
        assert result.dtype == np.float32


class TestMatrixInt:
    def test_i32(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_i32)

    def test_i64(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_i64)


class TestSolveF64:
    def test_simple_system_ndarray(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x = mtl5.solve(A, b)
        assert x.dtype == np.float64
        npt.assert_allclose(A @ x, b, atol=1e-14)

    def test_simple_system_views(self):
        A_np = np.array([[2.0, 1.0], [1.0, 3.0]])
        b_np = np.array([5.0, 7.0])
        A = mtl5.matrix(A_np)
        b = mtl5.vector(b_np)
        x = mtl5.solve(A, b)
        assert isinstance(x, mtl5.DenseVector_f64)

    def test_identity(self):
        n = 5
        A = np.eye(n)
        b = np.arange(1.0, n + 1.0)
        x = mtl5.solve(A, b)
        npt.assert_allclose(x, b, atol=1e-14)

    def test_matches_numpy(self, rng):
        n = 50
        A = rng.standard_normal((n, n))
        A += n * np.eye(n)
        b = rng.standard_normal(n)
        x_mtl5 = mtl5.solve(A, b)
        x_numpy = np.linalg.solve(A, b)
        npt.assert_allclose(x_mtl5, x_numpy, rtol=1e-10)


class TestSolveF32:
    def test_simple_system(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        b = np.array([5.0, 7.0], dtype=np.float32)
        x = mtl5.solve(A, b)
        assert x.dtype == np.float32
        npt.assert_allclose(A @ x, b, atol=1e-5)


class TestSolveErrors:
    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            mtl5.solve(np.ones((2, 3)), np.ones(2))

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            mtl5.solve(np.eye(3), np.ones(4))
