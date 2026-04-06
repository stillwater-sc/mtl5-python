"""Tests for dense matrix operations and linear solve across precisions."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


class TestDenseMatrixF64:
    def test_create_from_numpy(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_f64)
        assert M.shape == (2, 2)
        assert M.dtype == "f64"

    def test_getitem(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = mtl5.matrix(a)
        assert M[0, 0] == 1.0
        assert M[1, 1] == 4.0

    def test_setitem(self):
        M = mtl5.matrix(np.zeros((2, 2)))
        M[0, 1] = 99.0
        assert M[0, 1] == 99.0

    def test_to_numpy(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = mtl5.matrix(a)
        result = M.to_numpy()
        assert result.dtype == np.float64
        npt.assert_array_equal(result, a)

    def test_repr(self):
        M = mtl5.matrix(np.eye(3))
        r = repr(M)
        assert "DenseMatrix_f64" in r
        assert "3" in r


class TestDenseMatrixF32:
    def test_create_from_numpy(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_f32)
        assert M.dtype == "f32"

    def test_to_numpy_preserves_dtype(self):
        a = np.eye(3, dtype=np.float32)
        M = mtl5.matrix(a)
        result = M.to_numpy()
        assert result.dtype == np.float32
        npt.assert_array_equal(result, a)


class TestDenseMatrixInt:
    def test_i32(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_i32)
        assert M.dtype == "i32"

    def test_i64(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)
        M = mtl5.matrix(a)
        assert isinstance(M, mtl5.DenseMatrix_i64)
        assert M.dtype == "i64"


class TestSolveF64:
    def test_simple_system(self):
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x = mtl5.solve(A, b)
        assert x.dtype == np.float64
        npt.assert_allclose(A @ x, b, atol=1e-14)

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

    def test_matches_numpy(self, rng):
        n = 20
        A = rng.standard_normal((n, n)).astype(np.float32)
        A += n * np.eye(n, dtype=np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        x_mtl5 = mtl5.solve(A, b)
        x_numpy = np.linalg.solve(A, b)
        npt.assert_allclose(x_mtl5, x_numpy, rtol=1e-4)


class TestSolveErrors:
    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            mtl5.solve(np.ones((2, 3)), np.ones(2))

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            mtl5.solve(np.eye(3), np.ones(4))
