"""Tests for Universal number types — fp8, fp16."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


class TestFp16Vector:
    def test_create_from_numpy(self):
        a = np.array([1.0, 2.0, 3.0])
        v = mtl5.vector_fp16(a)
        assert isinstance(v, mtl5.DenseVector_fp16)
        assert len(v) == 3
        assert v.dtype == "fp16"

    def test_values_preserved(self):
        a = np.array([1.0, -2.5, 0.125])
        v = mtl5.vector_fp16(a)
        assert v[0] == pytest.approx(1.0, rel=1e-3)
        assert v[1] == pytest.approx(-2.5, rel=1e-3)
        assert v[2] == pytest.approx(0.125, rel=1e-3)

    def test_setitem(self):
        v = mtl5.vector_fp16(np.zeros(3))
        v[1] = 42.0
        assert v[1] == pytest.approx(42.0, rel=1e-3)

    def test_to_numpy(self):
        a = np.array([1.0, 2.0, 3.0])
        v = mtl5.vector_fp16(a)
        result = v.to_numpy()
        assert result.dtype == np.float64
        npt.assert_allclose(result, a, rtol=1e-3)

    def test_to_list(self):
        v = mtl5.vector_fp16(np.array([1.0, 2.0]))
        lst = v.to_list()
        assert isinstance(lst, list)
        assert len(lst) == 2

    def test_repr(self):
        v = mtl5.vector_fp16(np.array([1.0, 2.0]))
        assert "DenseVector_fp16" in repr(v)

    def test_index_error(self):
        v = mtl5.vector_fp16(np.array([1.0]))
        with pytest.raises(IndexError):
            _ = v[5]


class TestFp16Matrix:
    def test_create_from_numpy(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = mtl5.matrix_fp16(a)
        assert isinstance(M, mtl5.DenseMatrix_fp16)
        assert M.shape == (2, 2)
        assert M.dtype == "fp16"

    def test_values_preserved(self):
        a = np.array([[1.0, 0.5], [0.25, 2.0]])
        M = mtl5.matrix_fp16(a)
        assert M[0, 0] == pytest.approx(1.0, rel=1e-3)
        assert M[0, 1] == pytest.approx(0.5, rel=1e-3)

    def test_to_numpy(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        M = mtl5.matrix_fp16(a)
        result = M.to_numpy()
        assert result.dtype == np.float64
        npt.assert_allclose(result, a, rtol=1e-3)


class TestFp16Solve:
    def test_simple_system(self):
        A = mtl5.matrix_fp16(np.array([[2.0, 1.0], [1.0, 3.0]]))
        b = mtl5.vector_fp16(np.array([5.0, 7.0]))
        x = mtl5.solve(A, b)
        assert isinstance(x, mtl5.DenseVector_fp16)
        result = np.array(x.to_list())
        # fp16 has limited precision — use relaxed tolerance
        npt.assert_allclose(result, [1.6, 1.8], rtol=1e-2)

    def test_identity(self):
        n = 3
        A = mtl5.matrix_fp16(np.eye(n))
        b = mtl5.vector_fp16(np.array([1.0, 2.0, 3.0]))
        x = mtl5.solve(A, b)
        npt.assert_allclose(x.to_list(), [1.0, 2.0, 3.0], rtol=1e-2)


class TestFp16NormDot:
    def test_norm(self):
        v = mtl5.vector_fp16(np.array([3.0, 4.0]))
        assert mtl5.norm(v) == pytest.approx(5.0, rel=1e-2)

    def test_norm_l1(self):
        v = mtl5.vector_fp16(np.array([-3.0, 4.0]))
        assert mtl5.norm(v, ord=1) == pytest.approx(7.0, rel=1e-2)

    def test_dot(self):
        a = mtl5.vector_fp16(np.array([1.0, 2.0, 3.0]))
        b = mtl5.vector_fp16(np.array([4.0, 5.0, 6.0]))
        assert mtl5.dot(a, b) == pytest.approx(32.0, rel=1e-2)

    def test_dot_length_mismatch(self):
        a = mtl5.vector_fp16(np.array([1.0, 2.0]))
        b = mtl5.vector_fp16(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            mtl5.dot(a, b)


class TestFp8Vector:
    def test_create_from_numpy(self):
        a = np.array([1.0, 2.0, 0.5])
        v = mtl5.vector_fp8(a)
        assert isinstance(v, mtl5.DenseVector_fp8)
        assert len(v) == 3
        assert v.dtype == "fp8"

    def test_values_representable(self):
        # fp8 has very limited range/precision — use values it can represent
        a = np.array([1.0, 0.5, 2.0])
        v = mtl5.vector_fp8(a)
        assert v[0] == pytest.approx(1.0, abs=0.2)
        assert v[1] == pytest.approx(0.5, abs=0.2)
        assert v[2] == pytest.approx(2.0, abs=0.5)

    def test_to_numpy(self):
        v = mtl5.vector_fp8(np.array([1.0, 2.0]))
        result = v.to_numpy()
        assert result.dtype == np.float64
        assert len(result) == 2

    def test_repr(self):
        v = mtl5.vector_fp8(np.array([1.0]))
        assert "DenseVector_fp8" in repr(v)


class TestFp8Matrix:
    def test_create_from_numpy(self):
        a = np.array([[1.0, 0.5], [2.0, 1.0]])
        M = mtl5.matrix_fp8(a)
        assert isinstance(M, mtl5.DenseMatrix_fp8)
        assert M.shape == (2, 2)
        assert M.dtype == "fp8"


class TestFp8NormDot:
    def test_norm(self):
        v = mtl5.vector_fp8(np.array([1.0, 0.0]))
        assert mtl5.norm(v) == pytest.approx(1.0, abs=0.3)

    def test_dot(self):
        a = mtl5.vector_fp8(np.array([1.0, 2.0]))
        b = mtl5.vector_fp8(np.array([1.0, 1.0]))
        # fp8 precision is coarse — wide tolerance
        assert mtl5.dot(a, b) == pytest.approx(3.0, abs=1.0)


class TestImportUniversalTypes:
    def test_typed_classes_exist(self):
        assert hasattr(mtl5, "DenseVector_fp8")
        assert hasattr(mtl5, "DenseVector_fp16")
        assert hasattr(mtl5, "DenseMatrix_fp8")
        assert hasattr(mtl5, "DenseMatrix_fp16")

    def test_factory_functions_exist(self):
        assert hasattr(mtl5, "vector_fp8")
        assert hasattr(mtl5, "vector_fp16")
        assert hasattr(mtl5, "matrix_fp8")
        assert hasattr(mtl5, "matrix_fp16")
