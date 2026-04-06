"""Tests for dense vector operations across multiple precisions."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


class TestDenseVectorF64:
    def test_create_from_numpy(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = mtl5.vector(a)
        assert isinstance(v, mtl5.DenseVector_f64)
        assert len(v) == 3
        assert v[0] == 1.0
        assert v.dtype == "f64"

    def test_setitem(self):
        v = mtl5.vector(np.zeros(3, dtype=np.float64))
        v[1] = 42.0
        assert v[1] == 42.0

    def test_index_error(self):
        v = mtl5.vector(np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            _ = v[5]

    def test_to_numpy(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        v = mtl5.vector(a)
        result = v.to_numpy()
        assert result.dtype == np.float64
        npt.assert_array_equal(result, a)

    def test_repr(self):
        v = mtl5.vector(np.array([1.0, 2.0]))
        r = repr(v)
        assert "DenseVector_f64" in r
        assert "size=2" in r


class TestDenseVectorF32:
    def test_create_from_numpy(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = mtl5.vector(a)
        assert isinstance(v, mtl5.DenseVector_f32)
        assert len(v) == 3
        assert v.dtype == "f32"

    def test_to_numpy_preserves_dtype(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        v = mtl5.vector(a)
        result = v.to_numpy()
        assert result.dtype == np.float32

    def test_repr(self):
        v = mtl5.vector(np.array([1.0], dtype=np.float32))
        assert "DenseVector_f32" in repr(v)


class TestDenseVectorInt:
    def test_i32(self):
        a = np.array([10, 20, 30], dtype=np.int32)
        v = mtl5.vector(a)
        assert isinstance(v, mtl5.DenseVector_i32)
        assert v.dtype == "i32"
        assert v[2] == 30

    def test_i64(self):
        a = np.array([10, 20, 30], dtype=np.int64)
        v = mtl5.vector(a)
        assert isinstance(v, mtl5.DenseVector_i64)
        assert v.dtype == "i64"


class TestNorm:
    def test_l2_norm_f64(self):
        a = np.array([3.0, 4.0], dtype=np.float64)
        assert mtl5.norm(a) == pytest.approx(5.0)

    def test_l2_norm_f32(self):
        a = np.array([3.0, 4.0], dtype=np.float32)
        assert mtl5.norm(a) == pytest.approx(5.0, rel=1e-6)

    def test_l1_norm(self):
        a = np.array([-3.0, 4.0])
        assert mtl5.norm(a, ord=1) == pytest.approx(7.0)

    def test_linf_norm(self):
        a = np.array([-3.0, 4.0, -1.0])
        assert mtl5.norm(a, ord=-1) == pytest.approx(4.0)

    def test_norm_matches_numpy(self, rng):
        a = rng.standard_normal(100)
        npt.assert_allclose(mtl5.norm(a), np.linalg.norm(a), rtol=1e-14)
        npt.assert_allclose(mtl5.norm(a, ord=1), np.linalg.norm(a, ord=1), rtol=1e-14)

    def test_norm_f32_matches_numpy(self, rng):
        a = rng.standard_normal(100).astype(np.float32)
        npt.assert_allclose(mtl5.norm(a), np.linalg.norm(a), rtol=1e-5)


class TestDot:
    def test_dot_f64(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        assert mtl5.dot(a, b) == pytest.approx(32.0)

    def test_dot_f32(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert mtl5.dot(a, b) == pytest.approx(32.0, rel=1e-6)

    def test_dot_i32(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        assert mtl5.dot(a, b) == pytest.approx(32.0)

    def test_dot_matches_numpy(self, rng):
        a = rng.standard_normal(200)
        b = rng.standard_normal(200)
        npt.assert_allclose(mtl5.dot(a, b), np.dot(a, b), rtol=1e-14)

    def test_dot_length_mismatch(self):
        with pytest.raises(ValueError):
            mtl5.dot(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
