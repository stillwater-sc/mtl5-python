"""Tests for dense vector operations."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


class TestDenseVector:
    def test_create_from_numpy(self):
        a = np.array([1.0, 2.0, 3.0])
        v = mtl5.vector(a)
        assert len(v) == 3
        assert v[0] == 1.0
        assert v[1] == 2.0
        assert v[2] == 3.0

    def test_setitem(self):
        v = mtl5.vector(np.array([0.0, 0.0, 0.0]))
        v[1] = 42.0
        assert v[1] == 42.0

    def test_index_error(self):
        v = mtl5.vector(np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            _ = v[5]

    def test_to_numpy(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        v = mtl5.vector(a)
        result = v.to_numpy()
        npt.assert_array_equal(result, a)

    def test_repr(self):
        v = mtl5.vector(np.array([1.0, 2.0]))
        r = repr(v)
        assert "DenseVector" in r
        assert "size=2" in r


class TestNorm:
    def test_l2_norm(self):
        a = np.array([3.0, 4.0])
        assert mtl5.norm(a) == pytest.approx(5.0)

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


class TestDot:
    def test_dot_product(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert mtl5.dot(a, b) == pytest.approx(32.0)

    def test_dot_matches_numpy(self, rng):
        a = rng.standard_normal(200)
        b = rng.standard_normal(200)
        npt.assert_allclose(mtl5.dot(a, b), np.dot(a, b), rtol=1e-14)

    def test_dot_length_mismatch(self):
        with pytest.raises(ValueError):
            mtl5.dot(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
