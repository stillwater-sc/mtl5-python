"""Tests for dense vector operations across multiple precisions."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


class TestZeroCopyVectorF64:
    def test_zero_copy_from_numpy(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = mtl5.vector(a)
        assert isinstance(v, mtl5.DenseVector_f64)
        assert v.is_view
        assert len(v) == 3

    def test_shares_memory(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = mtl5.vector(a)
        # Modify via MTL5 view → visible in NumPy
        v[0] = 99.0
        assert a[0] == 99.0

    def test_numpy_to_mtl5_to_numpy_shares_memory(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        v = mtl5.vector(a)
        b = v.to_numpy()
        # Modify via returned NumPy array → visible in original
        b[2] = 42.0
        assert a[2] == 42.0

    def test_modify_numpy_visible_in_mtl5(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = mtl5.vector(a)
        a[1] = -7.0
        assert v[1] == -7.0

    def test_copy_is_independent(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = mtl5.vector_copy(a)
        assert not v.is_view
        v[0] = 99.0
        assert a[0] == 1.0  # original unchanged

    def test_dtype(self):
        v = mtl5.vector(np.array([1.0], dtype=np.float64))
        assert v.dtype == "f64"

    def test_device(self):
        v = mtl5.vector(np.array([1.0]))
        assert v.device == "cpu"

    def test_repr(self):
        v = mtl5.vector(np.array([1.0, 2.0]))
        r = repr(v)
        assert "DenseVector_f64" in r
        assert "view" in r
        assert "cpu" in r

    def test_index_error(self):
        v = mtl5.vector(np.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            _ = v[5]


class TestZeroCopyVectorF32:
    def test_zero_copy_f32(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = mtl5.vector(a)
        assert isinstance(v, mtl5.DenseVector_f32)
        assert v.is_view
        assert v.dtype == "f32"

    def test_shares_memory_f32(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v = mtl5.vector(a)
        v[0] = 99.0
        assert a[0] == np.float32(99.0)

    def test_to_numpy_preserves_dtype(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        v = mtl5.vector(a)
        result = v.to_numpy()
        assert result.dtype == np.float32


class TestZeroCopyVectorInt:
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


class TestNonContiguous:
    def test_non_contiguous_implicitly_copied(self):
        """nanobind implicitly copies non-contiguous arrays to make them contiguous.
        The resulting view is NOT shared with the original (safe behavior)."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sliced = a[::2]  # non-contiguous: [1, 3, 5]
        v = mtl5.vector(sliced)
        assert len(v) == 3
        assert v[0] == pytest.approx(1.0)
        assert v[1] == pytest.approx(3.0)
        # Mutation does NOT propagate back to original (implicit copy)
        v[0] = 99.0
        assert a[0] == 1.0


class TestNorm:
    def test_l2_norm_f64(self):
        a = np.array([3.0, 4.0], dtype=np.float64)
        assert mtl5.norm(a) == pytest.approx(5.0)

    def test_l2_norm_f32(self):
        a = np.array([3.0, 4.0], dtype=np.float32)
        assert mtl5.norm(a) == pytest.approx(5.0, rel=1e-6)

    def test_norm_on_view(self):
        a = np.array([3.0, 4.0])
        v = mtl5.vector(a)
        assert mtl5.norm(v) == pytest.approx(5.0)

    def test_l1_norm(self):
        a = np.array([-3.0, 4.0])
        assert mtl5.norm(a, ord=1) == pytest.approx(7.0)

    def test_linf_norm(self):
        a = np.array([-3.0, 4.0, -1.0])
        assert mtl5.norm(a, ord=-1) == pytest.approx(4.0)

    def test_norm_matches_numpy(self, rng):
        a = rng.standard_normal(100)
        npt.assert_allclose(mtl5.norm(a), np.linalg.norm(a), rtol=1e-14)

    def test_norm_f32_matches_numpy(self, rng):
        a = rng.standard_normal(100).astype(np.float32)
        npt.assert_allclose(mtl5.norm(a), np.linalg.norm(a), rtol=1e-5)


class TestDot:
    def test_dot_f64(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        assert mtl5.dot(a, b) == pytest.approx(32.0)

    def test_dot_on_views(self):
        a_np = np.array([1.0, 2.0, 3.0])
        b_np = np.array([4.0, 5.0, 6.0])
        a = mtl5.vector(a_np)
        b = mtl5.vector(b_np)
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


class TestDeviceAPI:
    def test_devices(self):
        devs = mtl5.devices()
        assert "cpu" in devs

    def test_to_cpu(self):
        a = np.array([1.0, 2.0, 3.0])
        v = mtl5.vector(a)
        v2 = v.to("cpu")
        assert v2.device == "cpu"
        assert not v2.is_view  # to() returns an owning copy

    def test_to_unknown_device_raises(self):
        v = mtl5.vector(np.array([1.0]))
        with pytest.raises(RuntimeError, match="not available"):
            v.to("gpu")
