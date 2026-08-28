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


class TestNarrowIntegerStorage:
    """int8 / int16 / uint8 — the operand types of MTL5's widening integer
    kernels. Storage only for now: see TestNarrowIntegerHasNoArithmetic."""

    NARROW = [(np.int8, "i8"), (np.int16, "i16"), (np.uint8, "u8")]

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_dispatches_to_its_own_type(self, dt, suffix):
        """The bug this closes: with no i8/i16/u8 overload registered, nanobind's
        converting second pass handed these to the float overload (registered
        first), so `vector(int8 array)` returned a DenseVector_f32 that reported
        is_view=True while being a view of the converted temporary."""
        v = mtl5.vector(np.array([10, 20, 30], dtype=dt))
        assert isinstance(v, getattr(mtl5, f"DenseVector_{suffix}"))
        assert v.dtype == suffix
        assert v[2] == 30

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_is_actually_zero_copy(self, dt, suffix):
        """`is_view` alone would not have caught the old behaviour — the view of
        the converted temporary reported True. Writing through the NumPy array
        is what distinguishes an alias from a copy."""
        a = np.array([10, 20, 30], dtype=dt)
        v = mtl5.vector(a)
        assert v.is_view
        a[0] = 99
        assert v[0] == 99, "the vector must alias the NumPy buffer, not a copy"

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_matrix_and_round_trip(self, dt, suffix):
        base = np.arange(6, dtype=dt).reshape(2, 3)
        m = mtl5.matrix(base)
        assert isinstance(m, getattr(mtl5, f"DenseMatrix_{suffix}"))
        assert m.dtype == suffix
        np.testing.assert_array_equal(m.to_numpy(), base)
        assert m.to_numpy().dtype == np.dtype(dt)


class TestNarrowIntegerHasNoArithmetic:
    """`dot`, `norm` and `__matmul__` are deliberately not registered for
    i8/i16/u8.

    All three default to accumulating in the ELEMENT type, which on 8- and
    16-bit operands overflows almost immediately. One input covers all of them,
    since each sums the same six products — vectors of six 100s, and a 6x6 of
    100s. Exact dot and exact A@A element are both 60000; exact two_norm is
    244.9490. Measured:

        dot       i8 -> 96       i16 -> -5536    u8 -> 96
        two_norm  i8 -> 9.798    i16 -> nan      u8 -> 9.798
        A @ A     i8 -> 96       i16 -> -5536    u8 -> 96

    two_norm is sqrt(dot), which is what makes the i16 nan legible: the sum
    wrapped negative, and sqrt(-5536) has no answer to give.

    These are MTL5's documented two's-complement wrapping and become correct
    once an int32 accumulator is supplied. i16 has real headroom — a 2x2 of
    100s gives the exact 20000 — so its failure needs a longer k rather than
    being immediate, but that is a difference of degree: k=4 already wraps it,
    and nothing in the API tells a caller where the edge is.

    A TypeError is the honest answer until the accumulator exists. If this test
    starts failing because someone registered the generic overloads, the fix is
    the accumulator, not deleting the test.
    """

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.uint8], ids=["i8", "i16", "u8"])
    def test_dot_is_not_offered(self, dt):
        v = mtl5.vector(np.full(6, 100, dtype=dt))
        with pytest.raises(TypeError):
            mtl5.dot(v, v)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.uint8], ids=["i8", "i16", "u8"])
    def test_norm_is_not_offered(self, dt):
        v = mtl5.vector(np.full(6, 100, dtype=dt))
        with pytest.raises(TypeError):
            mtl5.norm(v, 2)

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.uint8], ids=["i8", "i16", "u8"])
    def test_matmul_is_not_offered(self, dt):
        """`register_native_matrix` registers `__matmul__` on the class itself,
        so splitting norm/dot out of `register_native` was not enough — the
        narrow types still exposed a matrix product that accumulates in the
        element type (a 6x6 of 100s gave 96 where the answer is 60000)."""
        m = mtl5.matrix(np.full((6, 6), 100, dtype=dt))
        with pytest.raises(TypeError):
            m @ m

    @pytest.mark.parametrize("dt", [np.int8, np.int16, np.uint8], ids=["i8", "i16", "u8"])
    def test_matvec_is_not_offered(self, dt):
        m = mtl5.matrix(np.full((6, 6), 100, dtype=dt))
        v = mtl5.vector(np.full(6, 100, dtype=dt))
        with pytest.raises(TypeError):
            m @ v

    def test_existing_types_keep_their_matmul(self):
        """The split must not have removed arithmetic from f32/f64/i32/i64."""
        for dt in (np.float64, np.float32, np.int32, np.int64):
            a = np.eye(2, dtype=dt) * 3
            m = mtl5.matrix(a)
            np.testing.assert_array_equal((m @ m).to_numpy(), a @ a)


class TestNonContiguous:
    """A non-contiguous array is rejected rather than silently repacked.

    This used to be accepted, and what it did was worse than the old docstring
    said. nanobind's converting pass repacks layout AND dtype together, and it
    takes the first overload that converts — which is float32. So a float64
    slice came back as a *float32* vector, losing precision, while reporting
    `is_view=True` and not aliasing anything:

        a = np.array([0.1, 1.0, 0.2, 2.0, 0.3, 3.0])
        mtl5.vector(a[::2])[0]  ->  0.10000000149011612   (exact f64: 0.1)

    The values in the old test (1.0, 3.0, 5.0) are exact in float32, which is
    why it never noticed. `.noconvert()` on the factories makes this a
    TypeError; `np.ascontiguousarray` is the deliberate way to ask for the copy.
    """

    def test_non_contiguous_is_rejected(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with pytest.raises(TypeError):
            mtl5.vector(a[::2])

    def test_ascontiguousarray_is_the_way_through(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = mtl5.vector(np.ascontiguousarray(a[::2]))
        assert isinstance(v, mtl5.DenseVector_f64), "and it keeps float64"
        assert len(v) == 3
        assert v[1] == pytest.approx(3.0)
        # An explicit copy, so mutation does not reach the source -- same
        # end behaviour as before, but now the caller asked for it.
        v[0] = 99.0
        assert a[0] == 1.0

    def test_a_float64_slice_no_longer_becomes_float32(self):
        """The regression this closes: silent precision loss on ordinary
        slicing. 0.1 is not representable in float32."""
        a = np.array([0.1, 1.0, 0.2, 2.0, 0.3, 3.0], dtype=np.float64)
        with pytest.raises(TypeError):
            mtl5.vector(a[::2])
        v = mtl5.vector(np.ascontiguousarray(a[::2]))
        assert v[0] == 0.1, "float64 must survive the round trip exactly"


class TestFactoryRejectsUnregisteredDtypes:
    """`.noconvert()` also stops unregistered dtypes being silently converted.

    Before, `mtl5.vector(np.arange(4, dtype=np.float16))` returned a
    DenseVector_f32 reporting is_view=True — a view of the converted temporary,
    so writes through the NumPy array were invisible.
    """

    @pytest.mark.parametrize(
        "dt",
        [np.float16, np.uint16, np.uint32, np.uint64],
        ids=["f16", "u16", "u32", "u64"],
    )
    def test_unregistered_dtype_raises(self, dt):
        with pytest.raises(TypeError):
            mtl5.vector(np.arange(4, dtype=dt))

    @pytest.mark.parametrize(
        "dt",
        [
            np.float32,
            np.float64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.complex64,
            np.complex128,
        ],
    )
    def test_registered_dtypes_still_dispatch_exactly(self, dt):
        v = mtl5.vector(np.arange(4, dtype=dt))
        assert v.to_numpy().dtype == np.dtype(dt)


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
