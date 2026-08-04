"""The N-dimensional array layer (mtl/array).

Nearly every assertion here is written against NumPy rather than against a
hand-computed constant, because the whole point of the layer is to agree with
NumPy about shapes, strides and element order. Two areas get extra attention:

  * **reshape of a non-C-contiguous array.** MTL5's own `ndarray::reshape` used
    to return raw memory order there while reporting success (mtl5#359, since
    fixed upstream, where it now throws). The binding gives NumPy's semantics
    instead — a view when it can alias, a copy otherwise, never an error — and
    these tests are what hold that line.
  * **views versus copies.** A slice or transpose must alias its source, and a
    reshape that cannot alias must not.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

A = mtl5.array

DTYPES = [(np.float32, "f32"), (np.float64, "f64")]


@pytest.fixture(params=DTYPES, ids=[d[1] for d in DTYPES])
def dtype(request):
    return request.param


def arange(*shape, dt=np.float64):
    """A C-contiguous array with distinct values, so order mistakes show."""
    return np.arange(np.prod(shape), dtype=dt).reshape(*shape)


class TestConstruction:
    def test_asarray_is_a_zero_copy_view(self, dtype):
        np_dt, suffix = dtype
        a = arange(2, 3, dt=np_dt)
        x = A.asarray(a)
        assert x.ndim == 2
        assert x.shape == [2, 3]
        assert x.size == 6
        assert x.dtype == suffix
        assert x.is_view
        x[[1, 2]] = 99.0
        assert a[1, 2] == 99.0, "writing through the view must reach NumPy's buffer"

    def test_ranks_1_through_4(self):
        for shape in [(5,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            x = A.asarray(arange(*shape))
            assert x.ndim == len(shape)
            assert x.shape == list(shape)
            np.testing.assert_array_equal(x.to_numpy(), arange(*shape))

    def test_rank_5_is_refused(self):
        """Rank is a template parameter, so 5+ has no instantiation. It must
        raise rather than silently flatten."""
        with pytest.raises(TypeError):
            A.asarray(np.zeros((2, 2, 2, 2, 2)))

    def test_zeros(self):
        z = A.zeros([2, 3])
        assert z.shape == [2, 3]
        assert not z.is_view
        np.testing.assert_array_equal(z.to_numpy(), np.zeros((2, 3)))

    def test_zeros_f32(self):
        assert A.zeros([4], dtype="f32").dtype == "f32"

    def test_owning_arrays_report_no_owner(self):
        """NDArrayView's owner defaults to a NULL handle rather than
        nb::none(), because copy/arithmetic/axis-reductions build one inside a
        nogil scope and increfing Py_None there is a data race. This pins the
        behaviour that default stands for."""
        x = A.asarray(arange(2, 3))
        assert not x.copy().is_view
        assert not (x + x).is_view
        assert not x.sum_axis(0).is_view

    def test_zeros_rejects_other_dtypes(self):
        with pytest.raises(ValueError, match="f32.*f64"):
            A.zeros([4], dtype="posit16")

    def test_asarray_accepts_a_strided_numpy_view(self):
        """Any strided layout comes through without a copy."""
        a = arange(4, 6)
        strided = a[::2, ::3]
        x = A.asarray(strided)
        assert x.shape == [2, 2]
        np.testing.assert_array_equal(x.to_numpy(), strided)

    def test_negative_strides_are_refused_clearly(self):
        """MTL5 strides are unsigned, so a reversed NumPy view cannot be
        aliased. That has to say so rather than produce garbage."""
        with pytest.raises(ValueError, match="negative strides"):
            A.asarray(arange(4)[::-1])


class TestAsarrayDoesNotConvert:
    """`asarray` is documented zero-copy, so it must require an exact dtype
    match. A converting overload would return a view of a temporary — neither
    zero-copy nor the dtype asked for. Before `.noconvert()`, an int64 array
    silently became float32 while still reporting `is_view = True`."""

    def test_integer_input_is_refused(self):
        with pytest.raises(TypeError):
            A.asarray(np.arange(6, dtype=np.int64))

    def test_exact_dtypes_are_preserved(self):
        assert A.asarray(np.arange(4, dtype=np.float32)).dtype == "f32"
        assert A.asarray(np.arange(4, dtype=np.float64)).dtype == "f64"

    def test_float32_is_not_widened_to_float64(self):
        """Registration order used to decide this; now nothing converts."""
        a = np.arange(4, dtype=np.float32)
        x = A.asarray(a)
        assert x.dtype == "f32"
        x[[0]] = 7.0
        assert a[0] == 7.0, "still the same buffer"

    def test_read_only_input_is_refused(self):
        """The view aliases the buffer, so it has to be writable."""
        a = np.arange(6.0)
        a.flags.writeable = False
        with pytest.raises(TypeError):
            A.asarray(a)


class TestElementAccess:
    def test_get_and_set(self):
        a = arange(2, 3)
        x = A.asarray(a)
        assert x[1, 2] == a[1, 2]
        x[[0, 0]] = 42.0
        assert a[0, 0] == 42.0

    def test_out_of_range(self):
        x = A.asarray(arange(2, 3))
        with pytest.raises(IndexError):
            x[5, 0]

    def test_too_many_indices(self):
        x = A.asarray(arange(2, 3))
        with pytest.raises(IndexError, match="too many indices"):
            x[0, 0, 0]

    def test_len_is_the_first_extent(self):
        assert len(A.asarray(arange(7, 2))) == 7

    def test_setitem_accepts_negative_indices(self):
        """__getitem__ normalises them, so __setitem__ must agree — it used to
        reject them with a TypeError from the cast."""
        a = arange(2, 3)
        x = A.asarray(a)
        x[[-1, 0]] = 9.0
        assert a[1, 0] == 9.0
        assert x[-1, 0] == 9.0, "read and write must agree on what -1 means"

    def test_setitem_out_of_range_negative(self):
        x = A.asarray(arange(2, 3))
        with pytest.raises(IndexError, match="axis 0"):
            x[[-9, 0]] = 1.0


class TestTranspose:
    def test_matches_numpy(self):
        a = arange(2, 3)
        t = A.asarray(a).T
        assert t.shape == [3, 2]
        np.testing.assert_array_equal(t.to_numpy(), a.T)

    def test_strides_are_reversed_and_it_is_a_view(self):
        x = A.asarray(arange(2, 3))
        t = x.T
        assert t.strides == list(reversed(x.strides))
        assert t.is_view
        assert not t.is_c_contiguous
        assert t.is_f_contiguous, "the transpose of a C array is F-contiguous"

    def test_writes_alias_the_source(self):
        a = arange(2, 3)
        A.asarray(a).T[[2, 0]] = 77.0
        assert a[0, 2] == 77.0

    def test_transpose_method_and_property_agree(self):
        x = A.asarray(arange(3, 4))
        np.testing.assert_array_equal(x.transpose().to_numpy(), x.T.to_numpy())

    def test_higher_rank(self):
        a = arange(2, 3, 4)
        np.testing.assert_array_equal(A.asarray(a).T.to_numpy(), a.T)


class TestReshapeMatchesNumpy:
    """Reshape follows NumPy: a view when the layout allows aliasing, a copy
    when it does not, and never an error. Upstream once returned raw memory
    order here (mtl5#359) and now throws; neither is NumPy's contract, so the
    binding computes it. These tests hold that line.
    """

    def test_contiguous_reshape_is_a_view(self):
        a = arange(2, 3)
        r = A.asarray(a).reshape([6])
        np.testing.assert_array_equal(r.to_numpy(), a.reshape(6))
        r[[0]] = 55.0
        assert a[0, 0] == 55.0, "a contiguous reshape should alias"

    def test_transposed_reshape_matches_numpy(self):
        """Raw memory order would give [0,1,2,3,4,5]; throwing, as upstream now
        does, would not be NumPy's contract either."""
        a = arange(2, 3)
        got = A.asarray(a).T.reshape([6]).to_numpy()
        np.testing.assert_array_equal(got, a.T.reshape(6))
        assert list(got) == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]

    def test_transposed_reshape_is_a_copy_and_does_not_alias(self):
        a = arange(2, 3)
        r = A.asarray(a).T.reshape([6])
        assert not r.is_view
        r[[0]] = 123.0
        assert a[0, 0] == 0.0, "a packed reshape must not write back"

    def test_sliced_reshape_matches_numpy(self):
        a = arange(4, 4)
        s = a[:, 1:3]
        np.testing.assert_array_equal(A.asarray(a)[:, 1:3].reshape([8]).to_numpy(), s.reshape(8))

    def test_reshape_between_ranks(self):
        a = arange(2, 3, 4)
        x = A.asarray(a)
        np.testing.assert_array_equal(x.reshape([6, 4]).to_numpy(), a.reshape(6, 4))
        np.testing.assert_array_equal(x.reshape([24]).to_numpy(), a.reshape(24))
        np.testing.assert_array_equal(x.reshape([2, 2, 2, 3]).to_numpy(), a.reshape(2, 2, 2, 3))

    def test_wrong_total_size_raises(self):
        x = A.asarray(arange(2, 3))
        with pytest.raises(ValueError, match="cannot reshape"):
            x.reshape([4])

    def test_rank_5_raises(self):
        x = A.asarray(arange(2, 3))
        with pytest.raises(ValueError, match="ranks 1 through 4"):
            x.reshape([1, 1, 1, 2, 3])


class TestRavelAndFlatten:
    """NumPy's two behaviours, kept distinct: `ravel` returns a view when it
    can, `flatten` always copies. Both walk in logical order — which upstream's
    `flatten` did not, until mtl5#359 was fixed.
    """

    def test_ravel_of_a_contiguous_array_is_a_view(self):
        a = arange(2, 3)
        r = A.asarray(a).ravel()
        assert r.is_view
        np.testing.assert_array_equal(r.to_numpy(), a.ravel())
        r[[0]] = 21.0
        assert a[0, 0] == 21.0

    def test_ravel_of_a_transpose_matches_numpy(self):
        """Memory order would give [0,1,2,3,4,5]."""
        a = arange(2, 3)
        got = A.asarray(a).T.ravel().to_numpy()
        np.testing.assert_array_equal(got, a.T.ravel())
        assert list(got) == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]

    def test_ravel_of_a_transpose_is_a_copy(self):
        a = arange(2, 3)
        r = A.asarray(a).T.ravel()
        assert not r.is_view
        r[[0]] = 99.0
        assert a[0, 0] == 0.0

    def test_flatten_always_copies(self):
        a = arange(2, 3)
        f = A.asarray(a).flatten()
        assert not f.is_view, "NumPy's flatten always copies"
        f[[0]] = 99.0
        assert a[0, 0] == 0.0

    def test_flatten_of_a_transpose_matches_numpy(self):
        a = arange(2, 3)
        np.testing.assert_array_equal(A.asarray(a).T.flatten().to_numpy(), a.T.flatten())

    def test_over_a_slice(self):
        a = arange(4, 4)
        np.testing.assert_array_equal(A.asarray(a)[:, 1:3].ravel().to_numpy(), a[:, 1:3].ravel())

    def test_higher_rank(self):
        a = arange(2, 3, 4)
        np.testing.assert_array_equal(A.asarray(a).T.ravel().to_numpy(), a.T.ravel())


class TestSlicing:
    def test_integer_index_drops_the_axis(self):
        a = arange(2, 3)
        r = A.asarray(a)[1]
        assert r.ndim == 1
        np.testing.assert_array_equal(r.to_numpy(), a[1])

    def test_slice_keeps_the_axis(self):
        a = arange(4, 3)
        r = A.asarray(a)[1:3]
        assert r.shape == [2, 3]
        np.testing.assert_array_equal(r.to_numpy(), a[1:3])

    def test_column(self):
        a = arange(2, 3)
        np.testing.assert_array_equal(A.asarray(a)[:, 2].to_numpy(), a[:, 2])

    def test_mixed_int_and_slice(self):
        a = arange(2, 3, 4)
        x = A.asarray(a)
        np.testing.assert_array_equal(x[1, :, 2].to_numpy(), a[1, :, 2])
        np.testing.assert_array_equal(x[:, 1:3, :].to_numpy(), a[:, 1:3, :])

    def test_all_integers_gives_a_scalar(self):
        a = arange(2, 3)
        assert A.asarray(a)[1, 2] == a[1, 2]

    def test_step(self):
        a = arange(6, 4)
        np.testing.assert_array_equal(A.asarray(a)[::2].to_numpy(), a[::2])
        np.testing.assert_array_equal(A.asarray(a)[:, ::3].to_numpy(), a[:, ::3])

    def test_trailing_axes_are_implicit(self):
        a = arange(2, 3, 4)
        np.testing.assert_array_equal(A.asarray(a)[1].to_numpy(), a[1])

    def test_negative_index(self):
        a = arange(4, 3)
        np.testing.assert_array_equal(A.asarray(a)[-1].to_numpy(), a[-1])

    def test_slice_is_a_view(self):
        a = arange(3, 3)
        A.asarray(a)[1][[0]] = 88.0
        assert a[1, 0] == 88.0

    def test_negative_step_is_refused_clearly(self):
        x = A.asarray(arange(4, 3))
        with pytest.raises(ValueError, match="negative or zero slice steps"):
            x[::-1]

    def test_out_of_range_integer(self):
        x = A.asarray(arange(2, 3))
        with pytest.raises(IndexError, match="axis 0"):
            x[9]


class TestReductions:
    def test_whole_array(self, dtype):
        np_dt, _ = dtype
        a = arange(2, 3, dt=np_dt) + 1  # avoid prod == 0
        x = A.asarray(np.ascontiguousarray(a))
        assert x.sum() == pytest.approx(a.sum())
        assert x.prod() == pytest.approx(a.prod())
        assert x.mean() == pytest.approx(a.mean())
        assert x.min() == pytest.approx(a.min())
        assert x.max() == pytest.approx(a.max())

    def test_reductions_honour_strides(self):
        """A transposed view holds the same elements, so every reduction must
        agree with the original — this is the strided code path."""
        a = arange(2, 3) + 1
        x = A.asarray(a)
        assert x.T.sum() == a.sum()
        assert x.T.prod() == a.prod()
        assert x.T.mean() == a.mean()
        assert x.T.max() == a.max()

    def test_reduction_over_a_slice(self):
        a = arange(4, 4)
        np.testing.assert_allclose(A.asarray(a)[:, 1:3].sum(), a[:, 1:3].sum())

    def test_sum_axis(self):
        a = arange(2, 3)
        x = A.asarray(a)
        np.testing.assert_allclose(x.sum_axis(0).to_numpy(), a.sum(axis=0))
        np.testing.assert_allclose(x.sum_axis(1).to_numpy(), a.sum(axis=1))

    def test_mean_axis(self):
        a = arange(2, 3)
        x = A.asarray(a)
        np.testing.assert_allclose(x.mean_axis(0).to_numpy(), a.mean(axis=0))
        np.testing.assert_allclose(x.mean_axis(1).to_numpy(), a.mean(axis=1))

    def test_axis_reduction_drops_one_rank(self):
        x = A.asarray(arange(2, 3, 4))
        assert x.sum_axis(1).ndim == 2

    def test_axis_out_of_range(self):
        x = A.asarray(arange(2, 3))
        with pytest.raises(ValueError, match="axis out of range"):
            x.sum_axis(5)

    def test_rank_1_has_no_axis_reduction(self):
        """sum_axis would give a rank-0 array, which MTL5 does not have."""
        assert not hasattr(A.asarray(arange(4)), "sum_axis")

    def test_empty_reductions_raise(self):
        z = A.zeros([0])
        assert z.sum() == 0.0  # a well-defined identity
        for op in ("mean", "min", "max"):
            with pytest.raises(ValueError, match="empty array"):
                getattr(z, op)()


class TestArithmetic:
    def test_elementwise(self):
        a, b = arange(2, 3) + 1, arange(2, 3) + 2
        x, y = A.asarray(a), A.asarray(b)
        np.testing.assert_allclose((x + y).to_numpy(), a + b)
        np.testing.assert_allclose((x - y).to_numpy(), a - b)
        np.testing.assert_allclose((x * y).to_numpy(), a * b)
        np.testing.assert_allclose((x / y).to_numpy(), a / b)

    def test_extent_one_broadcasts(self):
        a = arange(2, 3)
        col = np.full((2, 1), 10.0)
        np.testing.assert_allclose((A.asarray(a) + A.asarray(col)).to_numpy(), a + col)

    def test_result_owns_its_memory(self):
        a = arange(2, 3)
        r = A.asarray(a) + A.asarray(a)
        assert not r.is_view
        r[[0, 0]] = 99.0
        assert a[0, 0] == 0.0

    def test_incompatible_shapes_raise(self):
        with pytest.raises(ValueError, match="broadcast"):
            A.asarray(arange(2, 3)) + A.asarray(arange(4, 5))

    def test_rank_promotion_is_not_supported(self):
        """MTL5's broadcast_shape takes two shape<N> of equal N, so NumPy's
        rank promotion is out of reach. It must fail loudly, not quietly."""
        with pytest.raises(TypeError):
            A.asarray(arange(2, 3)) + A.asarray(arange(3))


class TestCopyAndFill:
    def test_copy_is_dense_and_independent(self):
        a = arange(2, 3)
        c = A.asarray(a).T.copy()
        assert c.is_c_contiguous
        assert not c.is_view
        np.testing.assert_array_equal(c.to_numpy(), a.T)
        c[[0, 0]] = 42.0
        assert a[0, 0] == 0.0

    def test_fill_honours_strides(self):
        a = arange(3, 3)
        A.asarray(a)[:, 1].fill(7.0)
        np.testing.assert_array_equal(a[:, 1], [7.0, 7.0, 7.0])
        assert a[0, 0] == 0.0, "only the sliced column should change"


class TestNumpyRoundTrip:
    def test_to_numpy_is_zero_copy(self):
        a = arange(2, 3)
        out = A.asarray(a).to_numpy()
        out[0, 0] = 31.0
        assert a[0, 0] == 31.0

    def test_to_numpy_carries_strides(self):
        """A transposed view must map onto NumPy without materialising."""
        a = arange(2, 3)
        out = A.asarray(a).T.to_numpy()
        np.testing.assert_array_equal(out, a.T)
        assert not out.flags["C_CONTIGUOUS"]

    def test_slice_to_numpy(self):
        a = arange(4, 5)
        np.testing.assert_array_equal(A.asarray(a)[1:3, ::2].to_numpy(), a[1:3, ::2])


class TestInterop:
    """The reason this layer earns its place here: it makes a DenseMatrix
    sliceable along either axis without a copy."""

    def test_as_ndarray_of_a_dense_matrix(self):
        a = arange(2, 3)
        M = mtl5.matrix(a)
        nd = A.as_ndarray(M)
        assert nd.ndim == 2
        np.testing.assert_array_equal(nd.to_numpy(), a)

    def test_as_ndarray_aliases_the_matrix(self):
        a = arange(2, 3)
        M = mtl5.matrix(a)
        A.as_ndarray(M)[[0, 0]] = 64.0
        assert M[0, 0] == 64.0

    def test_column_of_a_dense_matrix_without_a_copy(self):
        a = arange(3, 4)
        col = A.as_ndarray(mtl5.matrix(a))[:, 2]
        np.testing.assert_array_equal(col.to_numpy(), a[:, 2])

    def test_as_ndarray_of_a_dense_vector(self):
        v = mtl5.vector(np.arange(5.0))
        nd = A.as_ndarray(v)
        assert nd.ndim == 1
        np.testing.assert_array_equal(nd.to_numpy(), np.arange(5.0))

    def test_as_matrix(self):
        a = arange(2, 3)
        M = A.as_matrix(A.asarray(a))
        np.testing.assert_array_equal(M.to_numpy(), a)

    def test_as_matrix_honours_strides(self):
        """A transposed ndarray is not contiguous; the copy must still come out
        in logical order rather than raw memory order."""
        a = arange(2, 3)
        M = A.as_matrix(A.asarray(a).T)
        assert M.shape == (3, 2)
        np.testing.assert_array_equal(M.to_numpy(), a.T)

    def test_as_vector(self):
        v = A.as_vector(A.asarray(np.arange(4.0)))
        np.testing.assert_array_equal(v.to_numpy(), np.arange(4.0))

    def test_as_vector_honours_strides(self):
        a = arange(3, 4)
        v = A.as_vector(A.asarray(a)[:, 1])
        np.testing.assert_array_equal(v.to_numpy(), a[:, 1])


class TestLifetime:
    def test_a_view_keeps_its_source_alive(self):
        """The view borrows memory; dropping every Python reference to the
        source must not free it underneath."""
        import gc

        x = A.asarray(arange(3, 4))
        t = x.T
        s = x[1]
        del x
        gc.collect()
        assert t.sum() == 66.0
        assert s.sum() == pytest.approx(arange(3, 4)[1].sum())

    def test_numpy_source_is_kept_alive(self):
        import gc

        x = A.asarray(arange(2, 3))
        gc.collect()
        np.testing.assert_array_equal(x.to_numpy(), arange(2, 3))


class TestPublicSurface:
    def test_array_submodule_is_exported(self):
        assert "array" in mtl5.__all__
        assert hasattr(mtl5, "array")

    def test_dtype_suffixes(self):
        assert A.asarray(arange(2, 2, dt=np.float32)).dtype == "f32"
        assert A.asarray(arange(2, 2, dt=np.float64)).dtype == "f64"

    def test_strides_are_documented_as_elements(self):
        """NumPy's .strides is in bytes; these are in elements, so a caller
        comparing them directly would otherwise be quietly wrong."""
        a = arange(2, 3)
        x = A.asarray(a)
        assert x.strides == [3, 1]
        assert a.strides == (24, 8)
