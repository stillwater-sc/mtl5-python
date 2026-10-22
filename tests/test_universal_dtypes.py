"""universal_dtypes arrays in and out of MTL5, bit for bit (#14).

The point is the word *bit*: the only route between the two libraries used to
be a round trip through float64, which is lossless for the narrow formats but
not for posit64 or the cascades, and is a rounding step either way. These tests
assert identity of encodings, not closeness of values.

They also check that the two libraries agree on what each encoding MEANS.
universal_dtypes and this package are built on different Universal releases,
and the formats are mapped by name, so a format that changed between releases
would otherwise cross bit-exactly and mean something else.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

ud = pytest.importorskip("universal_dtypes")

WORD = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
NARROW = ["posit8", "posit16", "fixpnt8", "fixpnt16", "lns16", "fp16"]
WIDE = ["posit32", "posit64", "lns32", "takum32"]
CASCADES = {"dd_cascade": 2, "td_cascade": 3, "qd_cascade": 4}


def _all_patterns(name):
    """Every encoding of an 8- or 16-bit format, as an array of that dtype."""
    dt = np.dtype(getattr(ud, name))
    w = WORD[dt.itemsize]
    return np.arange(2 ** (8 * dt.itemsize), dtype=np.uint64).astype(w).view(dt)


def _sampled_patterns(name, n=50_000, seed=0):
    dt = np.dtype(getattr(ud, name))
    w = WORD[dt.itemsize]
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, np.iinfo(w).max, size=n, dtype=w, endpoint=True)
    return raw.view(dt)


def _bits(a):
    return a.view(WORD[a.dtype.itemsize])


def _same_values(x, y):
    """Equal as float64, with NaN (NaR) matching NaN."""
    return np.all((x == y) | (np.isnan(x) & np.isnan(y)))


class TestBitExactRoundTrip:
    @pytest.mark.parametrize("name", NARROW)
    def test_every_pattern_survives(self, name):
        """ndarray[T] -> mtl5 -> ndarray[T] is the identity on all 2^n codes,
        NaR, NaN and both zeros included."""
        a = _all_patterns(name)
        back = mtl5.vector(a).to_universal()
        assert back.dtype == a.dtype
        np.testing.assert_array_equal(_bits(back), _bits(a))

    @pytest.mark.parametrize("name", WIDE)
    def test_sampled_patterns_survive(self, name):
        a = _sampled_patterns(name)
        back = mtl5.vector(a).to_universal()
        assert back.dtype == a.dtype
        np.testing.assert_array_equal(_bits(back), _bits(a))

    @pytest.mark.parametrize("name", NARROW + WIDE)
    def test_matrix_round_trip(self, name):
        a = _sampled_patterns(name, n=7 * 13, seed=1).reshape(7, 13)
        M = mtl5.matrix(a)
        assert M.shape == (7, 13)
        back = M.to_universal()
        assert back.shape == (7, 13) and back.dtype == a.dtype
        np.testing.assert_array_equal(_bits(back), _bits(a))

    def test_posit64_keeps_bits_float64_cannot(self):
        """The old float64 route cannot carry posit64: near 1 it has more
        fraction bits than a double."""
        a = np.array([1.0], dtype=ud.posit64)
        b = _bits(a).copy()
        b += 1  # the next posit64 above 1.0 -- not a float64 value
        a = b.view(a.dtype)
        assert _bits(mtl5.vector(a).to_universal())[0] == b[0]
        assert a.astype(np.float64)[0] == 1.0, "float64 rounds it back to 1"


class TestTheEncodingsMeanTheSameThing:
    """Bits crossing exactly is necessary, not sufficient: decode each one on
    both sides and compare. This is what pins the name-to-format mapping."""

    @pytest.mark.parametrize("name", NARROW)
    def test_every_pattern_decodes_alike(self, name):
        a = _all_patterns(name)
        assert _same_values(mtl5.vector(a).to_numpy(), a.astype(np.float64))

    @pytest.mark.parametrize("name", ["posit32", "lns32", "takum32"])
    def test_sampled_patterns_decode_alike(self, name):
        a = _sampled_patterns(name, seed=2)
        assert _same_values(mtl5.vector(a).to_numpy(), a.astype(np.float64))

    @pytest.mark.parametrize("name", NARROW + ["posit32", "lns32", "takum32"])
    def test_mtl5_encodes_as_universal_dtypes_does(self, name):
        """The other direction: round the same doubles on each side and compare
        the encodings produced."""
        x = np.random.default_rng(3).standard_normal(4000) * 4
        mine = mtl5.convert(x, name).to_universal()
        # The spread deliberately exceeds fixpnt8's [-8, 8), so saturation is
        # compared too; universal_dtypes reports it as a cast overflow.
        with np.errstate(over="ignore"):
            theirs = x.astype(getattr(ud, name))
        np.testing.assert_array_equal(_bits(mine), _bits(theirs))


class TestCascades:
    @pytest.mark.parametrize("name,limbs", CASCADES.items())
    def test_low_limbs_survive(self, name, limbs):
        """A cascade is limbs of float64, high limb first. 1/3 computed in the
        cascade has a non-zero second limb, which float64 would drop."""
        T = getattr(ud, name)
        a = np.array([1.0, 2.0, -5.0], dtype=T) / np.array([3.0, 7.0, 11.0], dtype=T)
        assert np.all(a.view(np.float64).reshape(3, limbs)[:, 1] != 0)
        back = mtl5.vector(a).to_universal()
        np.testing.assert_array_equal(back.view(np.float64), a.view(np.float64))

    @pytest.mark.parametrize("name,limbs", CASCADES.items())
    def test_both_sides_put_the_high_limb_first(self, name, limbs):
        v = mtl5.convert(np.array([0.1, -3.0]), name)
        limb = v.to_universal().view(np.float64).reshape(2, limbs)
        np.testing.assert_array_equal(limb[:, 0], [0.1, -3.0])
        assert np.all(limb[:, 1:] == 0)

    def test_matrix_of_cascades(self):
        T = ud.dd_cascade
        a = (np.arange(1.0, 7.0).astype(T) / np.full(6, 3.0).astype(T)).reshape(2, 3)
        back = mtl5.matrix(a).to_universal()
        assert back.shape == (2, 3)
        np.testing.assert_array_equal(back.view(np.float64), a.view(np.float64))


class TestUsableInMtl5:
    def test_linear_algebra_on_a_universal_array(self):
        a = np.array([3.0, 4.0], dtype=ud.posit16)
        v = mtl5.vector(a)
        assert isinstance(v, mtl5.DenseVector_posit16)
        assert mtl5.norm(v) == pytest.approx(5.0)

    def test_solve_returns_to_universal(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]]).astype(ud.posit32)
        b = np.array([1.0, 2.0]).astype(ud.posit32)
        x = mtl5.solve(mtl5.matrix(A), mtl5.vector(b)).to_universal()
        assert x.dtype == np.dtype(ud.posit32)
        np.testing.assert_allclose(x.astype(np.float64), [1 / 11, 7 / 11], rtol=1e-6)


class TestFactories:
    def test_copy_forms_accept_universal_arrays(self):
        a = _all_patterns("posit8")
        np.testing.assert_array_equal(_bits(mtl5.vector_copy(a).to_universal()), _bits(a))
        m = a[:256].reshape(16, 16)
        np.testing.assert_array_equal(_bits(mtl5.matrix_copy(m).to_universal()), _bits(m))

    def test_non_contiguous_input_is_accepted(self):
        """It is copied anyway, so a strided array costs nothing extra."""
        a = _all_patterns("posit16")[::3]
        np.testing.assert_array_equal(_bits(mtl5.vector(a).to_universal()), _bits(a))

    def test_the_input_is_copied_not_aliased(self):
        a = np.array([1.0, 2.0], dtype=ud.posit16)
        v = mtl5.vector(a)
        a[0] = 7.0
        assert v[0] == 1.0

    def test_native_dtypes_are_unaffected(self):
        a = np.arange(4.0)
        v = mtl5.vector(a)
        assert v.is_view
        assert mtl5.matrix(a.reshape(2, 2)).is_view

    def test_wrong_rank(self):
        with pytest.raises(TypeError, match="2-D array"):
            mtl5.matrix(np.ones(3, dtype=ud.posit16))
        with pytest.raises(TypeError, match="1-D array"):
            mtl5.vector(np.ones((2, 2), dtype=ud.posit16))

    def test_docstring_mentions_the_path(self):
        assert "universal_dtypes" in mtl5.vector.__doc__


class TestUnsupportedDtypesAreRefused:
    def test_fp8e5m2_is_not_mtl5s_fp8(self):
        """Same family name, different format: reading e5m2 bits as MTL5's
        e2m5 fp8 would cross exactly and mean something else."""
        with pytest.raises(TypeError, match="e2m5"):
            mtl5.vector(np.array([1.0], dtype=ud.fp8e5m2))

    @pytest.mark.parametrize("name", ["posit12", "posit8e0", "q15", "bfloat16", "takum16"])
    def test_no_mtl5_counterpart(self, name):
        with pytest.raises(TypeError, match="not supported"):
            mtl5.vector(np.array([0.5], dtype=getattr(ud, name)))

    def test_fp8_and_cfloat32_have_no_to_universal(self):
        assert not hasattr(mtl5.DenseVector_fp8, "to_universal")
        assert not hasattr(mtl5.DenseVector_cfloat32, "to_universal")
