"""Tests for the extended number types — cfloat32, takum32, and the cascades.

These five were bound for the number-system benchmarks (#69), which needed a
wider spread of formats than posit/fixpnt/lns gave us:

  cfloat32    IEEE binary32 through the emulation layer. The control: paired
              against hardware `float` it isolates emulation overhead from the
              cost of any particular format.
  takum32     tapered logarithmic, a different taper from posit's.
  dd/td/qd    cascades of 2/3/4 doubles — the only bound formats carrying MORE
              precision than float64.

A caveat that shapes every tolerance below: convert() takes float64 in and
to_numpy() hands float64 back, so a cascade's extra precision is only
observable in INTERMEDIATE accumulation, never at the Python boundary. A
round-tripped qd_cascade is exactly as accurate as a round-tripped float64;
what it buys is that the dot product in between did not lose anything.
"""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5

# Tolerances are for a round trip through float64, so they reflect each
# format's own precision — except the cascades, which are pinned at float64
# because that is what the boundary carries (see the module docstring).
EXTENDED = [
    ("cfloat32", 1e-6),
    ("takum32", 1e-6),
    ("dd_cascade", 1e-14),
    ("td_cascade", 1e-14),
    ("qd_cascade", 1e-14),
]
EXTENDED_NAMES = [name for name, _ in EXTENDED]


@pytest.mark.parametrize("name, tol", EXTENDED)
class TestExtendedTypeBasics:
    def test_listed_in_dtypes(self, name, tol):
        assert name in mtl5.dtypes()

    def test_vector_factory_and_class(self, name, tol):
        factory = getattr(mtl5, f"vector_{name}")
        vec_cls = getattr(mtl5, f"DenseVector_{name}")

        v = factory(np.array([1.0, 2.0, 3.0, 4.0]))
        assert isinstance(v, vec_cls)
        assert v.dtype == name
        assert len(v) == 4

    def test_matrix_factory_and_class(self, name, tol):
        factory = getattr(mtl5, f"matrix_{name}")
        mat_cls = getattr(mtl5, f"DenseMatrix_{name}")

        A = np.arange(12, dtype=np.float64).reshape(3, 4)
        M = factory(A)
        assert isinstance(M, mat_cls)
        assert M.shape == (3, 4)
        npt.assert_allclose(M.to_numpy(), A, rtol=tol)

    def test_convert_roundtrip(self, name, tol):
        x = np.linspace(-2.0, 2.0, 32)
        v = mtl5.convert(x, name)
        assert v.dtype == name
        npt.assert_allclose(v.to_numpy(), x, rtol=tol, atol=tol)

    def test_small_integers_are_exact(self, name, tol):
        # Every one of these formats has enough significand to hold small
        # integers exactly, so this is a rounding-free check that conversion
        # is wired to the right type.
        x = np.array([0.0, 1.0, 2.0, 4.0, 8.0, -3.0])
        v = mtl5.convert(x, name)
        npt.assert_allclose(v.to_numpy(), x, rtol=0, atol=0)


@pytest.mark.parametrize("name, tol", EXTENDED)
class TestExtendedTypeBlas:
    """dot / gemv / gemm through the mixed-precision entry points."""

    @staticmethod
    def _operands(n=48, seed=0):
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal(n),
            rng.standard_normal(n),
            rng.standard_normal((n, n)),
            rng.standard_normal((n, n)),
        )

    def test_dot(self, name, tol):
        a, b, _, _ = self._operands()
        got = mtl5.mixed.dot(mtl5.convert(a, name), mtl5.convert(b, name))
        npt.assert_allclose(got, float(a @ b), rtol=tol)

    # gemv/gemm accuracy is stated NORMWISE, not elementwise. An elementwise
    # rtol is the wrong instrument here: a result entry that lands near zero
    # through cancellation has a large relative error while its absolute error
    # is negligible, so elementwise rtol reports a failure that says nothing
    # about the kernel. ||computed - exact|| / ||exact|| is the standard
    # measure and is what the benchmark reports too.
    @staticmethod
    def _rel_err(got, exact):
        return float(np.linalg.norm(got - exact) / np.linalg.norm(exact))

    def test_gemv(self, name, tol):
        a, _, A, _ = self._operands()
        y = mtl5.mixed.matvec(mtl5.convert(A, name), mtl5.convert(a, name))
        assert self._rel_err(np.asarray(y.to_numpy()), A @ a) < tol

    def test_gemm(self, name, tol):
        _, _, A, B = self._operands()
        C = mtl5.mixed.matmul(mtl5.convert(A, name), mtl5.convert(B, name))
        assert self._rel_err(np.asarray(C.to_numpy()), A @ B) < tol

    def test_two_norm(self, name, tol):
        a, _, _, _ = self._operands()
        got = mtl5.mixed.norm(mtl5.convert(a, name), 2)
        npt.assert_allclose(got, float(np.linalg.norm(a)), rtol=tol)


class TestExtendedTypeQuire:
    """Quire availability follows Universal's fdp.hpp, not our type list.

    cfloat has one, so cfloat32 accumulates exactly. takum and the cascades do
    not, and must reject accumulator='quire' at runtime with the same clean
    error native float gets — never a silent fallback to a lesser accumulator,
    which would quietly invalidate an exactness claim.
    """

    def test_cfloat32_supports_quire(self):
        rng = np.random.default_rng(1)
        a, b = rng.standard_normal(64), rng.standard_normal(64)
        got = mtl5.mixed.dot(
            mtl5.convert(a, "cfloat32"),
            mtl5.convert(b, "cfloat32"),
            accumulator="quire",
        )
        # An exact accumulation of binary32 products beats the same dot summed
        # in binary32, so compare against the float64 reference generously —
        # the point is that it ran and stayed accurate, not the exact digits.
        npt.assert_allclose(got, float(a @ b), rtol=1e-6)

    @pytest.mark.parametrize("name", ["takum32", "dd_cascade", "td_cascade", "qd_cascade"])
    def test_quire_rejected_where_unavailable(self, name):
        rng = np.random.default_rng(2)
        a, b = rng.standard_normal(16), rng.standard_normal(16)
        with pytest.raises(ValueError, match="quire"):
            mtl5.mixed.dot(mtl5.convert(a, name), mtl5.convert(b, name), accumulator="quire")


class TestCfloat32IsEmulatedBinary32:
    """cfloat32 must round exactly like hardware binary32 — that equivalence is
    what makes it a valid control for emulation-overhead measurements. If it
    drifted from `float` semantics, a cfloat32/float ratio would be measuring
    two different computations."""

    def test_matches_numpy_float32_rounding(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(256) * 1e3
        got = mtl5.convert(x, "cfloat32").to_numpy()
        npt.assert_array_equal(got, x.astype(np.float32).astype(np.float64))

    def test_loses_precision_float64_keeps(self):
        # 1 + 2^-40 is representable in binary64 and not in binary32.
        x = np.array([1.0 + 2.0**-40])
        assert mtl5.convert(x, "cfloat32").to_numpy()[0] == 1.0


class TestCascadesExceedFloat64Internally:
    """The cascades' reason for existing: intermediate accumulation that
    float64 cannot do. The boundary is float64 either way, so this has to be
    measured on a sum whose ERROR is visible after rounding back."""

    def test_cascade_dot_beats_float64_on_a_cancelling_sum(self):
        # Classic catastrophic-cancellation pattern: large terms that cancel,
        # leaving a small exact answer. float64 loses the low bits during
        # accumulation; a cascade carries them.
        n = 1000
        rng = np.random.default_rng(4)
        a = rng.standard_normal(n) * 1e8
        b = np.ones(n)
        # Exact reference via Python's fsum (correctly rounded).
        import math

        exact = math.fsum(float(ai) * float(bi) for ai, bi in zip(a, b))

        naive = float(np.dot(a, b))
        cascade = mtl5.mixed.dot(mtl5.convert(a, "qd_cascade"), mtl5.convert(b, "qd_cascade"))

        err_naive = abs(naive - exact)
        err_cascade = abs(cascade - exact)
        # The cascade must be at least as good, and the inputs are exactly
        # representable, so it should be exact.
        assert err_cascade <= err_naive
        assert err_cascade == 0.0

    @pytest.mark.parametrize("name", ["dd_cascade", "td_cascade", "qd_cascade"])
    def test_cascade_roundtrip_is_float64_exact(self, name):
        # Boundary precision is float64 by construction — assert it rather than
        # leave a reader wondering whether the extra digits survive convert().
        x = np.array([1.0 + 2.0**-40, np.pi, 1e-300, 1e300])
        npt.assert_array_equal(mtl5.convert(x, name).to_numpy(), x)
