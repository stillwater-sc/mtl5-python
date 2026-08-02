"""Mixed precision: convert(), accumulator policies, and iterative refinement.

The point of these tests is not just "does it run" — it is that choosing a wider
accumulator measurably improves accuracy, and that the quire is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

QUIRE_DTYPES = [
    "fp8",
    "fp16",
    "posit8",
    "posit16",
    "posit32",
    "posit64",
    "fixpnt8",
    "fixpnt16",
    "lns16",
    "lns32",
]


class TestConvert:
    def test_dtypes_listed(self):
        d = mtl5.dtypes()
        assert "posit16" in d and "f32" in d and "lns32" in d

    def test_roundtrip_shape_and_dtype(self):
        x = np.linspace(-2.0, 2.0, 32)
        for dt in mtl5.dtypes():
            v = mtl5.convert(x, dt)
            assert v.dtype == dt
            assert len(v) == 32
            assert v.to_numpy().shape == (32,)

    def test_matrix_convert(self):
        A = np.arange(12, dtype=np.float64).reshape(3, 4)
        M = mtl5.convert(A, "posit32")
        assert M.shape == (3, 4)
        npt = M.to_numpy()
        assert npt.shape == (3, 4)
        # posit32 has ~28 bits of significand near 1; small integers are exact.
        np.testing.assert_allclose(npt, A, rtol=0, atol=0)

    def test_native_convert_yields_view_classes(self):
        x = np.linspace(0.0, 1.0, 8)
        assert isinstance(mtl5.convert(x, "f64"), mtl5.DenseVector_f64)
        assert isinstance(mtl5.convert(x, "f32"), mtl5.DenseVector_f32)

    def test_quantization_actually_loses_information(self):
        """A narrow type must round — otherwise convert() is not doing its job."""
        x = np.array([1.0 + 2.0**-12])
        assert mtl5.convert(x, "fp8").to_numpy()[0] != x[0]
        assert mtl5.convert(x, "posit64").to_numpy()[0] == x[0]

    def test_unknown_dtype(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            mtl5.convert(np.zeros(3), "bfloat16")


class TestAccumulatorPolicy:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal(4000)

    def test_accumulators_advertised(self):
        for dt in QUIRE_DTYPES:
            assert "quire" in mtl5.mixed.accumulators(dt)
        for dt in ("f32", "f64"):
            assert "quire" not in mtl5.mixed.accumulators(dt)

    def test_quire_is_exact_for_posit(self, data):
        """The headline claim: an exact dot product, not merely a better one."""
        v = mtl5.convert(data, "posit16")
        rounded = v.to_numpy()  # the posit16 values, as float64
        exact = float(np.dot(rounded, rounded))
        got = mtl5.mixed.dot(v, v, accumulator="quire")
        assert got == pytest.approx(exact, rel=1e-15, abs=0.0)

    def test_wider_accumulator_is_monotonically_better(self, data):
        v = mtl5.convert(data, "posit16")
        rounded = v.to_numpy()
        exact = float(np.dot(rounded, rounded))

        def relerr(acc):
            return abs(mtl5.mixed.dot(v, v, accumulator=acc) - exact) / abs(exact)

        in_precision = relerr(None)
        f32 = relerr("f32")
        f64 = relerr("f64")
        quire = relerr("quire")

        # Accumulating 4000 posit16 products in posit16 is catastrophic; each
        # widening step must strictly improve on it.
        assert in_precision > 1e-3, "expected in-precision accumulation to be poor"
        assert f32 < in_precision
        assert f64 < f32
        assert quire <= f64

    def test_fma_accumulator_available(self, data):
        v = mtl5.convert(data, "posit16")
        rounded = v.to_numpy()
        exact = float(np.dot(rounded, rounded))
        got = mtl5.mixed.dot(v, v, accumulator="fma")
        assert abs(got - exact) / abs(exact) < 1e-12

    def test_result_element_rounds_back(self, data):
        """result='element' rounds the accumulator out to the element type."""
        v = mtl5.convert(data[:64], "posit8")
        wide = mtl5.mixed.dot(v, v, accumulator="quire", result="f64")
        narrow = mtl5.mixed.dot(v, v, accumulator="quire", result="element")
        # posit8 near this magnitude is coarse, so the round-back must show.
        assert narrow != wide
        assert mtl5.convert(np.array([narrow]), "posit8").to_numpy()[0] == narrow

    def test_native_float_accumulates_in_double(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(10000).astype(np.float32)
        v = mtl5.convert(x.astype(np.float64), "f32")
        exact = float(np.dot(x.astype(np.float64), x.astype(np.float64)))
        wide = mtl5.mixed.dot(v, v, accumulator="f64")
        assert abs(wide - exact) / abs(exact) < 1e-14

    def test_quire_rejected_for_native(self):
        v = mtl5.convert(np.ones(4), "f64")
        with pytest.raises(ValueError, match="not available for dtype"):
            mtl5.mixed.dot(v, v, accumulator="quire")

    def test_unknown_accumulator(self):
        v = mtl5.convert(np.ones(4), "posit16")
        with pytest.raises(ValueError, match="unknown accumulator"):
            mtl5.mixed.dot(v, v, accumulator="float128")

    def test_length_mismatch(self):
        a = mtl5.convert(np.ones(4), "posit16")
        b = mtl5.convert(np.ones(5), "posit16")
        with pytest.raises(ValueError, match="same length"):
            mtl5.mixed.dot(a, b, accumulator="f64")


class TestAccumulatedNorms:
    """These exercise the local sum-of-squares loops that work around
    stillwater-sc/mtl5#324 (two_norm<Acc>/frobenius_norm<Acc>)."""

    def test_norm_matches_numpy_with_wide_accumulator(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(3000)
        v = mtl5.convert(x, "posit32")
        exact = float(np.linalg.norm(v.to_numpy()))
        for acc in ("f64", "fma", "quire"):
            got = mtl5.mixed.norm(v, 2, accumulator=acc)
            assert got == pytest.approx(exact, rel=1e-12)

    def test_frobenius_norm(self):
        rng = np.random.default_rng(6)
        A = rng.standard_normal((40, 50))
        M = mtl5.convert(A, "posit32")
        exact = float(np.linalg.norm(M.to_numpy(), "fro"))
        assert mtl5.mixed.frobenius_norm(M, accumulator="quire") == pytest.approx(exact, rel=1e-12)

    def test_non_euclidean_ord_rejected(self):
        v = mtl5.convert(np.ones(4), "posit16")
        with pytest.raises(ValueError, match="only ord=2"):
            mtl5.mixed.norm(v, 1, accumulator="f64")


class TestMixedMatmul:
    def test_accumulator_improves_matmul(self):
        rng = np.random.default_rng(7)
        n = 60
        A = rng.standard_normal((n, n))
        Am = mtl5.convert(A, "posit16")
        Ar = Am.to_numpy()
        exact = Ar @ Ar

        def relerr(acc):
            got = mtl5.mixed.matmul(Am, Am, accumulator=acc).to_numpy()
            return np.linalg.norm(got - exact) / np.linalg.norm(exact)

        assert relerr("f64") < relerr(None)
        assert relerr("quire") <= relerr("f64")

    def test_matvec_accumulator(self):
        rng = np.random.default_rng(8)
        n = 80
        A = rng.standard_normal((n, n))
        x = rng.standard_normal(n)
        Am, xm = mtl5.convert(A, "posit16"), mtl5.convert(x, "posit16")
        exact = Am.to_numpy() @ xm.to_numpy()
        got = mtl5.mixed.matvec(Am, xm, accumulator="quire").to_numpy()
        # The products are exact in the quire; only the final store rounds.
        assert np.linalg.norm(got - exact) / np.linalg.norm(exact) < 1e-2

    def test_shape_checks(self):
        A = mtl5.convert(np.ones((2, 3)), "posit16")
        B = mtl5.convert(np.ones((2, 3)), "posit16")
        with pytest.raises(ValueError, match="A.num_cols"):
            mtl5.mixed.matmul(A, B, accumulator="f64")


class TestLuIterativeRefine:
    @pytest.fixture
    def system(self):
        rng = np.random.default_rng(3)
        n = 100
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        x = rng.standard_normal(n)
        return A, x, A @ x

    def test_low_precision_factor_reaches_high_accuracy(self, system):
        A, xt, b = system
        x, info = mtl5.mixed.lu_iterative_refine(
            A, b, working="posit16", max_iter=30, rel_tol=1e-14
        )
        assert info["converged"]
        assert info["iters"] > 0, "a posit16 factor must need correction steps"
        err = np.linalg.norm(x - xt) / np.linalg.norm(xt)
        assert err < 1e-12, f"refinement did not recover accuracy: {err:.2e}"

    def test_f64_working_needs_no_refinement(self, system):
        A, xt, b = system
        x, info = mtl5.mixed.lu_iterative_refine(A, b, working="f64", max_iter=30, rel_tol=1e-14)
        assert info["iters"] == 0
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-12

    def test_result_is_the_best_iterate(self, system):
        """An over-long max_iter must never degrade the answer."""
        A, xt, b = system
        short, _ = mtl5.mixed.lu_iterative_refine(A, b, working="fp16", max_iter=5)
        long, _ = mtl5.mixed.lu_iterative_refine(A, b, working="fp16", max_iter=200)
        e_short = np.linalg.norm(short - xt)
        e_long = np.linalg.norm(long - xt)
        assert e_long <= e_short * (1 + 1e-9)

    def test_scaled_variant_runs(self, system):
        A, xt, b = system
        x, _ = mtl5.mixed.lu_iterative_refine(
            A, b, working="posit16", max_iter=30, rel_tol=1e-14, scaled=True
        )
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-10

    def test_unknown_working_precision(self, system):
        A, _, b = system
        with pytest.raises(ValueError, match="unknown working precision"):
            mtl5.mixed.lu_iterative_refine(A, b, working="bfloat16")

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="must be square"):
            mtl5.mixed.lu_iterative_refine(np.ones((2, 3)), np.ones(2), working="f64")
        with pytest.raises(ValueError, match="len\\(b\\)"):
            mtl5.mixed.lu_iterative_refine(np.eye(3), np.ones(2), working="f64")


class TestBackwardError:
    def test_exact_solution_has_tiny_backward_error(self):
        rng = np.random.default_rng(4)
        n = 50
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        b = rng.standard_normal(n)
        x = np.linalg.solve(A, b)
        assert mtl5.mixed.backward_error(A, x, b) < 1e-14

    def test_zero_vector_has_backward_error_one(self):
        A = np.eye(4)
        b = np.ones(4)
        assert mtl5.mixed.backward_error(A, np.zeros(4), b) == pytest.approx(1.0)


class TestSparseIterativeRefine:
    @pytest.fixture
    def spd_system(self):
        sp = pytest.importorskip("scipy.sparse")
        rng = np.random.default_rng(0)
        n = 300
        off = -np.ones(n - 1)
        A = sp.diags([off, 4.0 * np.ones(n), off], [-1, 0, 1], format="csr")
        x = rng.standard_normal(n)
        return A, x, A @ x

    def test_refines_through_ic0(self, spd_system):
        import mtl5.sparse as ms

        A, xt, b = spd_system
        Am = ms.from_scipy(A)
        M = mtl5._core.IC0_f64(Am)
        x, info = mtl5.mixed.iterative_refine(Am, M, b, max_iter=100, rel_tol=1e-12)
        assert info["converged"]
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-10

    def test_size_mismatch_rejected(self, spd_system):
        import mtl5.sparse as ms

        A, _, _ = spd_system
        Am = ms.from_scipy(A)
        M = mtl5._core.IC0_f64(Am)
        with pytest.raises(ValueError, match="len\\(b\\)"):
            mtl5.mixed.iterative_refine(Am, M, np.ones(5))


class TestFormerlyBrokenUpstream:
    """Regressions for MTL5 defects this binding work found and upstream fixed.

    Each was a case the upstream tests did not reach, so they are kept here as
    well: an ILU(0) that never got checked against a known solve, and the IC(0)
    control that always got it right.
    """

    def _diagonal_system(self):
        sp = pytest.importorskip("scipy.sparse")
        import mtl5.sparse as ms

        d = np.array([4.0, 2.0, 8.0])
        return ms.from_scipy(sp.diags([d], [0], format="csr")), d, np.ones(3)

    def test_ilu0_solve_is_exact_on_a_diagonal_matrix(self):
        """stillwater-sc/mtl5#323: the back substitution used to count the
        diagonal twice, returning (b - d*b)/d for every input."""
        Am, d, b = self._diagonal_system()
        got = mtl5._core.ILU0_f64(Am).solve(b).to_numpy()
        np.testing.assert_allclose(got, b / d, rtol=1e-14)

    def test_ic0_solve_is_exact_on_a_diagonal_matrix(self):
        """The control that was always correct, kept alongside it."""
        Am, d, b = self._diagonal_system()
        got = mtl5._core.IC0_f64(Am).solve(b).to_numpy()
        np.testing.assert_allclose(got, b / d, rtol=1e-14)

    def test_refinement_through_ilu0_now_converges(self):
        """With #323 fixed, ILU(0) is usable as a refinement factorization —
        before, it drove x to zero and reported rel_residual 1.0."""
        sp = pytest.importorskip("scipy.sparse")
        import mtl5.sparse as ms

        n = 200
        off = -np.ones(n - 1)
        A = sp.diags([off, 4.0 * np.ones(n), off], [-1, 0, 1], format="csr")
        rng = np.random.default_rng(0)
        xt = rng.standard_normal(n)
        Am = ms.from_scipy(A)
        x, info = mtl5.mixed.iterative_refine(
            Am, mtl5._core.ILU0_f64(Am), A @ xt, max_iter=100, rel_tol=1e-12
        )
        assert info["converged"]
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-10
