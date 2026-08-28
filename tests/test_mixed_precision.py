"""Mixed precision: convert(), accumulator policies, and iterative refinement.

The point of these tests is not just "does it run" — it is that choosing a wider
accumulator measurably improves accuracy, and that the quire is exact.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mtl5


def exact_sum_of_squares(a) -> float:
    """The correctly-rounded float64 value of ``sum(a[i]**2)``.

    The obvious reference, ``np.dot(a, a)``, is *not* exact — it is itself a
    sequential float64 accumulation carrying its own rounding error. That makes
    it useless for ranking accumulators that are more accurate than float64:
    the ``f64`` accumulator reproduces it bit for bit and scores a perfect 0,
    while the exact quire differs from it and therefore scores *worse*. Tests
    built on it rank accumulators by agreement with float64 rather than by
    accuracy, and penalise the quire for being right.

    ``math.fsum`` is correctly rounded regardless of summation order, so it is
    the true value to within one rounding, and the quire reproduces it exactly.
    ``np.longdouble`` is not a portable substitute: it is 80-bit only on x86
    Linux, and plain float64 on Windows and macOS/arm64, where it would
    silently reintroduce the same defect.

    See https://github.com/stillwater-sc/mtl5-python/issues/62.
    """
    return math.fsum(float(x) * float(x) for x in a)


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


def _wrap32(x: int) -> int:
    """Reduce an exact integer to what an int32 accumulator holds.

    Two's-complement wrapping is MTL5's documented contract for integer lanes,
    and it is what makes an integer reduction bit-identical across lane counts,
    backends and thread partitions. Tests assert against this rather than
    against "no overflow", so they stay true in the regime these operand widths
    actually run in.
    """
    return ((x + 2**31) % 2**32) - 2**31


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
        """The headline claim: an exact dot product, not merely a better one.

        Asserted as exact equality. A tolerance here would be the weaker claim,
        and would hide the reference being only as good as float64.
        """
        v = mtl5.convert(data, "posit16")
        rounded = v.to_numpy()  # the posit16 values, as float64
        exact = exact_sum_of_squares(rounded)
        got = mtl5.mixed.dot(v, v, accumulator="quire")
        assert got == exact

    def test_wider_accumulator_is_monotonically_better(self, data):
        v = mtl5.convert(data, "posit16")
        rounded = v.to_numpy()
        exact = exact_sum_of_squares(rounded)

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
        # The quire is exact, so it does not merely tie float64 -- it hits the
        # true value. Stated as == 0.0 rather than <= f64 because that is the
        # actual property; <= f64 follows from it and would also be satisfied
        # by a quire that was merely no worse.
        assert quire <= f64
        assert quire == 0.0, f"the quire should be exact, got {quire}"

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
    """`norm(ord=2)` and `frobenius_norm` go through
    `mtl::two_norm<Acc, double>` / `mtl::frobenius_norm<Acc, double>`.

    Until stillwater-sc/mtl5#379 added the `Result` parameter they could not:
    `two_norm<Acc>` rounded back to the element type, so every accumulator
    returned the same number and `accumulator=` was unobservable. The ordering
    test below is the one that could not be written then — it would have passed
    trivially while asserting nothing.
    """

    def test_a_wider_accumulator_is_strictly_better(self):
        """posit16 elements, where the accumulator has room to matter. Measured
        against the exact norm of the same posit16 data, so the only error is
        the accumulation.

        The reference was ``np.longdouble``, which is 80-bit only on x86 Linux
        and plain float64 on Windows and macOS/arm64 -- so on two of the three
        CI platforms the ``err["quire"] == 0.0`` assertion below was being made
        against a float64 reference and passing on luck. ``math.fsum`` is
        correctly rounded everywhere.
        """
        rng = np.random.default_rng(0)
        v = mtl5.convert(rng.standard_normal(4000), "posit16")
        exact = math.sqrt(exact_sum_of_squares(v.to_numpy()))

        err = {
            acc: abs(mtl5.mixed.norm(v, 2, accumulator=acc) - exact) / exact
            for acc in ("f32", "f64", "quire")
        }
        assert err["f64"] < err["f32"], err
        assert err["quire"] <= err["f64"], err
        assert err["quire"] == 0.0, f"the quire should be exact, got {err['quire']}"

    def test_the_accumulator_beats_no_accumulator(self):
        """Without one, the sum is carried in posit16 and loses badly."""
        rng = np.random.default_rng(0)
        v = mtl5.convert(rng.standard_normal(4000), "posit16")
        none = mtl5.mixed.norm(v, 2)
        wide = mtl5.mixed.norm(v, 2, accumulator="quire")
        assert abs(none - wide) / wide > 1e-2, (none, wide)

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


class TestIntegerAccumulator:
    """`accumulator='i32'` over 8- and 16-bit integer operands (#88 phase 2).

    MTL5 v5.11.0 routes `dot<int32_t>` over narrow integer operands onto the
    hardware widening multiply-accumulate — vpmaddwd / vpdpbusd on x86, SMLAL /
    SDOT on NEON. An int32 accumulator is what those instructions accumulate
    into, so it is the accumulator this path exists for.
    """

    NARROW = [(np.int8, "i8"), (np.int16, "i16"), (np.uint8, "u8")]

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_accumulators_offers_only_i32(self, dt, suffix):
        assert mtl5.mixed.accumulators(suffix) == ["i32"]

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_matches_an_exact_int64_reference(self, dt, suffix):
        rng = np.random.default_rng(0)
        info = np.iinfo(dt)
        a = rng.integers(info.min, info.max + 1, size=2000, dtype=dt)
        b = rng.integers(info.min, info.max + 1, size=2000, dtype=dt)
        got = mtl5.mixed.dot(a, b, accumulator="i32")
        want = int(np.dot(a.astype(np.int64), b.astype(np.int64)))
        assert int(got) == _wrap32(want)

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_container_and_ndarray_forms_agree(self, dt, suffix):
        a = np.arange(1, 17, dtype=dt)
        via_np = mtl5.mixed.dot(a, a, accumulator="i32")
        via_vec = mtl5.mixed.dot(mtl5.vector(a), mtl5.vector(a), accumulator="i32")
        assert via_np == via_vec

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_accumulator_is_required(self, dt, suffix):
        """Omitting it means element precision, which is exactly the wrapping
        phase 1 refused to expose. Ask rather than silently redefine None."""
        v = mtl5.vector(np.full(6, 100, dtype=dt))
        with pytest.raises(ValueError, match="requires an explicit accumulator"):
            mtl5.mixed.dot(v, v)

    @pytest.mark.parametrize("dt,suffix", NARROW, ids=[s for _, s in NARROW])
    def test_result_element_is_refused(self, dt, suffix):
        """Rounding the int32 sum back to an 8- or 16-bit element would
        re-introduce the wrap the accumulator exists to avoid."""
        v = mtl5.vector(np.full(6, 100, dtype=dt))
        with pytest.raises(ValueError, match="does not accept result="):
            mtl5.mixed.dot(v, v, accumulator="i32", result="element")

    def test_i32_is_refused_on_non_integer_dtypes(self):
        v = mtl5.vector(np.ones(4))
        with pytest.raises(ValueError, match="not available for dtype"):
            mtl5.mixed.dot(v, v, accumulator="i32")

    def test_the_wrap_is_twos_complement_not_undefined(self):
        """Overflow is the contract, not an error case — and it is well defined.

        This is what the v5.11.0 pin buys: before it, the generic integer loops
        were UB on overflow rather than the documented modular wrap, and for
        these operand widths overflow is the normal regime.

        Full range i8: each -128 * -128 product is 16384, so an int32 holds
        exactly 131071 terms and 131072 is the first that does not.
        """
        for n in (131071, 131072, 131073):
            a = np.full(n, -128, dtype=np.int8)
            exact = n * 16384
            assert int(mtl5.mixed.dot(a, a, accumulator="i32")) == _wrap32(exact)
        assert 131071 * 16384 <= 2**31 - 1, "131071 terms must still be exact"
        assert 131072 * 16384 > 2**31 - 1, "...and 131072 must not"

    def test_i16_headroom_is_two_products(self):
        """One full-range i16 product uses 2^30 of the int32 range, so two
        already overflow it — five orders of magnitude less headroom than i8,
        which is why the quantized-inference instructions are 8-bit."""
        one = np.full(1, -32768, dtype=np.int16)
        assert int(mtl5.mixed.dot(one, one, accumulator="i32")) == 2**30
        two = np.full(2, -32768, dtype=np.int16)
        assert int(mtl5.mixed.dot(two, two, accumulator="i32")) == _wrap32(2**31)


class TestIntegerAccumulatorMixedSignedness:
    """`u8 x i8` is VNNI's native pairing on x86 — unsigned activations against
    signed weights — and is what quantized inference is written in.

    MTL5 accepts every pairing at the `dot` level and swaps the operands onto
    whichever form the machine implements, because a dot product is symmetric.
    The kernel below it (`simd::reduce_dot_widen`) rejects `(int8, uint8)`, but
    that restriction is the kernel's; re-exposing it here would refuse a call
    the library can serve.
    """

    def _operands(self):
        rng = np.random.default_rng(7)
        u = rng.integers(0, 256, size=2048, dtype=np.uint8)
        i = rng.integers(-128, 128, size=2048, dtype=np.int8)
        return u, i, int(np.dot(u.astype(np.int64), i.astype(np.int64)))

    def test_u8_times_i8(self):
        u, i, exact = self._operands()
        assert int(mtl5.mixed.dot(u, i, accumulator="i32")) == _wrap32(exact)

    def test_i8_times_u8_is_accepted_too(self):
        u, i, exact = self._operands()
        assert int(mtl5.mixed.dot(i, u, accumulator="i32")) == _wrap32(exact)

    def test_the_two_orders_agree(self):
        u, i, _ = self._operands()
        assert mtl5.mixed.dot(u, i, accumulator="i32") == mtl5.mixed.dot(i, u, accumulator="i32")

    def test_container_form_works_for_pairs(self):
        u, i, exact = self._operands()
        got = mtl5.mixed.dot(mtl5.vector(u), mtl5.vector(i), accumulator="i32")
        assert int(got) == _wrap32(exact)
