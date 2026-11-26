"""SumK and DotK, the compensated and exact accumulators, and GenDot / GenSum.

Ogita, Rump and Oishi, "Accurate sum and dot product", SIAM J. Sci. Comput.
26(6), 2005. The claims tested are the paper's error bounds, against exact
references computed with `fractions.Fraction`, on the paper's own
ill-conditioned test data (`mtl5.generators.gen_dot` / `gen_sum`).
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import reduce

import numpy as np
import pytest

import mtl5
import mtl5.sparse as ms

U64 = 2.0**-53  # unit roundoff of float64
U32 = 2.0**-24  # and of float32


def exact_dot(x, y) -> Fraction:
    return sum((Fraction(float(a)) * Fraction(float(b)) for a, b in zip(x, y)), Fraction(0))


def exact_sum(p) -> Fraction:
    return sum((Fraction(float(a)) for a in p), Fraction(0))


def round_f64(q: Fraction) -> float:
    # Fraction.__float__ divides numerator by denominator, which Python rounds
    # correctly, to nearest even.
    return float(q)


def round_f32(q: Fraction) -> float:
    """The float32 nearest to q, ties to even. Rounding q to float64 first and
    then to float32 can round twice in the wrong direction near a tie, so the
    neighbours of that first guess are compared exactly."""
    guess = np.float32(float(q))
    cands = [
        np.nextafter(guess, np.float32(-np.inf)),
        guess,
        np.nextafter(guess, np.float32(np.inf)),
    ]
    best = min(cands, key=lambda c: (abs(Fraction(float(c)) - q), int(c.view(np.int32)) & 1))
    return float(best)


def gamma(m: int, u: float) -> float:
    return m * u / (1 - m * u)


def rel_err(got: float, exact: Fraction) -> float:
    return float(abs(Fraction(got) - exact) / abs(exact))


# ---------------------------------------------------------------------------
# sum_k
# ---------------------------------------------------------------------------
class TestSumK:
    def test_cancellation_that_a_plain_sum_loses(self):
        p = np.array([1e16, 1.0, -1e16])
        assert mtl5.sum_k(p, k=1) == 0.0  # 1e16 + 1 rounds back to 1e16
        assert mtl5.sum_k(p) == 1.0

    def test_k1_is_the_plain_recursive_sum(self):
        p = np.random.default_rng(1).standard_normal(1000) * 1e3
        assert mtl5.sum_k(p, k=1) == reduce(lambda a, b: a + b, p.tolist(), 0.0)

    @pytest.mark.parametrize("cond", [1e4, 1e12, 1e20, 1e40])
    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_meets_the_papers_bound(self, cond, k):
        g = mtl5.generators.gen_sum(200, cond, seed=7)
        p = np.asarray(g.p)
        n = len(p)
        s = exact_sum(p)
        S = sum(abs(Fraction(float(a))) for a in p)
        got = mtl5.sum_k(p, k=k)
        if k == 2:  # Proposition 4.5
            bound = U64 * abs(s) + Fraction(gamma(n - 1, U64)) ** 2 * S
        else:  # Proposition 4.10
            bound = (U64 + 3 * Fraction(gamma(n - 1, U64)) ** 2) * abs(s) + Fraction(
                gamma(2 * n - 2, U64)
            ) ** k * S
        assert abs(Fraction(got) - s) <= bound

    def test_each_pass_buys_a_factor_of_u(self):
        # At cond ~ 1e40, Sum2 is useless, Sum3 is far better, and Sum4
        # is correctly rounded or within an ulp of it.
        g = mtl5.generators.gen_sum(200, 1e40, seed=3)
        p = np.asarray(g.p)
        s = exact_sum(p)
        errs = [rel_err(mtl5.sum_k(p, k=k), s) for k in (2, 3, 4)]
        assert errs[0] > 1e-3
        assert errs[1] < errs[0]
        assert errs[2] <= 2 * U64

    def test_float32(self):
        g = mtl5.generators.gen_sum(100, 1e12, seed=5, dtype="f32")
        p = np.asarray(g.p)
        assert p.dtype == np.float32
        got = mtl5.sum_k(p, k=4)
        assert np.float32(got) == got  # a float32 result, widened losslessly
        assert got == round_f32(exact_sum(p)) == g.sum

    def test_empty_and_single(self):
        assert mtl5.sum_k(np.array([])) == 0.0
        assert mtl5.sum_k(np.array([2.5]), k=5) == 2.5

    def test_densevector(self):
        p = np.array([1e16, 1.0, -1e16, 3.0])
        assert mtl5.sum_k(mtl5.vector(p)) == mtl5.sum_k(p) == 4.0

    @pytest.mark.parametrize("k", [0, -1, 11])
    def test_k_out_of_range(self, k):
        with pytest.raises(ValueError, match="k must be between 1 and 10"):
            mtl5.sum_k(np.ones(3), k=k)


# ---------------------------------------------------------------------------
# dot_k
# ---------------------------------------------------------------------------
class TestDotK:
    @pytest.mark.parametrize("cond", [1e4, 1e12, 1e24, 1e40])
    @pytest.mark.parametrize("k", [2, 3, 4, 6])
    def test_meets_the_papers_bound(self, cond, k):
        g = mtl5.generators.gen_dot(100, cond, seed=11)
        x, y = np.asarray(g.x), np.asarray(g.y)
        n = len(x)
        d = exact_dot(x, y)
        xy = exact_dot(np.abs(x), np.abs(y))
        got = mtl5.dot_k(x, y, k=k)
        if k == 2:  # Proposition 5.5
            bound = U64 * abs(d) + Fraction(gamma(n, U64)) ** 2 * xy
        else:  # Proposition 5.11
            g4 = Fraction(gamma(4 * n - 2, U64))
            bound = (U64 + 2 * g4**2) * abs(d) + g4**k * xy
        assert abs(Fraction(got) - d) <= bound

    def test_generator_value_is_the_exact_dot_rounded(self):
        g = mtl5.generators.gen_dot(60, 1e30, seed=2)
        assert g.dot == round_f64(exact_dot(g.x, g.y))

    def test_dot2_is_the_compensated_accumulator_bit_for_bit(self):
        for seed in range(5):
            g = mtl5.generators.gen_dot(100, 1e20, seed=seed)
            x, y = np.asarray(g.x), np.asarray(g.y)
            assert mtl5.dot_k(x, y, k=2) == mtl5.mixed.dot(x, y, accumulator="compensated")

    def test_float32(self):
        g = mtl5.generators.gen_dot(50, 1e10, seed=4, dtype="f32")
        x, y = np.asarray(g.x), np.asarray(g.y)
        assert x.dtype == y.dtype == np.float32
        got = mtl5.dot_k(x, y, k=4)
        assert got == round_f32(exact_dot(x, y)) == g.dot

    def test_densevector(self):
        g = mtl5.generators.gen_dot(30, 1e8, seed=9)
        x, y = np.asarray(g.x), np.asarray(g.y)
        assert mtl5.dot_k(g.x, g.y, k=3) == mtl5.dot_k(x, y, k=3)

    def test_empty(self):
        assert mtl5.dot_k(np.array([]), np.array([])) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            mtl5.dot_k(np.ones(3), np.ones(4))

    @pytest.mark.parametrize("k", [1, 0, 11])
    def test_k_out_of_range(self, k):
        with pytest.raises(ValueError, match="k must be between 2 and 10"):
            mtl5.dot_k(np.ones(3), np.ones(3), k=k)

    def test_float32_with_float64_widens_exactly(self):
        # nanobind converts the float32 operand to float64, which is exact, so
        # the result is the float64 DotK of the same values.
        x = np.random.default_rng(8).standard_normal(20).astype(np.float32)
        y = np.random.default_rng(9).standard_normal(20)
        assert mtl5.dot_k(x, y, k=3) == mtl5.dot_k(x.astype(np.float64), y, k=3)


# ---------------------------------------------------------------------------
# accumulator='compensated' and 'exact'
# ---------------------------------------------------------------------------
class TestAccumulators:
    @pytest.mark.parametrize("dt", ["f32", "f64"])
    def test_advertised_for_ieee(self, dt):
        acc = mtl5.mixed.accumulators(dt)
        assert "compensated" in acc and "exact" in acc

    @pytest.mark.parametrize("dt", ["posit16", "takum32", "dd_cascade", "fp16"])
    def test_not_advertised_elsewhere(self, dt):
        acc = mtl5.mixed.accumulators(dt)
        assert "compensated" not in acc and "exact" not in acc

    @pytest.mark.parametrize("acc", ["compensated", "exact"])
    def test_refused_for_universal_types(self, acc):
        v = mtl5.convert(np.ones(4), "posit16")
        with pytest.raises(ValueError, match="float32 and float64"):
            mtl5.mixed.dot(v, v, accumulator=acc)

    @pytest.mark.parametrize("acc", ["compensated", "exact"])
    def test_refused_for_sparse_factorizations(self, acc):
        A = mtl5.generators.laplacian_2d(4, 4)
        with pytest.raises(ValueError, match="sparse"):
            ms.splu(A, accumulator=acc)

    @pytest.mark.parametrize("cond", [1e10, 1e50, 1e150, 1e300])
    def test_exact_is_correctly_rounded_at_any_condition_number(self, cond):
        g = mtl5.generators.gen_dot(80, cond, seed=13)
        x, y = np.asarray(g.x), np.asarray(g.y)
        assert mtl5.mixed.dot(x, y, accumulator="exact") == round_f64(exact_dot(x, y)) == g.dot

    def test_exact_float32_rounds_once(self):
        g = mtl5.generators.gen_dot(80, 1e20, seed=17, dtype="f32")
        x, y = np.asarray(g.x), np.asarray(g.y)
        q = exact_dot(x, y)
        # result='element' rounds the exact sum once, to float32.
        assert mtl5.mixed.dot(x, y, accumulator="exact", result="element") == round_f32(q)
        # The default delivers it rounded once to float64 instead.
        assert mtl5.mixed.dot(x, y, accumulator="exact") == round_f64(q)

    def test_exact_norm_is_the_rounded_square_root_of_the_exact_sum(self):
        x = np.random.default_rng(3).standard_normal(500) * 1e3
        ss = round_f64(sum((Fraction(float(v)) ** 2 for v in x), Fraction(0)))
        assert mtl5.mixed.norm(mtl5.vector(x), accumulator="exact") == math.sqrt(ss)

    def test_compensated_beats_plain_on_ill_conditioned_data(self):
        g = mtl5.generators.gen_dot(100, 1e14, seed=21)
        x, y = np.asarray(g.x), np.asarray(g.y)
        d = exact_dot(x, y)
        plain = rel_err(mtl5.mixed.dot(x, y, accumulator="f64"), d)
        comp = rel_err(mtl5.mixed.dot(x, y, accumulator="compensated"), d)
        assert comp <= 2 * U64 < 1e-6 < plain


# ---------------------------------------------------------------------------
# gen_dot / gen_sum
# ---------------------------------------------------------------------------
class TestGenerators:
    def test_seeded_and_deterministic(self):
        a = mtl5.generators.gen_dot(40, 1e20, seed=1)
        b = mtl5.generators.gen_dot(40, 1e20, seed=1)
        c = mtl5.generators.gen_dot(40, 1e20, seed=2)
        assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
        assert np.array_equal(np.asarray(a.y), np.asarray(b.y))
        assert (a.dot, a.cond) == (b.dot, b.cond)
        assert not np.array_equal(np.asarray(a.x), np.asarray(c.x))

    def test_dot_result_fields(self):
        g = mtl5.generators.gen_dot(50, 1e16, seed=6)
        assert isinstance(g, mtl5.IllConditionedDot)
        x, y = np.asarray(g.x), np.asarray(g.y)
        assert x.shape == y.shape == (50,)
        d = exact_dot(x, y)
        want = 2 * exact_dot(np.abs(x), np.abs(y)) / abs(d)
        assert g.cond == pytest.approx(float(want), rel=1e-12)
        # The anticipated condition number is reached to within a few decades.
        assert 1e13 < g.cond < 1e19

    def test_sum_result_fields(self):
        g = mtl5.generators.gen_sum(64, 1e16, seed=6)
        assert isinstance(g, mtl5.IllConditionedSum)
        p = np.asarray(g.p)
        assert p.shape == (64,)
        s = exact_sum(p)
        assert g.sum == round_f64(s)
        want = sum(abs(Fraction(float(a))) for a in p) / abs(s)
        assert g.cond == pytest.approx(float(want), rel=1e-12)

    @pytest.mark.parametrize(
        "call,match",
        [
            (lambda: mtl5.generators.gen_dot(5, 1e10, seed=0), "n must be >= 6"),
            (lambda: mtl5.generators.gen_dot(10, 0.5, seed=0), "c must be >= 1"),
            (lambda: mtl5.generators.gen_dot(10, 1e40, seed=0, dtype="f32"), "too large"),
            (lambda: mtl5.generators.gen_dot(10, 1e10, seed=0, dtype="posit16"), "dtype"),
            (lambda: mtl5.generators.gen_sum(13, 1e10, seed=0), "even and >= 12"),
            (lambda: mtl5.generators.gen_sum(10, 1e10, seed=0), "even and >= 12"),
        ],
    )
    def test_invalid_arguments(self, call, match):
        with pytest.raises(ValueError, match=match):
            call()

    def test_seed_is_required(self):
        with pytest.raises(TypeError):
            mtl5.generators.gen_dot(10, 1e10)
