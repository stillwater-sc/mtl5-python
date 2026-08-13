"""LU and QR over the Universal number systems (#69).

Both were float32/float64 only until the number-system benchmarks needed them
for every format. Householder QR and partial-pivoting LU need nothing but sqrt
and the arithmetic operators, so all fifteen bound Universal types compile —
but compiling is not the interesting claim. What matters for mixed-precision
work is *where each format stops being usable*, and that is what this file
pins.

Three tiers, measured on an 8x8 in-range matrix (see TIERS below):

  32-bit and wider   usable: posit32 5.7e-8, takum32 7.9e-8, cascades ~1e-16
  16-bit             coarse but honest: posit16 2.6e-3, fp16 1.0e-2
  8-bit              not usable, and two of them fail SILENTLY

The silent failure is the reason this file exists. For fp8 and fixpnt8 every
Householder reflector rounds to zero, so Q is returned as the exact identity
and R as the upper triangle of A. The orthogonality check then reads a perfect
0.0 while the factorization reconstructs A to only ~14%. Anyone benchmarking
number systems will meet this, and it looks like a pass.
"""

import numpy as np
import pytest

import mtl5

N = 8

# (dtype, max ||QR-A||/||A||, max ||Q^T Q - I||, max LU residual)
#
# Bounds are ~3x the measured values, loose enough not to be a change detector
# for rounding-order changes and tight enough that a format silently dropping a
# tier fails. The 8-bit types are asserted as UNUSABLE rather than accurate —
# that is the honest claim, and pinning it stops a future change quietly
# "fixing" them without anyone noticing.
TIERS = [
    ("f64", 1e-14, 1e-14, 1e-14),
    ("cfloat32", 1e-6, 1e-5, 1e-6),
    ("posit32", 1e-6, 1e-6, 1e-7),
    ("takum32", 1e-6, 1e-6, 1e-7),
    ("posit64", 1e-14, 1e-14, 1e-14),
    ("dd_cascade", 1e-14, 1e-14, 1e-14),
    ("td_cascade", 1e-14, 1e-14, 1e-14),
    ("qd_cascade", 1e-14, 1e-14, 1e-14),
    ("lns32", 1e-3, 1e-3, 1e-3),
    ("posit16", 1e-1, 1e-1, 1e-2),
    ("fp16", 1e-1, 1e-1, 1e-2),
    ("lns16", 1e-1, 1e-1, 1e-1),
]

UNUSABLE_8BIT = ["posit8", "fp8", "fixpnt8"]

# These two do not merely lose accuracy — they return the identity for Q.
DEGENERATE_QR = ["fp8", "fixpnt8"]

ALL_UNIVERSAL = [
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
    "cfloat32",
    "takum32",
    "dd_cascade",
    "td_cascade",
    "qd_cascade",
]


def _problem(n=N, seed=42):
    """A well-conditioned system scaled to fit every format's dynamic range.

    The scaling is not cosmetic. fp8 tops out at 3.94 and fixpnt8 at 7.94, so
    the usual diagonally-dominant `randn(n,n) + n*I` construction overflows
    both and produces NaN — which would make this file assert on an
    out-of-range artifact rather than on the factorization.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.1 + np.eye(n)
    b = A @ np.ones(n)
    return A, b


def _qr_errors(dtype, A):
    q = mtl5.qr(mtl5.convert(A, dtype))
    Q = np.asarray(q.Q.to_numpy())
    R = np.asarray(q.R.to_numpy())
    resid = np.linalg.norm(Q @ R - A) / np.linalg.norm(A)
    orth = np.linalg.norm(Q.T @ Q - np.eye(A.shape[0]))
    return resid, orth, Q


def _lu_residual(dtype, A, b):
    factor = mtl5.lu(mtl5.convert(A, dtype))
    x = np.asarray(factor.solve(mtl5.convert(b, dtype)).to_numpy())
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


class TestAvailability:
    @pytest.mark.parametrize("dtype", ALL_UNIVERSAL)
    def test_qr_factor_class_exists(self, dtype):
        assert hasattr(mtl5._core, f"QRFactor_{dtype}")

    @pytest.mark.parametrize("dtype", ALL_UNIVERSAL)
    def test_lu_factor_class_exists(self, dtype):
        assert hasattr(mtl5._core, f"LUFactor_{dtype}")

    @pytest.mark.parametrize("dtype", ALL_UNIVERSAL)
    def test_qr_and_lu_run(self, dtype):
        A, b = _problem()
        M = mtl5.convert(A, dtype)
        q = mtl5.qr(M)
        assert q.shape == (N, N)
        assert q.dtype == dtype
        lu = mtl5.lu(M)
        assert lu.shape == (N, N)
        assert lu.dtype == dtype

    def test_numpy_error_now_points_at_convert(self):
        """The float-only hint had to stop being emitted for qr.

        mtl5/__init__.py picks the message by probing _core for a Universal
        instantiation, so this asserts the probe actually followed the C++
        change rather than silently still claiming float32/float64 only.
        """
        with pytest.raises(TypeError, match="convert"):
            mtl5.qr(np.arange(16, dtype=np.int32).reshape(4, 4))


@pytest.mark.parametrize("dtype, qr_tol, orth_tol, lu_tol", TIERS)
class TestUsableTiers:
    def test_qr_reconstructs_a(self, dtype, qr_tol, orth_tol, lu_tol):
        A, _ = _problem()
        resid, _, _ = _qr_errors(dtype, A)
        assert resid < qr_tol

    def test_q_is_orthogonal(self, dtype, qr_tol, orth_tol, lu_tol):
        A, _ = _problem()
        _, orth, _ = _qr_errors(dtype, A)
        assert orth < orth_tol

    def test_lu_solves(self, dtype, qr_tol, orth_tol, lu_tol):
        A, b = _problem()
        assert _lu_residual(dtype, A, b) < lu_tol


class TestNarrowFormatsAreNotUsable:
    """Asserted as failure, deliberately.

    An 8-bit format cannot carry these factorizations, and saying so in a test
    is more useful than omitting the types: it documents the boundary, and it
    means a future change that appears to fix them gets looked at rather than
    absorbed.
    """

    @pytest.mark.parametrize("dtype", UNUSABLE_8BIT)
    def test_qr_residual_is_large(self, dtype):
        A, _ = _problem()
        resid, _, _ = _qr_errors(dtype, A)
        assert resid > 1e-2, "an 8-bit QR that suddenly got accurate deserves a look"

    @pytest.mark.parametrize("dtype", DEGENERATE_QR)
    def test_reflectors_vanish_and_q_is_the_identity(self, dtype):
        """The silent failure, pinned exactly.

        Not "inaccurate" — structurally degenerate. Every reflector rounds to
        zero, so Q is bit-for-bit the identity. The orthogonality check reads
        0.0, which is why this cannot be caught by asserting orthogonality.
        """
        A, _ = _problem()
        resid, orth, Q = _qr_errors(dtype, A)

        np.testing.assert_array_equal(Q, np.eye(N))
        assert orth == 0.0  # a perfect score, from a useless factorization
        assert resid > 1e-2  # while failing to reconstruct A at all

    @pytest.mark.parametrize("dtype", DEGENERATE_QR)
    def test_r_is_just_the_upper_triangle_of_a(self, dtype):
        """Follows from Q == I, and is the part that explains the residual:
        R is A with the subdiagonal dropped, not a factor of it."""
        A, _ = _problem()
        q = mtl5.qr(mtl5.convert(A, dtype))
        R = np.asarray(q.R.to_numpy())
        A_rounded = np.asarray(mtl5.convert(A, dtype).to_numpy())
        np.testing.assert_array_equal(R, np.triu(A_rounded))

    def test_posit8_degrades_without_degenerating(self):
        """posit8 is unusable too, but by a different mechanism — its
        reflectors do not vanish, so it is inaccurate rather than structurally
        broken. Distinguished so the two failure modes are not conflated."""
        A, _ = _problem()
        resid, orth, Q = _qr_errors("posit8", A)
        assert not np.array_equal(Q, np.eye(N))
        assert orth > 1e-2
        assert resid > 1e-2


class TestFactorizationObjectReuse:
    """The reason LU is exposed as an object rather than only through solve():
    factor once at O(n^3), solve many at O(n^2)."""

    def test_one_factor_serves_several_right_hand_sides(self):
        A, _ = _problem()
        rng = np.random.default_rng(7)
        factor = mtl5.lu(mtl5.convert(A, "posit32"))
        for _ in range(3):
            b = rng.standard_normal(N)
            x = np.asarray(factor.solve(mtl5.convert(b, "posit32")).to_numpy())
            assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6

    def test_solve_rejects_a_mismatched_rhs(self):
        A, _ = _problem()
        factor = mtl5.lu(mtl5.convert(A, "posit32"))
        with pytest.raises(ValueError, match="dimension"):
            factor.solve(mtl5.convert(np.ones(N + 1), "posit32"))

    def test_lu_rejects_a_non_square_matrix(self):
        rng = np.random.default_rng(8)
        A = rng.standard_normal((4, 6))
        with pytest.raises(ValueError, match="square"):
            mtl5.lu(mtl5.convert(A, "posit32"))

    def test_qr_rejects_a_wide_matrix(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((3, 5))
        with pytest.raises(ValueError, match="num_rows >= num_cols"):
            mtl5.qr(mtl5.convert(A, "posit32"))
