"""Dense factorizations: QR, LQ, LDL^T, and Cholesky across number systems.

The LDL^T coverage exists for a specific reason (#18): Cholesky takes square
roots and so refuses an indefinite matrix, while LDL^T does not. Being able to
run both in the same low precision is what makes that comparison possible.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

UNIVERSAL = ["fp16", "posit16", "posit32", "posit64", "fixpnt16", "lns32"]


def sym(n: int, seed: int = 0) -> np.ndarray:
    A = np.random.default_rng(seed).standard_normal((n, n))
    return (A + A.T) / 2


def spd(n: int, seed: int = 0) -> np.ndarray:
    A = np.random.default_rng(seed).standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


class TestQR:
    @pytest.fixture
    def lstsq(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((200, 40))
        return A, rng.standard_normal(200)

    def test_matches_numpy_lstsq(self, lstsq):
        A, b = lstsq
        x = mtl5.qr(mtl5.matrix(A)).solve(mtl5.vector(b)).to_numpy()
        ref = np.linalg.lstsq(A, b, rcond=None)[0]
        assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-10

    def test_q_is_orthogonal_and_qr_reconstructs(self, lstsq):
        A, _ = lstsq
        fac = mtl5.qr(mtl5.matrix(A))
        Q, R = fac.Q.to_numpy(), fac.R.to_numpy()
        assert np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0])) < 1e-11
        assert np.linalg.norm(Q @ R - A) / np.linalg.norm(A) < 1e-12

    def test_r_is_upper_triangular(self, lstsq):
        A, _ = lstsq
        R = mtl5.qr(mtl5.matrix(A)).R.to_numpy()
        assert np.allclose(np.tril(R, -1), 0.0)

    def test_square_system(self):
        A = spd(20)
        xt = np.random.default_rng(3).standard_normal(20)
        x = mtl5.qr(mtl5.matrix(A)).solve(mtl5.vector(A @ xt)).to_numpy()
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-10

    def test_metadata(self, lstsq):
        A, _ = lstsq
        fac = mtl5.qr(mtl5.matrix(A))
        assert fac.shape == A.shape
        assert fac.dtype == "f64"
        assert "QRFactor_f64" in repr(fac)

    def test_rejects_underdetermined(self):
        A = np.random.default_rng(0).standard_normal((5, 12))
        with pytest.raises(ValueError, match="num_rows >= num_cols"):
            mtl5.qr(mtl5.matrix(A))

    def test_rhs_length_checked(self, lstsq):
        A, _ = lstsq
        with pytest.raises(ValueError, match="does not match the row count"):
            mtl5.qr(mtl5.matrix(A)).solve(mtl5.vector(np.ones(3)))

    def test_float32(self, lstsq):
        A, b = lstsq
        fac = mtl5.qr(mtl5.matrix(A.astype(np.float32)))
        assert fac.dtype == "f32"
        x = np.asarray(fac.solve(mtl5.vector(b.astype(np.float32))).to_numpy(), dtype=float)
        ref = np.linalg.lstsq(A, b, rcond=None)[0]
        assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-4


class TestLQ:
    def test_reconstructs_and_q_is_orthogonal(self):
        A = np.random.default_rng(1).standard_normal((40, 200))
        fac = mtl5.lq(mtl5.matrix(A))
        L, Q = fac.L.to_numpy(), fac.Q.to_numpy()
        assert np.linalg.norm(L @ Q - A) / np.linalg.norm(A) < 1e-12
        assert np.linalg.norm(Q @ Q.T - np.eye(Q.shape[0])) < 1e-11

    def test_l_is_lower_triangular(self):
        A = np.random.default_rng(2).standard_normal((30, 60))
        L = mtl5.lq(mtl5.matrix(A)).L.to_numpy()
        assert np.allclose(np.triu(L[:, : L.shape[0]], 1), 0.0)

    def test_metadata(self):
        A = np.random.default_rng(2).standard_normal((30, 60))
        fac = mtl5.lq(mtl5.matrix(A))
        assert fac.shape == A.shape
        assert "LQFactor_f64" in repr(fac)

    @pytest.mark.parametrize("shape", [(40, 200), (20, 20), (200, 40), (1, 5), (5, 1)])
    def test_accepts_any_shape(self, shape):
        """LQ has no shape restriction, unlike QR.

        A tall A gives the economy form — L is num_rows x num_cols and Q is
        num_cols x num_cols — and still reconstructs exactly, so there is
        nothing to reject.
        """
        A = np.random.default_rng(11).standard_normal(shape)
        fac = mtl5.lq(mtl5.matrix(A))
        L, Q = fac.L.to_numpy(), fac.Q.to_numpy()
        assert Q.shape == (shape[1], shape[1])
        assert L.shape == shape
        assert np.linalg.norm(L @ Q - A) / np.linalg.norm(A) < 1e-12
        assert np.linalg.norm(Q @ Q.T - np.eye(Q.shape[0])) < 1e-11


class TestLDLT:
    def test_solves_an_indefinite_system(self):
        """The whole point: a matrix Cholesky would refuse."""
        S = sym(30, seed=4)
        b = np.random.default_rng(5).standard_normal(30)
        x = mtl5.ldlt(mtl5.matrix(S)).solve(mtl5.vector(b)).to_numpy()
        assert (
            np.linalg.norm(x - np.linalg.solve(S, b)) / np.linalg.norm(np.linalg.solve(S, b))
            < 1e-10
        )

    def test_diagonal_reveals_indefiniteness(self):
        S = sym(30, seed=4)
        d = mtl5.ldlt(mtl5.matrix(S)).diagonal()
        assert d.shape == (30,)
        assert (d > 0).any() and (d < 0).any(), "a random symmetric matrix is indefinite"
        # The inertia from D must match the eigenvalue signs.
        eig = np.linalg.eigvalsh(S)
        assert (d > 0).sum() == (eig > 0).sum()
        assert (d < 0).sum() == (eig < 0).sum()

    def test_spd_diagonal_is_all_positive(self):
        d = mtl5.ldlt(mtl5.matrix(spd(15, seed=6))).diagonal()
        assert (d > 0).all()

    def test_agrees_with_cholesky_on_spd(self):
        P = spd(20, seed=7)
        b = np.random.default_rng(8).standard_normal(20)
        x_ldlt = mtl5.ldlt(mtl5.matrix(P)).solve(mtl5.vector(b)).to_numpy()
        x_chol = mtl5.cholesky(mtl5.matrix(P)).solve(mtl5.vector(b)).to_numpy()
        np.testing.assert_allclose(x_ldlt, x_chol, rtol=1e-10)

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="must be square"):
            mtl5.ldlt(mtl5.matrix(np.ones((3, 5))))

    def test_rhs_length_checked(self):
        fac = mtl5.ldlt(mtl5.matrix(sym(8, seed=9)))
        with pytest.raises(ValueError, match="does not match factor size"):
            fac.solve(mtl5.vector(np.ones(3)))

    def test_zero_pivot_is_reported_not_silently_wrong(self):
        """LDL^T does not pivot, so it must refuse rather than return garbage."""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(RuntimeError, match="zero pivot"):
            mtl5.ldlt(mtl5.matrix(A))


class TestAcrossNumberSystems:
    """#18: Cholesky and LDL^T available in the same precision, so they can be
    compared on a covariance matrix that drifts out of positive-definiteness."""

    @pytest.mark.parametrize("dt", UNIVERSAL)
    def test_ldlt_available(self, dt):
        P = spd(8, seed=10)
        fac = mtl5.ldlt(mtl5.convert(P, dt))
        assert fac.dtype == dt
        assert fac.n == 8

    @pytest.mark.parametrize("dt", UNIVERSAL)
    def test_cholesky_available(self, dt):
        P = spd(8, seed=10)
        fac = mtl5.cholesky(mtl5.convert(P, dt))
        assert fac.dtype == dt

    @pytest.mark.parametrize("dt", ["posit32", "posit64"])
    def test_ldlt_solves_accurately_in_wide_universal(self, dt):
        P = spd(10, seed=11)
        b = np.random.default_rng(12).standard_normal(10)
        fac = mtl5.ldlt(mtl5.convert(P, dt))
        x = np.asarray(fac.solve(mtl5.convert(b, dt)).to_numpy(), dtype=float)
        assert (
            np.linalg.norm(x - np.linalg.solve(P, b)) / np.linalg.norm(np.linalg.solve(P, b)) < 1e-4
        )

    def test_cholesky_refuses_indefinite_where_ldlt_succeeds(self):
        """The comparison #18 is after, in one assertion."""
        P = np.eye(6)
        P[3, 3] = -1e-3  # a covariance update has pushed one direction negative
        for dt in ("f64", "posit32", "posit16"):
            M = mtl5.matrix(P) if dt == "f64" else mtl5.convert(P, dt)
            with pytest.raises(RuntimeError):
                mtl5.cholesky(M)
            mtl5.ldlt(M)  # must not raise
            d = mtl5.ldlt(M).diagonal()
            assert (d < 0).any(), f"{dt}: D should record the negative direction"

    def test_accepts_a_raw_numpy_array(self):
        """Same convenience mtl5.cholesky() has always offered."""
        S = sym(6, seed=13)
        b = np.random.default_rng(14).standard_normal(6)
        x = mtl5.ldlt(S).solve(mtl5.vector(b)).to_numpy()
        np.testing.assert_allclose(x, np.linalg.solve(S, b), rtol=1e-10)
        assert mtl5.qr(np.random.default_rng(0).standard_normal((9, 3))).dtype == "f64"

    def test_rejects_an_unsupported_numpy_dtype(self):
        with pytest.raises(TypeError, match="must be float32 or float64"):
            mtl5.ldlt(np.eye(3, dtype=np.int64))

    def test_errors_name_the_public_function_not_the_class(self):
        """The message must not leak 'ldltfactor'/'qrfactor'."""
        for fn, name in ((mtl5.qr, "qr"), (mtl5.lq, "lq"), (mtl5.ldlt, "ldlt")):
            with pytest.raises(TypeError) as exc:
                fn(np.eye(3, dtype=np.int64))
            msg = str(exc.value)
            assert msg.startswith(f"{name}:")
            assert "factor" not in msg.lower().replace("factorization", "")

    def test_convert_hint_offered_by_every_dispatched_factorization(self):
        """Every dense factorization now has Universal instantiations (#73), so
        pointing an unsupported NumPy dtype at convert() is always the useful
        answer.

        cholesky is excluded because it is not dispatched through the Python
        layer at all — it is bound straight from _core, so nanobind implicitly
        converts an int64 array to float32 and it never reaches this message.
        Long-standing behaviour, unrelated to #73, but it does mean
        `mtl5.cholesky(int_array)` silently picks a precision where the other
        four raise.
        """
        for fn in (mtl5.ldlt, mtl5.qr, mtl5.lq, mtl5.bunch_kaufman):
            with pytest.raises(TypeError, match="convert"):
                fn(np.eye(3, dtype=np.int64))

    def test_cholesky_silently_accepts_an_integer_array(self):
        """Pinning the asymmetry above rather than leaving it as folklore.

        Asserted on the class rather than a `.dtype` property, because the
        native CholeskyFactor does not carry one — only the Universal
        instantiations do. A second small inconsistency in the same corner.
        """
        fac = mtl5.cholesky(np.eye(3, dtype=np.int64))
        assert type(fac).__name__ == "CholeskyFactor_f32"

    def test_float_only_hint_survives_for_a_factorization_without_universal_support(self):
        """The other branch of that hint no longer has a caller.

        Nothing bound is float-only any more, so the "supports float32 and
        float64 only" message is unreachable through the public functions. It
        is kept for the next factorization added ahead of its instantiations,
        and exercised here directly rather than deleted along with the last
        function that happened to demonstrate it.
        """
        from mtl5 import _as_mtl5_matrix

        with pytest.raises(TypeError) as exc:
            _as_mtl5_matrix("newfac", "NoSuchFactor", np.eye(3, dtype=np.int64))
        assert "convert" not in str(exc.value)
        assert "float32 and float64 only" in str(exc.value)

    @pytest.mark.parametrize(
        "fn, name",
        [
            (mtl5.qr, "qr"),
            (mtl5.lq, "lq"),
            (mtl5.ldlt, "ldlt"),
            (mtl5.cholesky, "cholesky"),
            (mtl5.bunch_kaufman, "bunch_kaufman"),
        ],
    )
    def test_every_factorization_accepts_a_universal_dtype(self, fn, name):
        M = mtl5.convert(spd(4, seed=26), "posit32")
        assert fn(M).dtype == "posit32"

    def test_rejects_a_non_matrix(self):
        with pytest.raises(TypeError, match="expected an MTL5 matrix"):
            mtl5.ldlt("not a matrix")


class TestBunchKaufman:
    """Bound once stillwater-sc/mtl5#335 was fixed. The tests below are the ones
    that caught the original defect — the failure needed a pivot interchange,
    which n=2/n=3 cases never produce."""

    def test_solves_an_indefinite_system(self):
        S = sym(30, seed=4)
        b = np.random.default_rng(5).standard_normal(30)
        x = mtl5.bunch_kaufman(mtl5.matrix(S)).solve(mtl5.vector(b)).to_numpy()
        ref = np.linalg.solve(S, b)
        assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-10

    @pytest.mark.parametrize("n", [4, 5, 8, 12, 20, 33])
    def test_correct_at_sizes_that_force_interchanges(self, n):
        """n >= 4 is where the original bug lived; n=2/3 never pivot."""
        S = sym(n, seed=n)
        b = np.random.default_rng(n).standard_normal(n)
        x = mtl5.bunch_kaufman(mtl5.matrix(S)).solve(mtl5.vector(b)).to_numpy()
        ref = np.linalg.solve(S, b)
        assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-9, f"n={n}"

    def test_agrees_with_ldlt_where_both_apply(self):
        S = sym(16, seed=21)
        b = np.random.default_rng(22).standard_normal(16)
        x_bk = mtl5.bunch_kaufman(mtl5.matrix(S)).solve(mtl5.vector(b)).to_numpy()
        x_ld = mtl5.ldlt(mtl5.matrix(S)).solve(mtl5.vector(b)).to_numpy()
        np.testing.assert_allclose(x_bk, x_ld, rtol=1e-9)

    def test_survives_a_zero_pivot_that_plain_ldlt_rejects(self):
        """The reason Bunch-Kaufman exists."""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(RuntimeError, match="zero pivot"):
            mtl5.ldlt(mtl5.matrix(A))
        b = np.array([1.0, 2.0])
        x = mtl5.bunch_kaufman(mtl5.matrix(A)).solve(mtl5.vector(b)).to_numpy()
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-12)

    def test_ipiv_reports_the_pivot_record(self):
        fac = mtl5.bunch_kaufman(mtl5.matrix(sym(10, seed=23)))
        piv = fac.ipiv()
        assert piv.shape == (10,)
        assert piv.dtype == np.int64

    def test_metadata_and_shape_checks(self):
        fac = mtl5.bunch_kaufman(mtl5.matrix(sym(6, seed=24)))
        assert fac.n == 6 and fac.shape == (6, 6) and fac.dtype == "f64"
        assert "BunchKaufmanFactor_f64" in repr(fac)
        with pytest.raises(ValueError, match="must be square"):
            mtl5.bunch_kaufman(mtl5.matrix(np.ones((3, 5))))
        with pytest.raises(ValueError, match="does not match factor size"):
            fac.solve(mtl5.vector(np.ones(3)))

    def test_available_for_universal_dtypes(self):
        """Was float32/float64 only until #73. The 1x1/2x2 block pivoting is
        comparisons and arithmetic, so no number system had to opt in."""
        fac = mtl5.bunch_kaufman(mtl5.convert(sym(4, seed=25), "posit32"))
        assert fac.dtype == "posit32"
