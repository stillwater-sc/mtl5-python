"""Eigenvalues, BLAS levels 2/3, and the property predicates.

Every eigen result is checked against numpy.linalg, and every BLAS routine
against the equivalent NumPy expression, because these are exactly the places
where an off-by-a-transpose or a wrong triangle produces plausible-looking
numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5


def M(a):
    return mtl5.matrix(np.ascontiguousarray(a, dtype=np.float64))


def V(a):
    return mtl5.vector(np.ascontiguousarray(a, dtype=np.float64))


def sym(n, seed=0):
    A = np.random.default_rng(seed).standard_normal((n, n))
    return (A + A.T) / 2


def spd(n, seed=0):
    A = np.random.default_rng(seed).standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


class TestEigen:
    @pytest.mark.parametrize("n", [3, 6, 12])
    def test_eigvalsh_matches_numpy(self, n):
        S = sym(n, seed=n)
        got = np.sort(mtl5.eigvalsh(M(S)))
        ref = np.sort(np.linalg.eigvalsh(S))
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("n", [3, 6, 12])
    def test_eigvals_matches_numpy(self, n):
        A = np.random.default_rng(n).standard_normal((n, n))
        got = np.sort_complex(mtl5.eigvals(M(A)))
        ref = np.sort_complex(np.linalg.eigvals(A))
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)

    def test_eigvals_returns_complex(self):
        """A rotation has a genuinely complex spectrum — the result must carry it."""
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        w = mtl5.eigvals(M(R))
        assert np.iscomplexobj(w)
        assert np.abs(np.abs(w.imag) - 1.0).max() < 1e-10

    def test_eigh_reconstructs(self):
        S = sym(10, seed=5)
        w, Q = mtl5.eigh(M(S))
        assert not np.iscomplexobj(w)
        assert np.linalg.norm(Q @ np.diag(w) @ Q.T - S) / np.linalg.norm(S) < 1e-10

    def test_eigh_eigenvectors_are_orthonormal(self):
        _, Q = mtl5.eigh(M(sym(10, seed=6)))
        assert np.linalg.norm(Q.T @ Q - np.eye(10)) < 1e-9

    def test_eig_satisfies_the_eigen_equation(self):
        A = np.random.default_rng(7).standard_normal((8, 8))
        w, Vv = mtl5.eig(M(A))
        assert np.iscomplexobj(Vv)
        assert np.linalg.norm(A @ Vv - Vv @ np.diag(w)) / np.linalg.norm(A) < 1e-8

    def test_spectral_radius(self):
        S = sym(8, seed=8)
        got = mtl5.spectral_radius(M(S))
        assert got == pytest.approx(np.abs(np.linalg.eigvals(S)).max(), rel=1e-9)

    def test_inertia_matches_eigenvalue_signs(self):
        S = sym(14, seed=9)
        eig = np.linalg.eigvalsh(S)
        got = mtl5.inertia(M(S))
        assert got["positive"] == int((eig > 0).sum())
        assert got["negative"] == int((eig < 0).sum())

    def test_is_indefinite(self):
        assert mtl5.is_indefinite(M(sym(10, seed=11)))
        assert not mtl5.is_indefinite(M(spd(10, seed=12)))

    def test_rejects_non_square(self):
        for fn in (mtl5.eigvals, mtl5.eigvalsh, mtl5.eig, mtl5.eigh):
            with pytest.raises(ValueError, match="must be square"):
                fn(M(np.ones((3, 5))))


class TestSVD:
    """Bound once stillwater-sc/mtl5#337 was fixed. These are the checks that
    caught the original defect, kept as regressions: NaN-freedom, agreement
    with LAPACK, and the sigma_max == spectral_radius identity."""

    @pytest.mark.parametrize("shape", [(4, 4), (8, 8), (10, 6), (6, 10), (20, 5)])
    def test_svdvals_matches_numpy(self, shape):
        A = np.random.default_rng(sum(shape)).standard_normal(shape)
        got = np.sort(mtl5.svdvals(M(A)))[::-1]
        ref = np.linalg.svd(A, compute_uv=False)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)

    def test_no_nan_across_many_symmetric_inputs(self):
        """The original failure was all-NaN for ~30% of symmetric matrices."""
        rng = np.random.default_rng(0)
        for trial in range(60):
            n = 3 + trial % 18
            G = rng.standard_normal((n, n))
            sv = mtl5.svdvals(M((G + G.T) / 2))
            assert np.isfinite(sv).all(), f"NaN singular values at n={n}"

    def test_sigma_max_equals_spectral_radius_for_symmetric(self):
        """The identity that needs no reference implementation."""
        rng = np.random.default_rng(1)
        for trial in range(30):
            n = 3 + trial % 15
            G = rng.standard_normal((n, n))
            S = (G + G.T) / 2
            smax = mtl5.svdvals(M(S)).max()
            assert smax == pytest.approx(mtl5.spectral_radius(M(S)), rel=1e-9)

    def test_svd_reconstructs(self):
        A = np.random.default_rng(2).standard_normal((9, 5))
        U, s_, Vm = mtl5.svd(M(A))
        assert U.shape == (9, 9) and Vm.shape == (5, 5) and s_.shape == (5,)
        Sigma = np.zeros((9, 5))
        Sigma[:5, :5] = np.diag(s_)
        assert np.linalg.norm(U @ Sigma @ Vm.T - A) / np.linalg.norm(A) < 1e-10

    @pytest.mark.parametrize("tol", [1e-8, 1e-10, 1e-12, 1e-14])
    def test_u_orthogonality_tracks_the_requested_tolerance(self, tol):
        """U is built by the iteration, so its orthogonality is bounded by tol
        rather than by machine precision — asking for more gets more.

        Measured over 160 matrices (n = 3..20, four tolerances): median ~1.4x
        tol, worst 4.5x. The 20x bound here is deliberate headroom, so treat
        the multiple as untested slack rather than a contract.
        """
        A = np.random.default_rng(3).standard_normal((8, 8))
        U, _, _ = mtl5.svd(M(A), tol)
        assert np.linalg.norm(U.T @ U - np.eye(8)) < 20 * tol

    def test_v_is_orthogonal_to_machine_precision(self):
        A = np.random.default_rng(3).standard_normal((8, 8))
        _, _, Vm = mtl5.svd(M(A))
        assert np.linalg.norm(Vm.T @ Vm - np.eye(8)) < 1e-13

    def test_condition_number_matches_numpy(self):
        for seed in (4, 5, 6):
            S = sym(8, seed=seed)
            assert mtl5.condition_number(M(S)) == pytest.approx(np.linalg.cond(S), rel=1e-8)

    def test_rcond_is_the_reciprocal(self):
        S = sym(8, seed=7)
        assert mtl5.rcond(M(S)) == pytest.approx(1.0 / mtl5.condition_number(M(S)), rel=1e-10)

    def test_numerical_rank_and_nullity(self):
        full = np.random.default_rng(8).standard_normal((6, 6))
        assert mtl5.numerical_rank(M(full)) == 6
        assert mtl5.nullity(M(full)) == 0
        deficient = np.array(full)
        deficient[:, 3] = deficient[:, 0] + deficient[:, 1]
        assert mtl5.numerical_rank(M(deficient)) == 5
        assert mtl5.nullity(M(deficient)) == 1

    def test_singular_matrix_has_infinite_condition_number(self):
        S = np.eye(4)
        S[2, 2] = 0.0
        assert not np.isfinite(mtl5.condition_number(M(S)))
        assert mtl5.rcond(M(S)) == pytest.approx(0.0)


class TestBlasLevel2:
    def test_ger(self):
        rng = np.random.default_rng(0)
        A0 = rng.standard_normal((5, 7))
        x, y = rng.standard_normal(5), rng.standard_normal(7)
        A = np.array(A0)
        mtl5.ger(2.5, V(x), V(y), M(A))
        np.testing.assert_allclose(A, A0 + 2.5 * np.outer(x, y), rtol=1e-12)

    def test_symv(self):
        rng = np.random.default_rng(1)
        S = sym(6, seed=1)
        x, y0 = rng.standard_normal(6), rng.standard_normal(6)
        y = np.array(y0)
        mtl5.symv(1.5, M(S), V(x), 0.5, V(y))
        np.testing.assert_allclose(y, 1.5 * (S @ x) + 0.5 * y0, rtol=1e-11)

    @pytest.mark.parametrize("upper", [True, False])
    def test_trmv(self, upper):
        rng = np.random.default_rng(2)
        A = rng.standard_normal((6, 6))
        Tri = np.triu(A) if upper else np.tril(A)
        x0 = rng.standard_normal(6)
        x = np.array(x0)
        mtl5.trmv(M(A), V(x), upper=upper)
        np.testing.assert_allclose(x, Tri @ x0, rtol=1e-11)

    @pytest.mark.parametrize("upper", [True, False])
    def test_trsv_inverts_trmv(self, upper):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((6, 6)) + 6 * np.eye(6)
        Tri = np.triu(A) if upper else np.tril(A)
        b = rng.standard_normal(6)
        x = np.array(b)
        mtl5.trsv(M(A), V(x), upper=upper)
        np.testing.assert_allclose(x, np.linalg.solve(Tri, b), rtol=1e-10)

    def test_trsv_unit_diagonal(self):
        rng = np.random.default_rng(4)
        A = rng.standard_normal((5, 5))
        Tri = np.tril(A, -1) + np.eye(5)
        b = rng.standard_normal(5)
        x = np.array(b)
        mtl5.trsv(M(A), V(x), upper=False, unit_diag=True)
        np.testing.assert_allclose(x, np.linalg.solve(Tri, b), rtol=1e-10)


class TestBlasLevel3:
    def test_trmm(self):
        rng = np.random.default_rng(5)
        A = rng.standard_normal((5, 5))
        B0 = rng.standard_normal((5, 4))
        B = np.array(B0)
        mtl5.trmm(2.0, M(A), M(B), upper=True)
        np.testing.assert_allclose(B, 2.0 * (np.triu(A) @ B0), rtol=1e-11)

    def test_trsm_inverts_trmm(self):
        rng = np.random.default_rng(6)
        A = rng.standard_normal((5, 5)) + 5 * np.eye(5)
        B0 = rng.standard_normal((5, 3))
        B = np.array(B0)
        mtl5.trsm(1.0, M(A), M(B), upper=True)
        np.testing.assert_allclose(B, np.linalg.solve(np.triu(A), B0), rtol=1e-10)

    def test_symm(self):
        rng = np.random.default_rng(7)
        S = sym(5, seed=7)
        B = rng.standard_normal((5, 4))
        C0 = rng.standard_normal((5, 4))
        C = np.array(C0)
        mtl5.symm(1.5, M(S), M(B), 0.5, M(C))
        np.testing.assert_allclose(C, 1.5 * (S @ B) + 0.5 * C0, rtol=1e-11)

    def test_syrk_writes_both_triangles(self):
        rng = np.random.default_rng(8)
        A = rng.standard_normal((5, 3))
        C0 = sym(5, seed=8)
        C = np.array(C0)
        mtl5.syrk(2.0, M(A), 0.5, M(C))
        expected = 2.0 * (A @ A.T) + 0.5 * C0
        np.testing.assert_allclose(C, expected, rtol=1e-11)
        np.testing.assert_allclose(C, C.T, rtol=1e-12)

    def test_syr2k(self):
        rng = np.random.default_rng(9)
        A = rng.standard_normal((5, 3))
        B = rng.standard_normal((5, 3))
        C0 = sym(5, seed=9)
        C = np.array(C0)
        mtl5.syr2k(1.5, M(A), M(B), 0.5, M(C))
        expected = 1.5 * (A @ B.T + B @ A.T) + 0.5 * C0
        np.testing.assert_allclose(C, expected, rtol=1e-11)

    def test_shape_checks(self):
        with pytest.raises(ValueError, match="dimension mismatch|must be square"):
            mtl5.symv(1.0, M(np.ones((3, 4))), V(np.ones(4)), 0.0, V(np.ones(3)))
        with pytest.raises(ValueError, match="same shape"):
            mtl5.syr2k(1.0, M(np.ones((4, 3))), M(np.ones((4, 2))), 0.0, M(np.ones((4, 4))))


class TestStructuralPredicates:
    def test_square_and_empty(self):
        assert mtl5.is_square(M(np.eye(4)))
        assert not mtl5.is_square(M(np.ones((3, 4))))

    def test_symmetry(self):
        assert mtl5.is_symmetric(M(sym(6, seed=1)))
        assert not mtl5.is_symmetric(M(np.random.default_rng(0).standard_normal((6, 6))))

    def test_triangular_family(self):
        A = np.random.default_rng(2).standard_normal((6, 6))
        assert mtl5.is_upper_triangular(M(np.triu(A)))
        assert mtl5.is_lower_triangular(M(np.tril(A)))
        assert mtl5.is_triangular(M(np.triu(A)))
        assert not mtl5.is_triangular(M(A))
        assert mtl5.is_diagonal(M(np.diag(np.diag(A))))

    def test_banded(self):
        B = np.diag(np.ones(6)) + np.diag(np.ones(5), 1) + np.diag(np.ones(5), -1)
        assert mtl5.is_banded(M(B), 1, 1)
        assert not mtl5.is_banded(M(B), 0, 0)

    def test_diagonally_dominant(self):
        D = np.array([[5.0, 1.0], [1.0, 4.0]])
        assert mtl5.is_diagonally_dominant(M(D))
        assert not mtl5.is_diagonally_dominant(M(np.array([[1.0, 3.0], [3.0, 1.0]])))

    def test_orthogonal(self):
        Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((6, 6)))
        assert mtl5.is_orthogonal(M(Q))
        assert not mtl5.is_orthogonal(M(2.0 * Q))
        assert mtl5.is_normal(M(Q))


class TestFactorizationPredicates:
    def test_spd(self):
        assert mtl5.is_spd(M(spd(8, seed=1)))
        assert not mtl5.is_spd(M(sym(8, seed=1)))

    def test_singularity(self):
        A = np.eye(5)
        assert mtl5.is_nonsingular(M(A))
        assert mtl5.is_invertible(M(A))
        S = np.eye(5)
        S[2, 2] = 0.0
        assert mtl5.is_singular(M(S))


class TestVectorPredicates:
    def test_zero_and_finite(self):
        assert mtl5.is_zero(V(np.zeros(5)))
        assert not mtl5.is_zero(V(np.array([0.0, 1e-9, 0.0, 0.0, 0.0])))
        assert mtl5.is_finite(V(np.ones(4)))
        assert mtl5.has_nan(V(np.array([1.0, np.nan])))
        assert mtl5.has_inf(V(np.array([1.0, np.inf])))
        assert not mtl5.is_finite(V(np.array([1.0, np.inf])))

    def test_normalized(self):
        v = np.array([3.0, 4.0])
        assert not mtl5.is_normalized(V(v))
        assert mtl5.is_normalized(V(v / np.linalg.norm(v)))
        assert mtl5.is_unit(V(v / np.linalg.norm(v)))

    def test_orthogonal_to(self):
        assert mtl5.is_orthogonal_to(V(np.array([1.0, 0.0])), V(np.array([0.0, 2.0])))
        assert not mtl5.is_orthogonal_to(V(np.array([1.0, 1.0])), V(np.array([1.0, 0.0])))

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            mtl5.is_orthogonal_to(V(np.ones(3)), V(np.ones(4)))
