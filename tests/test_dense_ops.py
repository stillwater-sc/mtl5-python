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

    def test_svd_is_not_exposed(self):
        """Withheld pending stillwater-sc/mtl5#337 — see the module docstring."""
        for name in ("svd", "svdvals", "condition_number", "rcond", "numerical_rank", "nullity"):
            assert not hasattr(mtl5, name), f"{name} should not be exposed while #337 is open"


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
