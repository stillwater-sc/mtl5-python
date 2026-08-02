"""Test-matrix generators and range vectors.

These are checked against the property each matrix is *named for* — Clement's
exact integer spectrum, Hilbert's conditioning growth, randspd's prescribed
eigenvalues — rather than against a stored reference. A generator that produced
plausible-looking numbers with the wrong structure would be worse than useless,
since its whole job is to be a known-pathology input.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

g = mtl5.generators


class TestKnownSpectra:
    def test_clement_has_exact_integer_eigenvalues(self):
        """Clement(n) has eigenvalues -(n-1), -(n-3), ..., n-1 exactly."""
        for n in (5, 7, 10):
            ev = np.sort(np.linalg.eigvals(g.clement(n).to_numpy()).real)
            expected = np.arange(-(n - 1), n, 2, dtype=float)
            np.testing.assert_allclose(ev, expected, atol=1e-9)

    def test_wilkinson_diagonal_is_the_v_shape(self):
        d = np.diag(g.wilkinson(5).to_numpy())
        np.testing.assert_allclose(d, [2, 1, 0, 1, 2])

    def test_wilkinson_rejects_even_n(self):
        with pytest.raises(ValueError, match="must be odd"):
            g.wilkinson(6)

    def test_rosser_is_8x8_and_singular(self):
        R = g.rosser().to_numpy()
        assert R.shape == (8, 8)
        assert np.abs(np.linalg.eigvals(R)).min() < 1e-9, "Rosser has a zero eigenvalue"

    def test_magic_rows_columns_and_diagonals_agree(self):
        for n in (3, 5, 8):
            M = g.magic(n).to_numpy()
            target = M.sum(axis=1)[0]
            np.testing.assert_allclose(M.sum(axis=1), target)
            np.testing.assert_allclose(M.sum(axis=0), target)
            assert np.trace(M) == pytest.approx(target)

    def test_magic_rejects_singly_even(self):
        with pytest.raises((ValueError, RuntimeError)):
            g.magic(6)


class TestConditioning:
    def test_hilbert_conditioning_grows(self):
        conds = [np.linalg.cond(g.hilbert(n).to_numpy()) for n in (4, 6, 8)]
        assert conds[0] < conds[1] < conds[2]
        assert conds[0] > 1e3, "even Hilbert(4) is badly conditioned"

    def test_hilbert_entries(self):
        H = g.hilbert(4).to_numpy()
        expected = 1.0 / (np.arange(4)[:, None] + np.arange(4)[None, :] + 1)
        np.testing.assert_allclose(H, expected, rtol=1e-15)

    def test_minij_is_spd(self):
        M = g.minij(6).to_numpy()
        np.testing.assert_allclose(M, M.T)
        assert (np.linalg.eigvalsh(M) > 0).all()

    def test_lehmer_is_spd_and_symmetric(self):
        L = g.lehmer(6).to_numpy()
        np.testing.assert_allclose(L, L.T)
        assert (np.linalg.eigvalsh(L) > 0).all()

    def test_pascal_is_spd(self):
        P = g.pascal(5).to_numpy()
        np.testing.assert_allclose(P, P.T)
        assert (np.linalg.eigvalsh(P) > 0).all()

    def test_ones(self):
        np.testing.assert_allclose(g.ones(3).to_numpy(), np.ones((3, 3)))
        np.testing.assert_allclose(g.ones(2, 5).to_numpy(), np.ones((2, 5)))


class TestPrescribedStructure:
    def test_randspd_has_exactly_the_requested_spectrum(self):
        target = [1.0, 10.0, 100.0, 1000.0]
        A = g.randspd(4, target).to_numpy()
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(A)), target, rtol=1e-9)
        np.testing.assert_allclose(A, A.T, atol=1e-12)

    def test_randspd_rejects_nonpositive_eigenvalues(self):
        with pytest.raises(ValueError, match="must be positive"):
            g.randspd(3, [1.0, -2.0, 3.0])

    def test_randspd_rejects_wrong_count(self):
        with pytest.raises(ValueError, match="exactly n eigenvalues"):
            g.randspd(3, [1.0, 2.0])

    def test_randsym_allows_an_indefinite_spectrum(self):
        target = [-5.0, -1.0, 2.0, 7.0]
        A = g.randsym(4, target).to_numpy()
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(A)), target, rtol=1e-9)

    def test_randsvd_hits_the_requested_condition_number(self):
        for kappa in (1e3, 1e6, 1e9):
            A = g.randsvd(20, 20, kappa).to_numpy()
            assert np.linalg.cond(A) == pytest.approx(kappa, rel=1e-6)

    def test_randsvd_rejects_kappa_below_one(self):
        with pytest.raises(ValueError, match="kappa must be >= 1"):
            g.randsvd(4, 4, 0.5)

    def test_randorth_is_orthogonal(self):
        Q = g.randorth(10).to_numpy()
        assert np.linalg.norm(Q.T @ Q - np.eye(10)) < 1e-12
        assert abs(abs(np.linalg.det(Q)) - 1.0) < 1e-12

    def test_companion_eigenvalues_are_the_polynomial_roots(self):
        # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        coeffs = [-6.0, 11.0, -6.0]
        ev = np.sort(np.linalg.eigvals(g.companion(coeffs).to_numpy()).real)
        np.testing.assert_allclose(ev, [1.0, 2.0, 3.0], atol=1e-9)

    def test_vandermonde_structure(self):
        nodes = [1.0, 2.0, 3.0]
        V = g.vandermonde(nodes).to_numpy()
        assert V.shape == (3, 3)
        # Some column (or row) must be the plain powers of the nodes.
        powers = np.vander(nodes, 3, increasing=True)
        assert np.allclose(V, powers) or np.allclose(V, powers.T)


class TestParameterized:
    def test_forsythe_alpha_controls_the_spectrum(self):
        """Eigenvalues are the n-th roots of alpha, so they shrink with alpha."""
        big = np.abs(np.linalg.eigvals(g.forsythe(5, 1e-2).to_numpy())).max()
        small = np.abs(np.linalg.eigvals(g.forsythe(5, 1e-10).to_numpy())).max()
        assert small < big

    def test_kahan_is_upper_triangular(self):
        K = g.kahan(6).to_numpy()
        assert np.allclose(np.tril(K, -1), 0.0)

    def test_moler_is_spd_with_a_small_eigenvalue(self):
        M = g.moler(8).to_numpy()
        ev = np.linalg.eigvalsh(M)
        assert (ev > 0).all()
        assert ev.min() / ev.max() < 1e-3


class TestSparseGenerators:
    def test_laplacian_1d(self):
        L = g.laplacian_1d(10)
        assert L.shape == (10, 10)
        import mtl5.sparse as ms

        dense = ms.to_scipy(L).toarray()
        np.testing.assert_allclose(np.diag(dense), 2.0)
        np.testing.assert_allclose(np.diag(dense, 1), -1.0)

    def test_laplacian_2d_shape_and_spd(self):
        import mtl5.sparse as ms

        L = g.laplacian_2d(6, 6)
        assert L.shape == (36, 36)
        assert (np.linalg.eigvalsh(ms.to_scipy(L).toarray()) > 0).all()

    def test_poisson2d(self):
        L = g.poisson2d(5, 5)
        assert L.shape == (25, 25)

    def test_sparse_generators_reject_dtype(self):
        with pytest.raises(TypeError, match="not supported for the sparse generators"):
            g.laplacian_1d(4, dtype="posit16")


class TestRanges:
    def test_linspace_matches_numpy(self):
        np.testing.assert_allclose(mtl5.linspace(0.0, 1.0, 5).to_numpy(), np.linspace(0.0, 1.0, 5))

    def test_linspace_endpoint_false(self):
        np.testing.assert_allclose(
            mtl5.linspace(0.0, 1.0, 5, endpoint=False).to_numpy(),
            np.linspace(0.0, 1.0, 5, endpoint=False),
        )

    def test_arange_matches_numpy(self):
        np.testing.assert_allclose(mtl5.arange(0, 10, 3).to_numpy(), np.arange(0, 10, 3))

    def test_arange_rejects_zero_step(self):
        with pytest.raises(ValueError, match="step must be nonzero"):
            mtl5.arange(0, 10, 0)

    def test_logspace_matches_numpy(self):
        np.testing.assert_allclose(mtl5.logspace(0.0, 3.0, 4).to_numpy(), np.logspace(0.0, 3.0, 4))

    def test_geomspace_is_a_true_geometric_progression(self):
        """MTL5 fixed Universal's bug where geomspace aliased logspace and read
        its endpoints as exponents — geomspace(1, 1000, 4) is 1,10,100,1000."""
        got = mtl5.geomspace(1.0, 1000.0, 4).to_numpy()
        np.testing.assert_allclose(got, [1.0, 10.0, 100.0, 1000.0], rtol=1e-12)
        np.testing.assert_allclose(got, np.geomspace(1.0, 1000.0, 4), rtol=1e-12)

    def test_geomspace_rejects_zero_endpoint(self):
        with pytest.raises(ValueError, match="must be nonzero"):
            mtl5.geomspace(0.0, 10.0, 4)


class TestDtype:
    @pytest.mark.parametrize("dt", ["f32", "posit16", "posit32", "fp16"])
    def test_dense_generators_honour_dtype(self, dt):
        H = g.hilbert(6, dtype=dt)
        assert H.dtype == dt
        assert H.shape == (6, 6)

    def test_dtype_rounds_rather_than_recomputing(self):
        """The dtype result must equal the float64 result rounded to dtype."""
        exact = g.hilbert(6).to_numpy()
        rounded = g.hilbert(6, dtype="posit16").to_numpy()
        expected = mtl5.convert(exact, "posit16").to_numpy()
        np.testing.assert_array_equal(rounded, expected)

    def test_ranges_honour_dtype(self):
        v = mtl5.linspace(0.0, 1.0, 5, dtype="posit16")
        assert v.dtype == "posit16"
        np.testing.assert_allclose(v.to_numpy(), [0, 0.25, 0.5, 0.75, 1.0])

    def test_narrow_dtype_actually_loses_precision(self):
        """Otherwise dtype= is not doing anything."""
        exact = g.hilbert(8).to_numpy()
        narrow = g.hilbert(8, dtype="fp8").to_numpy()
        assert not np.allclose(exact, narrow)


class TestCatalog:
    def test_names_and_kappa(self):
        names = g.testsuite_names()
        assert len(names) > 10
        assert "lambers_ill" in names
        assert g.testsuite_kappa("lambers_ill") > 1e7

    def test_unknown_name_raises(self):
        with pytest.raises(RuntimeError, match="unknown matrix"):
            g.testsuite_kappa("not_a_matrix")

    def test_by_name_is_not_exposed(self):
        """It reads .mtx files from the MTL5 source tree, which a wheel lacks."""
        assert not hasattr(g, "by_name")
