"""Sparse direct solvers: sparse LU, native KLU, orderings, and refactorization.

The properties worth pinning are the ones that distinguish this from
scipy.sparse.linalg.splu: refactorization reuses the analysis and stays
correct, a fill-reducing ordering actually reduces fill, and a narrow factor
refined against a float64 residual recovers full accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

sp = pytest.importorskip("scipy.sparse")

import mtl5.sparse as ms  # noqa: E402


def laplacian_2d(k: int):
    """k x k 5-point Laplacian — structured, SPD, realistic fill."""
    eye = sp.eye(k, format="csr")
    tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k), format="csr")
    return (sp.kron(eye, tri) + sp.kron(tri, eye)).tocsr()


def perturbed(A, seed: int):
    """Same sparsity pattern, different values — and symmetry preserved.

    The scaling is a function of (row + col), which is symmetric, so a
    symmetric A stays symmetric. Scaling by linear index instead would quietly
    break symmetry, and the Cholesky/LDL^T factorizations would then solve a
    different (implicitly symmetrized) system than A2 @ x produced.
    """
    B = A.tocoo()
    factor = 1.0 + 0.01 * np.sin(seed * (B.row + B.col).astype(np.float64))
    return sp.coo_matrix((B.data * factor, (B.row, B.col)), shape=A.shape).tocsr()


@pytest.fixture
def system():
    A = laplacian_2d(20)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(A.shape[0])
    return A, x, A @ x


class TestSparseLU:
    def test_solves_accurately(self, system):
        A, xt, b = system
        lu = ms.splu(A)
        x = lu.solve(b).to_numpy()
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-12

    def test_all_orderings_solve(self, system):
        A, xt, b = system
        for name in ms.orderings():
            lu = ms.splu(A, ordering=name)
            assert lu.ordering == name
            x = lu.solve(b).to_numpy()
            assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-12, name

    def test_fill_reducing_ordering_reduces_fill(self, system):
        """AMD must beat the natural ordering on a 2-D Laplacian, or the
        ordering is not wired to the symbolic analysis."""
        A, _, _ = system
        assert ms.splu(A, ordering="amd").nnz < ms.splu(A, ordering="natural").nnz

    def test_clean_factor_reports_no_perturbation(self, system):
        A, _, _ = system
        assert ms.splu(A).num_perturbed == 0

    def test_metadata(self, system):
        A, _, _ = system
        lu = ms.splu(A, ordering="amd")
        assert lu.n == A.shape[0]
        assert lu.shape == A.shape
        assert lu.dtype == "f64"
        assert lu.nnz > A.nnz  # a factorization fills in
        assert "SparseLU_f64" in repr(lu)

    def test_float32_factor(self, system):
        A, xt, b = system
        lu = ms.splu(A.astype(np.float32))
        assert lu.dtype == "f32"
        x = np.asarray(lu.solve(b).to_numpy(), dtype=np.float64)
        err = np.linalg.norm(x - xt) / np.linalg.norm(xt)
        # float32 alone lands around 1e-6 — good enough to be a useful factor,
        # far from float64. That gap is what refinement closes below.
        assert 1e-9 < err < 1e-3

    def test_rejects_non_square(self):
        A = sp.random(4, 6, density=0.5, format="csr", random_state=0)
        with pytest.raises(ValueError, match="must be square"):
            ms.splu(A)

    def test_rejects_unknown_ordering(self, system):
        A, _, _ = system
        with pytest.raises(ValueError, match="unknown ordering"):
            ms.splu(A, ordering="metis")

    def test_rhs_length_checked(self, system):
        A, _, _ = system
        lu = ms.splu(A)
        with pytest.raises(ValueError, match="does not match factor size"):
            lu.solve(np.ones(3))


class TestRefactor:
    def test_refactor_matches_a_full_factor(self, system):
        """The whole premise: reusing the analysis must not change the answer."""
        A, _, _ = system
        A2 = perturbed(A, 7)
        rng = np.random.default_rng(1)
        xt = rng.standard_normal(A.shape[0])
        b = A2 @ xt

        lu = ms.splu(A, ordering="amd")
        lu.refactor(ms.from_scipy(A2))
        refactored = lu.solve(b).to_numpy()

        fresh = ms.splu(A2, ordering="amd").solve(b).to_numpy()
        np.testing.assert_allclose(refactored, fresh, rtol=1e-10)
        assert np.linalg.norm(refactored - xt) / np.linalg.norm(xt) < 1e-12

    def test_repeated_refactor(self, system):
        A, _, _ = system
        lu = ms.splu(A, ordering="amd")
        rng = np.random.default_rng(2)
        for seed in range(1, 4):
            A2 = perturbed(A, seed)
            xt = rng.standard_normal(A.shape[0])
            lu.refactor(ms.from_scipy(A2))
            x = lu.solve(A2 @ xt).to_numpy()
            assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-11, seed

    def test_refactor_preserves_the_ordering(self, system):
        A, _, _ = system
        lu = ms.splu(A, ordering="amd")
        nnz_before = lu.nnz
        lu.refactor(ms.from_scipy(perturbed(A, 3)))
        assert lu.ordering == "amd"
        assert lu.nnz == nnz_before  # same pattern => same fill

    def test_refactor_rejects_wrong_size(self, system):
        A, _, _ = system
        lu = ms.splu(A)
        smaller = laplacian_2d(5)
        with pytest.raises(ValueError, match="do not match"):
            lu.refactor(ms.from_scipy(smaller))

    def test_klu_refactor(self, system):
        A, _, _ = system
        A2 = perturbed(A, 11)
        rng = np.random.default_rng(3)
        xt = rng.standard_normal(A.shape[0])
        k = ms.klu(A)
        k.refactor(ms.from_scipy(A2))
        x = k.solve(A2 @ xt).to_numpy()
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-11


class TestKLU:
    def test_solves_accurately(self, system):
        A, xt, b = system
        x = ms.klu(A).solve(b).to_numpy()
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-12

    def test_block_triangular_form_is_found(self):
        """A block-diagonal matrix is maximally reducible; KLU must see it."""
        blocks = [sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 3.0]])) for _ in range(6)]
        A = sp.block_diag(blocks, format="csr")
        assert ms.klu(A).nblocks >= 6

    def test_irreducible_matrix_is_one_block(self):
        A = laplacian_2d(6)
        assert ms.klu(A).nblocks == 1

    def test_unscaled_variant(self, system):
        A, xt, b = system
        x = ms.klu(A, scale=False).solve(b).to_numpy()
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-12

    def test_metadata(self, system):
        A, _, _ = system
        k = ms.klu(A)
        assert k.n == A.shape[0]
        assert k.dtype == "f64"
        assert k.num_perturbed == 0
        assert "KLU_f64" in repr(k)


class TestOrderings:
    def test_all_are_valid_permutations(self, system):
        A, _, _ = system
        n = A.shape[0]
        for name in ms.orderings():
            p = ms.ordering(A, name)
            assert p.shape == (n,)
            assert sorted(p.tolist()) == list(range(n)), name

    def test_named_helpers_agree(self, system):
        A, _, _ = system
        np.testing.assert_array_equal(ms.amd(A), ms.ordering(A, "amd"))
        np.testing.assert_array_equal(ms.colamd(A), ms.ordering(A, "colamd"))
        np.testing.assert_array_equal(ms.rcm(A), ms.ordering(A, "rcm"))

    def test_natural_is_the_identity(self, system):
        A, _, _ = system
        np.testing.assert_array_equal(ms.ordering(A, "natural"), np.arange(A.shape[0]))

    def test_rcm_reduces_bandwidth(self):
        """RCM on a deliberately scrambled banded matrix must recover locality."""
        n = 200
        band = sp.diags([1.0, 4.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")
        rng = np.random.default_rng(0)
        q = rng.permutation(n)
        scrambled = band[q][:, q].tocsr()

        def bandwidth(M):
            M = M.tocoo()
            return int(np.abs(M.row - M.col).max())

        p = ms.rcm(scrambled)
        assert bandwidth(scrambled[p][:, p].tocsr()) < bandwidth(scrambled)

    def test_unknown_ordering(self, system):
        A, _, _ = system
        with pytest.raises(ValueError, match="unknown ordering"):
            ms.ordering(A, "nested_dissection")


class TestMixedPrecisionDirectSolve:
    """The pairing that has no SciPy equivalent: factor narrow, refine wide."""

    @pytest.fixture
    def system(self):
        A = laplacian_2d(24)
        rng = np.random.default_rng(5)
        x = rng.standard_normal(A.shape[0])
        return A, x, A @ x

    def test_float32_lu_refined_reaches_float64_accuracy(self, system):
        A, xt, b = system
        Ad = ms.from_scipy(A)
        lu32 = ms.splu(A.astype(np.float32), ordering="amd")

        raw = np.asarray(lu32.solve(b).to_numpy(), dtype=np.float64)
        raw_err = np.linalg.norm(raw - xt) / np.linalg.norm(xt)

        x, info = mtl5.mixed.iterative_refine(Ad, lu32, b, max_iter=50, rel_tol=1e-14)
        err = np.linalg.norm(x - xt) / np.linalg.norm(xt)

        assert info["converged"]
        assert info["iters"] > 0
        assert err < 1e-13
        assert err < raw_err / 1e5, "refinement must gain several orders"

    def test_float32_klu_refined(self, system):
        A, xt, b = system
        Ad = ms.from_scipy(A)
        k32 = ms.klu(A.astype(np.float32))
        x, info = mtl5.mixed.iterative_refine(Ad, k32, b, max_iter=50, rel_tol=1e-14)
        assert info["converged"]
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-13

    def test_float64_factor_needs_no_refinement(self, system):
        A, xt, b = system
        Ad = ms.from_scipy(A)
        x, info = mtl5.mixed.iterative_refine(Ad, ms.splu(A), b, max_iter=50, rel_tol=1e-14)
        assert info["iters"] <= 2
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-13

    def test_factor_size_must_match(self, system):
        A, _, b = system
        Ad = ms.from_scipy(A)
        wrong = ms.splu(laplacian_2d(6))
        with pytest.raises(ValueError, match="does not match A"):
            mtl5.mixed.iterative_refine(Ad, wrong, b)


# ===========================================================================
# The symmetric and least-squares factorizations
# ===========================================================================

ALL_SQUARE = ["splu", "klu", "cholesky", "ldlt", "supernodal_lu", "supernodal_ldlt"]


class TestSymmetricFactorizations:
    @pytest.mark.parametrize("kind", ALL_SQUARE)
    def test_every_square_factorization_solves(self, kind, system):
        A, xt, b = system
        fac = getattr(ms, kind)(A)
        x = fac.solve(b).to_numpy()
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-11, kind

    @pytest.mark.parametrize("kind", ALL_SQUARE)
    def test_every_square_factorization_refactors(self, kind, system):
        """refactor must agree with a fresh factorization of the same matrix."""
        A, _, _ = system
        A2 = perturbed(A, 5)
        rng = np.random.default_rng(9)
        xt = rng.standard_normal(A.shape[0])
        b = A2 @ xt

        fac = getattr(ms, kind)(A)
        fac.refactor(ms.from_scipy(A2))
        got = fac.solve(b).to_numpy()

        fresh = getattr(ms, kind)(A2).solve(b).to_numpy()
        np.testing.assert_allclose(got, fresh, rtol=1e-9, err_msg=kind)
        assert np.linalg.norm(got - xt) / np.linalg.norm(xt) < 1e-11, kind

    def test_cholesky_is_sparser_than_lu(self, system):
        """Exploiting symmetry must pay off in fill, or it is not doing so."""
        A, _, _ = system
        assert ms.cholesky(A, ordering="amd").nnz < ms.splu(A, ordering="amd").nnz

    def test_cholesky_rejects_indefinite(self):
        A = sp.csr_matrix(np.diag([1.0, -2.0, 3.0]))
        with pytest.raises(RuntimeError):
            ms.cholesky(A)

    def test_ldlt_diagonal_reveals_inertia(self):
        """LDL^T handles what Cholesky refuses, and D's signs say why."""
        A = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, -3.0]]))
        fac = ms.ldlt(A)
        d = fac.diagonal()
        assert d.shape == (2,)
        assert (d < 0).any(), "an indefinite matrix must show a negative pivot"

    def test_supernodal_lu_forms_supernodes(self, system):
        A, _, _ = system
        assert ms.supernodal_lu(A).nsuper > 0

    def test_supernodal_lu_rejects_natural_ordering(self, system):
        A, _, _ = system
        with pytest.raises(ValueError, match="must be 'amd', 'colamd' or 'rcm'"):
            ms.supernodal_lu(A, ordering="natural")

    def test_supernodal_lu_scaled_variant(self, system):
        A, xt, b = system
        x = ms.supernodal_lu(A, scale=True).solve(b).to_numpy()
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-11

    @pytest.mark.parametrize("kind", ["cholesky", "ldlt", "supernodal_ldlt"])
    def test_symmetric_factorizations_reject_non_square(self, kind):
        A = sp.random(4, 6, density=0.5, format="csr", random_state=0)
        with pytest.raises(ValueError, match="must be square"):
            getattr(ms, kind)(A)


class TestSparseQR:
    @pytest.fixture
    def lstsq(self):
        rng = np.random.default_rng(4)
        m, n = 300, 60
        A = sp.random(m, n, density=0.06, format="csr", random_state=3)
        A = (A + sp.eye(m, n, format="csr")).tocsr()
        b = rng.standard_normal(m)
        return A, b

    def test_matches_numpy_lstsq(self, lstsq):
        A, b = lstsq
        x = np.asarray(ms.qr(A).solve(b).to_numpy(), dtype=np.float64)
        ref = np.linalg.lstsq(A.toarray(), b, rcond=None)[0]
        assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-9

    def test_square_system(self, system):
        A, xt, b = system
        x = np.asarray(ms.qr(A).solve(b).to_numpy(), dtype=np.float64)
        assert np.linalg.norm(x - xt) / np.linalg.norm(xt) < 1e-10

    def test_shape_and_metadata(self, lstsq):
        A, _ = lstsq
        fac = ms.qr(A)
        assert fac.shape == A.shape
        assert fac.dtype == "f64"
        assert fac.nnz > 0
        assert "SparseQR_f64" in repr(fac)

    def test_refactor(self, lstsq):
        A, b = lstsq
        A2 = perturbed(A, 13)
        fac = ms.qr(A)
        fac.refactor(ms.from_scipy(A2))
        x = np.asarray(fac.solve(b).to_numpy(), dtype=np.float64)
        ref = np.linalg.lstsq(A2.toarray(), b, rcond=None)[0]
        assert np.linalg.norm(x - ref) / np.linalg.norm(ref) < 1e-9

    def test_rejects_underdetermined(self):
        A = sp.random(20, 50, density=0.2, format="csr", random_state=0)
        with pytest.raises(ValueError, match="num_rows >= num_cols"):
            ms.qr(A)

    def test_rhs_length_checked(self, lstsq):
        A, _ = lstsq
        with pytest.raises(ValueError, match="does not match the row count"):
            ms.qr(A).solve(np.ones(7))


class TestRefineThroughEveryFactorization:
    """Every square factorization must be usable as a refinement factor, and a
    float32 one must recover float64 accuracy."""

    @pytest.fixture
    def system(self):
        A = laplacian_2d(18)
        rng = np.random.default_rng(6)
        x = rng.standard_normal(A.shape[0])
        return A, x, A @ x

    @pytest.mark.parametrize("kind", ALL_SQUARE)
    def test_float32_factor_refines_to_float64(self, kind, system):
        A, xt, b = system
        Ad = ms.from_scipy(A)
        fac32 = getattr(ms, kind)(A.astype(np.float32))
        x, info = mtl5.mixed.iterative_refine(Ad, fac32, b, max_iter=50, rel_tol=1e-14)
        err = np.linalg.norm(x - xt) / np.linalg.norm(xt)
        assert info["converged"], kind
        assert err < 1e-12, f"{kind}: {err:.2e}"
