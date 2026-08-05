"""Smoothers, grid transfer and multigrid.

The load-bearing assertion is `TestMultigridConverges::test_the_convergence
_factor_is_multigrid_like`. A V-cycle that merely reduces the residual has not
demonstrated much — a lone smoother does that too. Multigrid's claim is a
convergence factor bounded well below 1 and roughly *independent of problem
size*, and that is what is checked. An implementation whose coarse-grid
correction is mis-scaled still converges, just slowly, which is exactly how
this was nearly missed during development.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

sp = pytest.importorskip("scipy.sparse")
import mtl5.sparse as ms  # noqa: E402

g = mtl5.mg

SMOOTHERS = [
    "jacobi",
    "gauss_seidel",
    "backward_gauss_seidel",
    "symmetric_gauss_seidel",
    "sor",
    "backward_sor",
    "symmetric_sor",
]


def poisson1d(n):
    """1-D Poisson. n = 2^k - 1 keeps the coarsening exact."""
    return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tocsr()


def residual(A, x, b):
    return float(np.linalg.norm(b - A @ x))


class TestSmoothers:
    @pytest.mark.parametrize("kind", SMOOTHERS)
    def test_each_reduces_the_residual(self, kind):
        n = 63
        A = poisson1d(n)
        b = np.ones(n)
        x0 = np.zeros(n)
        x = g.smooth(
            ms.from_scipy(A), mtl5.vector(x0), mtl5.vector(b), kind=kind, sweeps=20
        ).to_numpy()
        assert residual(A, x, b) < residual(A, x0, b)

    @pytest.mark.parametrize("kind", SMOOTHERS)
    def test_more_sweeps_do_not_hurt(self, kind):
        n = 31
        A = poisson1d(n)
        b = np.ones(n)
        few = g.smooth(
            ms.from_scipy(A),
            mtl5.vector(np.zeros(n)),
            mtl5.vector(b),
            kind=kind,
            sweeps=5,
        ).to_numpy()
        many = g.smooth(
            ms.from_scipy(A),
            mtl5.vector(np.zeros(n)),
            mtl5.vector(b),
            kind=kind,
            sweeps=50,
        ).to_numpy()
        assert residual(A, many, b) <= residual(A, few, b)

    def test_sor_with_omega_one_is_gauss_seidel(self):
        """A property of the method, so it pins that omega actually reaches the
        kernel rather than being accepted and dropped."""
        n = 31
        A = ms.from_scipy(poisson1d(n))
        b = mtl5.vector(np.ones(n))
        x0 = mtl5.vector(np.zeros(n))
        sor = g.smooth(A, x0, b, kind="sor", sweeps=10, omega=1.0).to_numpy()
        gs = g.smooth(A, x0, b, kind="gauss_seidel", sweeps=10).to_numpy()
        np.testing.assert_allclose(sor, gs, rtol=1e-12)

    def test_omega_changes_the_result(self):
        n = 31
        A = ms.from_scipy(poisson1d(n))
        b = mtl5.vector(np.ones(n))
        x0 = mtl5.vector(np.zeros(n))
        a = g.smooth(A, x0, b, kind="sor", sweeps=10, omega=1.0).to_numpy()
        c = g.smooth(A, x0, b, kind="sor", sweeps=10, omega=1.5).to_numpy()
        assert not np.allclose(a, c)

    def test_the_input_is_not_modified(self):
        n = 15
        A = ms.from_scipy(poisson1d(n))
        x0 = mtl5.vector(np.zeros(n))
        g.smooth(A, x0, mtl5.vector(np.ones(n)), sweeps=5)
        np.testing.assert_array_equal(x0.to_numpy(), np.zeros(n))

    def test_unknown_smoother(self):
        A = ms.from_scipy(poisson1d(7))
        with pytest.raises(ValueError, match="unknown smoother"):
            g.smooth(A, mtl5.vector(np.zeros(7)), mtl5.vector(np.ones(7)), kind="cheby")

    def test_sweeps_is_validated(self):
        A = ms.from_scipy(poisson1d(7))
        with pytest.raises(ValueError, match="sweeps"):
            g.smooth(A, mtl5.vector(np.zeros(7)), mtl5.vector(np.ones(7)), sweeps=0)

    def test_smoothers_are_listed(self):
        assert sorted(g.smoothers()) == sorted(SMOOTHERS)


class TestGridTransfer:
    def test_shapes_chain(self):
        R = g.make_restriction_1d(63)
        assert R.shape == (31, 63)
        P = g.make_prolongation_1d(31)
        assert P.shape == (63, 31)

    def test_restrict_then_prolongate_is_smoothing(self):
        """Not an identity — the round trip cannot recover what the coarse grid
        cannot represent. It should preserve a smooth vector approximately and
        annihilate the highest-frequency mode."""
        n = 63
        R, P = g.make_restriction_1d(n), g.make_prolongation_1d(31)
        smooth = np.sin(np.pi * np.arange(1, n + 1) / (n + 1))
        out = g.prolongate(P, g.restrict(R, mtl5.vector(smooth))).to_numpy()
        assert np.dot(out, smooth) / np.dot(smooth, smooth) > 0.5

        oscillatory = np.array([(-1.0) ** i for i in range(n)])
        killed = g.prolongate(P, g.restrict(R, mtl5.vector(oscillatory))).to_numpy()
        assert np.linalg.norm(killed) < 0.5 * np.linalg.norm(oscillatory)

    def test_dimension_mismatch(self):
        R = g.make_restriction_1d(63)
        with pytest.raises(ValueError, match="num_cols"):
            g.restrict(R, mtl5.vector(np.ones(10)))


class TestGalerkin:
    def test_matches_the_dense_triple_product(self):
        n = 63
        A = poisson1d(n)
        R, P = g.make_restriction_1d(n), g.make_prolongation_1d(31)
        Ac = g.galerkin(R, ms.from_scipy(A), P)
        ref = (ms.to_scipy(R) @ A @ ms.to_scipy(P)).toarray()
        np.testing.assert_allclose(ms.to_scipy(Ac).toarray(), ref, atol=1e-14)

    def test_the_result_stays_sparse(self):
        """The reason this exists: R @ A @ P through operator* would return a
        dense matrix, materialising a fine-sized intermediate."""
        n = 127
        Ac = g.galerkin(
            g.make_restriction_1d(n),
            ms.from_scipy(poisson1d(n)),
            g.make_prolongation_1d(63),
        )
        assert Ac.shape == (63, 63)
        assert Ac.nnz < 4 * 63, f"expected a banded coarse operator, got {Ac.nnz} nnz"

    def test_shape_mismatch_is_refused(self):
        n = 63
        with pytest.raises(ValueError, match="chain"):
            g.galerkin(
                g.make_restriction_1d(n),
                ms.from_scipy(poisson1d(31)),
                g.make_prolongation_1d(31),
            )


class TestMultigridConverges:
    def test_hierarchy_shape(self):
        M = g.multigrid_1d(ms.from_scipy(poisson1d(63)), n_levels=4)
        assert M.n == 63
        assert M.level_sizes == [63, 31, 15, 7]
        assert M.n_levels == 4
        assert "Multigrid" in repr(M)

    def test_coarsening_stops_early_rather_than_overshooting(self):
        """n_levels is an upper bound — a small matrix runs out of grid first."""
        M = g.multigrid_1d(ms.from_scipy(poisson1d(15)), n_levels=8)
        assert M.level_sizes == [15, 7, 3]

    def test_the_convergence_factor_is_multigrid_like(self):
        """The claim. A mis-scaled coarse-grid correction still converges — it
        just degrades to roughly the smoother's own rate, which during
        development looked like 0.77 per cycle rather than this."""
        n = 127
        A = poisson1d(n)
        b = np.ones(n)
        M = g.multigrid_1d(ms.from_scipy(A), n_levels=5)
        x = mtl5.vector(np.zeros(n))
        prev = residual(A, np.zeros(n), b)
        factors = []
        for _ in range(5):
            x = M.vcycle(x, mtl5.vector(b))
            cur = residual(A, x.to_numpy(), b)
            factors.append(cur / prev)
            prev = cur
        assert max(factors[1:]) < 0.2, f"not multigrid-like: {factors}"

    def test_the_factor_does_not_degrade_with_size(self):
        """Mesh independence is what separates multigrid from a smoother: the
        rate should stay put as the problem grows, not worsen with n."""

        def factor(n):
            A = poisson1d(n)
            b = np.ones(n)
            M = g.multigrid_1d(ms.from_scipy(A), n_levels=6)
            x = mtl5.vector(np.zeros(n))
            for _ in range(3):
                x = M.vcycle(x, mtl5.vector(b))
            before = residual(A, x.to_numpy(), b)
            x = M.vcycle(x, mtl5.vector(b))
            return residual(A, x.to_numpy(), b) / before

        small, large = factor(63), factor(255)
        assert large < 0.25, large
        assert large < 4 * small, (small, large)

    def test_it_actually_solves(self):
        n = 63
        A = poisson1d(n)
        rng = np.random.default_rng(0)
        x_true = rng.standard_normal(n)
        b = A @ x_true
        M = g.multigrid_1d(ms.from_scipy(A), n_levels=4)
        x = mtl5.vector(np.zeros(n))
        for _ in range(40):
            x = M.vcycle(x, mtl5.vector(b))
        np.testing.assert_allclose(x.to_numpy(), x_true, rtol=1e-6, atol=1e-6)

    def test_wcycle_also_converges(self):
        n = 63
        A = poisson1d(n)
        b = np.ones(n)
        M = g.multigrid_1d(ms.from_scipy(A), n_levels=4)
        x = M.vcycle(mtl5.vector(np.zeros(n)), mtl5.vector(b), cycles=0 + 1)
        w = M.wcycle(mtl5.vector(np.zeros(n)), mtl5.vector(b), cycles=1)
        assert residual(A, w.to_numpy(), b) < residual(A, np.zeros(n), b)
        assert residual(A, w.to_numpy(), b) <= residual(A, x.to_numpy(), b) * 1.5

    @pytest.mark.parametrize(
        "kind", ["gauss_seidel", "backward_gauss_seidel", "symmetric_gauss_seidel", "sor"]
    )
    def test_each_smoother_works_in_the_hierarchy(self, kind):
        n = 63
        A = poisson1d(n)
        b = np.ones(n)
        M = g.multigrid_1d(ms.from_scipy(A), n_levels=4, smoother=kind)
        x = M.vcycle(mtl5.vector(np.zeros(n)), mtl5.vector(b), cycles=5)
        assert residual(A, x.to_numpy(), b) < 0.1 * residual(A, np.zeros(n), b)

    def test_cycles_argument_compounds(self):
        n = 63
        A = poisson1d(n)
        b = np.ones(n)
        M = g.multigrid_1d(ms.from_scipy(A), n_levels=4)
        one = M.vcycle(mtl5.vector(np.zeros(n)), mtl5.vector(b), cycles=1)
        three = M.vcycle(mtl5.vector(np.zeros(n)), mtl5.vector(b), cycles=3)
        assert residual(A, three.to_numpy(), b) < residual(A, one.to_numpy(), b)


class TestUndampedJacobiIsAPoorSmoother:
    """Jacobi is excluded from the hierarchy test above, deliberately.

    Multigrid needs a smoother that damps *high-frequency* error, which is what
    the coarse grid cannot represent. Undamped Jacobi does not — its iteration
    matrix has eigenvalues near -1 for the highest modes, so they survive. The
    standard remedy is damped Jacobi with omega around 2/3, and MTL5's `jacobi`
    takes no damping parameter, so it cannot express that.

    Recorded rather than hidden: the exclusion is a property of the method, and
    a reader who tries `smoother='jacobi'` should find out here why it barely
    moves rather than concluding the multigrid is broken.
    """

    def test_it_barely_converges(self):
        n = 63
        A = poisson1d(n)
        b = np.ones(n)
        M = g.multigrid_1d(ms.from_scipy(A), n_levels=4, smoother="jacobi")
        x = M.vcycle(mtl5.vector(np.zeros(n)), mtl5.vector(b), cycles=5)
        ratio = residual(A, x.to_numpy(), b) / residual(A, np.zeros(n), b)
        assert ratio > 0.5, (
            f"undamped Jacobi unexpectedly smoothed well (ratio {ratio:.3f}) — "
            "if MTL5 gained a damping parameter, revisit this"
        )

    def test_gauss_seidel_on_the_same_problem_does_converge(self):
        """The control: same hierarchy, same cycles, a smoother that damps."""
        n = 63
        A = poisson1d(n)
        b = np.ones(n)
        M = g.multigrid_1d(ms.from_scipy(A), n_levels=4, smoother="gauss_seidel")
        x = M.vcycle(mtl5.vector(np.zeros(n)), mtl5.vector(b), cycles=5)
        assert residual(A, x.to_numpy(), b) < 0.01 * residual(A, np.zeros(n), b)


class TestRejections:
    def test_non_square(self):
        A = ms.from_scipy(sp.csr_matrix(np.ones((4, 6))))
        with pytest.raises(ValueError, match="square"):
            g.multigrid_1d(A)

    def test_too_few_levels(self):
        with pytest.raises(ValueError, match="n_levels"):
            g.multigrid_1d(ms.from_scipy(poisson1d(63)), n_levels=1)

    def test_too_small_to_coarsen(self):
        with pytest.raises(ValueError, match="too small"):
            g.multigrid_1d(ms.from_scipy(poisson1d(3)))

    def test_unknown_smoother_is_caught_at_build_time(self):
        """Not on the first cycle, which would be a confusing place to learn."""
        with pytest.raises(ValueError, match="unknown smoother"):
            g.multigrid_1d(ms.from_scipy(poisson1d(63)), smoother="chebyshev")

    def test_mismatched_vector(self):
        M = g.multigrid_1d(ms.from_scipy(poisson1d(63)))
        with pytest.raises(ValueError, match="match the hierarchy"):
            M.vcycle(mtl5.vector(np.zeros(10)), mtl5.vector(np.ones(10)))


class TestPublicSurface:
    def test_mg_is_exported(self):
        assert "mg" in mtl5.__all__
        assert hasattr(mtl5, "mg")

    def test_the_entry_points_exist(self):
        for name in (
            "smooth",
            "smoothers",
            "make_restriction_1d",
            "make_prolongation_1d",
            "restrict",
            "prolongate",
            "galerkin",
            "multigrid_1d",
            "Multigrid",
        ):
            assert hasattr(g, name), f"mtl5.mg.{name} missing"
