"""Krylov solver history: `callback=` and `full_output=` (#147).

The solvers run in C++ without the GIL, so the history is recorded by an
iteration controller in the bindings (`recording_iteration` in
python/src/mtl5_krylov.cpp), which every MTL5 solver reports to through
`finished()`. These tests pin what a caller can rely on:

  * one history entry per iteration, from iteration 0 (||b||, since x0 = 0),
    padded with NaN after a breakdown;
  * the report's residual is the true ||b - A x||, and on convergence the
    relative residual is at most rtol; unpreconditioned, the history ends on
    it too;
  * gmres's history is the preconditioned residual ||M^-1 r||, which is what
    MTL5's left-preconditioned GMRES tests;
  * the callback sees the same values as the history, in iteration order;
  * an exception raised in the callback stops the solve and propagates;
  * the default return stays SciPy's (x, info).
"""

from __future__ import annotations

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")
import mtl5.sparse as ms  # noqa: E402

SYMMETRIC = ["cg", "minres"]
ALL = sorted(ms.solvers())


def system(nonsym):
    """A 2-D Laplacian on a 12 x 12 grid, skewed when `nonsym`."""
    k = 12
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k))
    A = sp.kron(sp.eye(k), T) + sp.kron(T, sp.eye(k))
    if nonsym:
        A = A + sp.diags([0.3], [1], shape=A.shape)
    b = np.linspace(1.0, 2.0, A.shape[0])
    return A.tocsr(), b


@pytest.mark.parametrize("solver", ALL)
class TestReport:
    def solve(self, solver, **kwargs):
        A, b = system(nonsym=solver not in SYMMETRIC)
        x, info, report = ms.iterative_solve(
            A, b, solver=solver, rtol=1e-10, maxiter=2000, full_output=True, **kwargs
        )
        return A, b, x, info, report

    def test_one_entry_per_iteration(self, solver):
        _, b, _, info, report = self.solve(solver)
        assert info == 0
        assert report.iterations > 0
        assert len(report.history) == report.iterations + 1
        assert not np.isnan(report.history).any()
        assert report.history[0] == pytest.approx(np.linalg.norm(b), rel=1e-14)
        assert report.history[-1] == pytest.approx(report.residual, rel=1e-6, abs=1e-14)

    def test_converged_residual_is_true_and_within_rtol(self, solver):
        A, b, x, _, report = self.solve(solver)
        true = np.linalg.norm(b - A @ x)
        assert report.residual == pytest.approx(true, rel=1e-10, abs=1e-14)
        assert report.relative_residual == pytest.approx(true / np.linalg.norm(b), rel=1e-10)
        assert report.relative_residual <= 1e-10

    def test_callback_sees_the_history(self, solver):
        seen = []
        _, _, _, _, report = self.solve(solver, callback=lambda k, r: seen.append((k, r)))
        ks = [k for k, _ in seen]
        assert ks[0] == 0 and ks[-1] == report.iterations
        assert ks == sorted(ks)  # in iteration order; a recheck repeats k
        last = {}
        for k, r in seen:
            last[k] = r
        assert sorted(last) == list(range(report.iterations + 1))
        np.testing.assert_array_equal([last[k] for k in sorted(last)], report.history)

    def test_callback_exception_stops_the_solve(self, solver):
        calls = []

        def stop_at_3(k, r):
            calls.append(k)
            if k == 3:
                raise KeyboardInterrupt  # not an Exception: the strictest case

        A, b = system(nonsym=solver not in SYMMETRIC)
        with pytest.raises(KeyboardInterrupt):
            ms.iterative_solve(A, b, solver=solver, callback=stop_at_3)
        assert calls[-1] == 3  # called no more after it raised


def test_breakdown_pads_the_history():
    """CG breaks down on an indefinite matrix with p^T A p == 0: it advances
    the count and stops without testing again. Before the padding, history
    had `iterations` entries, one short."""
    A = sp.csr_matrix(np.diag([1.0, -1.0]))
    b = np.ones(2)
    x, info, report = ms.cg(A, b, full_output=True)
    assert info > 1  # a breakdown, not maxiter
    assert len(report.history) == report.iterations + 1
    assert np.isnan(report.history[-1])
    assert report.residual == pytest.approx(np.linalg.norm(b - A @ x))


def test_gmres_history_is_the_preconditioned_residual():
    """MTL5's GMRES applies M on the left and tests ||M^-1 r||. On 100 * A
    with the diagonal preconditioner, ||M^-1 b|| is far below ||b||, and the
    report's residual, computed from x, is the unpreconditioned one."""
    A, b = system(nonsym=True)
    A = (100.0 * A).tocsr()
    M = ms.diagonal(A)
    x, info, report = ms.gmres(A, b, M=M, rtol=1e-10, full_output=True)
    assert info == 0
    assert report.history[0] == pytest.approx(np.linalg.norm(M.solve(b).to_numpy()), rel=1e-12)
    assert report.history[0] < 0.1 * np.linalg.norm(b)
    assert report.residual == pytest.approx(np.linalg.norm(b - A @ x), rel=1e-10)


def test_default_return_is_still_x_info():
    A, b = system(nonsym=False)
    out = ms.cg(A, b)
    assert isinstance(out, tuple) and len(out) == 2


def test_maxiter_reached():
    A, b = system(nonsym=False)
    _, info, report = ms.cg(A, b, maxiter=5, full_output=True)
    assert info == 1
    assert report.iterations == 5
    assert len(report.history) == 6
    assert report.relative_residual > 1e-10


def test_preconditioning_shows_in_the_history():
    A, b = system(nonsym=True)
    _, _, plain = ms.gmres(A, b, rtol=1e-10, full_output=True)
    _, _, ilu = ms.gmres(A, b, M=ms.ilu0(A), rtol=1e-10, full_output=True)
    assert ilu.iterations < plain.iterations
    # each history falls from ||b|| to below rtol * ||b||
    for report in (plain, ilu):
        assert report.history[-1] <= 1e-10 * report.history[0]


def test_float32():
    A, b = system(nonsym=False)
    _, info, report = ms.cg(A.astype(np.float32), b.astype(np.float32), rtol=1e-5, full_output=True)
    assert info == 0
    assert report.history.dtype == np.float64
    assert len(report.history) == report.iterations + 1
    assert report.relative_residual <= 1e-5


def test_named_wrappers_pass_full_output_and_callback():
    A, b = system(nonsym=True)
    seen = []
    _, info, report = ms.bicgstab(A, b, callback=lambda k, r: seen.append(k), full_output=True)
    assert info == 0 and seen[-1] == report.iterations


def test_a_non_callable_callback_is_refused():
    A, b = system(nonsym=False)
    with pytest.raises(TypeError, match="callable"):
        ms.cg(A, b, callback=3)


def test_report_is_a_named_tuple():
    A, b = system(nonsym=False)
    _, _, report = ms.cg(A, b, full_output=True)
    iterations, residual, relative, history = report
    assert isinstance(report, ms.SolveReport)
    assert (iterations, residual) == (report.iterations, report.residual)
