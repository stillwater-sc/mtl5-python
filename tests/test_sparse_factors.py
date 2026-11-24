"""The factors of the sparse direct solvers, exposed for inspection (#125).

Each factorization hands back its triangular factors as SparseMatrix copies,
plus the permutations that relate them to A. The properties worth pinning:

- the documented identity reconstructs A, so the permutation convention
  (p[new] = old, indexing the ORIGINAL matrix) is the one actually returned;
- the factors have the triangular shape and diagonal they claim;
- the factor nnz agrees with the `.nnz` the solver already reported;
- the column permutation of an LU is exactly the ordering asked for;
- a factor is a copy, so refactor() neither leaves it stale on the next access
  nor changes one taken earlier.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

sp = pytest.importorskip("scipy.sparse")

import mtl5.sparse as ms  # noqa: E402


def laplacian_2d(k: int):
    eye = sp.eye(k, format="csr")
    tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k), format="csr")
    return (sp.kron(eye, tri) + sp.kron(tri, eye)).tocsr()


def unsymmetric(n: int = 120, seed: int = 0):
    """Unsymmetric sparse matrix whose diagonal is too small to pivot on.

    Threshold pivoting with threshold=1.0 takes the largest entry in each
    column, so a diagonal two orders of magnitude below its off-diagonals
    forces row interchanges. Without that the row permutation would be the
    identity and the test could not tell row_perm from its inverse.
    """
    rng = np.random.default_rng(seed)
    R = sp.random(n, n, density=0.04, random_state=rng, format="csr")
    R.data = rng.uniform(1.0, 2.0, R.nnz)
    # A cyclic off-diagonal guarantees every column has an entry to pivot on.
    cyc = sp.csr_matrix((np.full(n, 3.0), (np.roll(np.arange(n), 1), np.arange(n))), shape=(n, n))
    return (R + cyc + 0.01 * sp.eye(n)).tocsr()


def dense(M) -> np.ndarray:
    return ms.to_scipy(M).toarray()


def rel(X, Y) -> float:
    return np.linalg.norm(X - Y) / np.linalg.norm(Y)


def assert_permutation(p, n: int):
    assert p.dtype == np.int64
    assert sorted(p.tolist()) == list(range(n))


class TestSparseLU:
    @pytest.mark.parametrize("ordering", ["colamd", "amd", "rcm", "natural"])
    def test_reconstructs(self, ordering):
        A = unsymmetric()
        lu = ms.splu(A, ordering=ordering)
        L, U = dense(lu.L), dense(lu.U)
        assert_permutation(lu.row_perm, A.shape[0])
        assert_permutation(lu.col_perm, A.shape[0])
        PAQ = A.toarray()[lu.row_perm][:, lu.col_perm]
        assert rel(L @ U, PAQ) < 1e-13

    def test_pivoting_happened(self):
        # Guards the test above: with an identity row_perm, p and its inverse
        # coincide and the convention would go unchecked.
        lu = ms.splu(unsymmetric(), ordering="natural")
        assert not np.array_equal(lu.row_perm, np.arange(lu.n))

    def test_triangular_shape(self):
        lu = ms.splu(unsymmetric())
        L, U = dense(lu.L), dense(lu.U)
        assert np.array_equal(L, np.tril(L))
        assert np.array_equal(np.diag(L), np.ones(lu.n))
        assert np.array_equal(U, np.triu(U))

    def test_nnz_matches(self):
        lu = ms.splu(unsymmetric())
        assert lu.L.nnz + lu.U.nnz == lu.nnz

    @pytest.mark.parametrize("ordering", ["colamd", "amd", "rcm"])
    def test_col_perm_is_the_ordering(self, ordering):
        A = unsymmetric()
        lu = ms.splu(A, ordering=ordering)
        assert np.array_equal(lu.col_perm, ms.ordering(A, ordering))

    def test_ordering_shows_in_the_factors(self):
        """The point of exposing them: the fill is visible in L and U."""
        A = laplacian_2d(20)
        nat = ms.splu(A, ordering="natural")
        amd = ms.splu(A, ordering="amd")
        assert amd.L.nnz + amd.U.nnz < nat.L.nnz + nat.U.nnz

    def test_float32_factors(self):
        A = unsymmetric().astype(np.float32)
        lu = ms.splu(A)
        assert isinstance(lu.L, ms.SparseMatrix_f32)
        assert isinstance(lu.U, ms.SparseMatrix_f32)
        PAQ = A.toarray()[lu.row_perm][:, lu.col_perm].astype(np.float64)
        assert rel(dense(lu.L).astype(np.float64) @ dense(lu.U), PAQ) < 1e-5

    def test_factors_are_copies(self):
        A = unsymmetric()
        lu = ms.splu(A)
        L_old = lu.L
        L_before = dense(L_old)
        A2 = A.copy()
        A2.data = A2.data * (1.0 + 0.1 * np.cos(np.arange(A2.nnz)))
        lu.refactor(ms.from_scipy(A2))
        # Taken earlier: unchanged by the refactor.
        assert np.array_equal(dense(L_old), L_before)
        assert not np.array_equal(dense(lu.L), L_before)
        # Taken now: the new factors, with the pivot sequence reused.
        PAQ2 = A2.toarray()[lu.row_perm][:, lu.col_perm]
        assert rel(dense(lu.L) @ dense(lu.U), PAQ2) < 1e-12

    def test_spy_accepts_a_factor(self, tmp_path):
        lu = ms.splu(laplacian_2d(10))
        out = tmp_path / "U.png"
        mtl5.io.spy(lu.U, out)
        assert out.stat().st_size > 0


class TestSupernodalLU:
    @pytest.mark.parametrize("scale", [False, True])
    def test_reconstructs(self, scale):
        A = unsymmetric()
        f = ms.supernodal_lu(A, scale=scale)
        assert_permutation(f.row_perm, A.shape[0])
        assert_permutation(f.col_perm, A.shape[0])
        r = f.row_scale
        if scale:
            assert r is not None and r.shape == (A.shape[0],)
        else:
            assert r is None
            r = np.ones(A.shape[0])
        RAPQ = (r[:, None] * A.toarray())[f.row_perm][:, f.col_perm]
        assert rel(dense(f.L) @ dense(f.U), RAPQ) < 1e-13

    def test_triangular_and_nnz(self):
        f = ms.supernodal_lu(unsymmetric())
        L, U = dense(f.L), dense(f.U)
        assert np.array_equal(L, np.tril(L))
        assert np.array_equal(np.diag(L), np.ones(f.n))
        assert np.array_equal(U, np.triu(U))
        assert f.L.nnz + f.U.nnz == f.nnz


class TestCholesky:
    @pytest.mark.parametrize("ordering", ["amd", "rcm", "natural"])
    def test_reconstructs(self, ordering):
        A = laplacian_2d(15)
        c = ms.cholesky(A, ordering=ordering)
        L = dense(c.L)
        assert_permutation(c.perm, A.shape[0])
        assert np.array_equal(L, np.tril(L))
        assert c.L.nnz == c.nnz
        PAP = A.toarray()[c.perm][:, c.perm]
        assert rel(L @ L.T, PAP) < 1e-13

    def test_perm_is_the_ordering(self):
        A = laplacian_2d(15)
        assert np.array_equal(ms.cholesky(A, ordering="amd").perm, ms.ordering(A, "amd"))


def indefinite(k: int = 12):
    """Symmetric indefinite with no zero pivot: a shifted 2-D Laplacian."""
    A = laplacian_2d(k)
    return (A - 1.7 * sp.eye(A.shape[0])).tocsr()


@pytest.mark.parametrize("factor", [ms.ldlt, ms.supernodal_ldlt], ids=["ldlt", "supernodal_ldlt"])
class TestLDLT:
    def test_reconstructs(self, factor):
        A = indefinite()
        f = factor(A)
        L, d = dense(f.L), f.diagonal()
        assert_permutation(f.perm, A.shape[0])
        assert (d < 0).any() and (d > 0).any()  # really indefinite
        PAP = A.toarray()[f.perm][:, f.perm]
        assert rel(L @ np.diag(d) @ L.T, PAP) < 1e-12

    def test_unit_diagonal_is_explicit(self, factor):
        f = factor(indefinite())
        L = dense(f.L)
        assert np.array_equal(L, np.tril(L))
        assert np.array_equal(np.diag(L), np.ones(f.n))
        # .nnz counts what is stored, which excludes the implicit diagonal.
        assert f.L.nnz == f.nnz + f.n


class TestQR:
    def test_r_matches_normal_equations(self):
        rng = np.random.default_rng(3)
        A = sp.random(80, 30, density=0.1, random_state=rng, format="csr")
        A = (A + sp.eye(80, 30)).tocsr()
        q = ms.qr(A)
        R = dense(q.R)
        assert R.shape == (30, 30)
        assert np.array_equal(R, np.triu(R))
        assert q.R.nnz == q.nnz
        assert_permutation(q.col_perm, 30)
        AQ = A.toarray()[:, q.col_perm]
        assert rel(R.T @ R, AQ.T @ AQ) < 1e-12


def test_klu_exposes_no_factors():
    # KLU's block triangular form has no single (L, U) pair; see the comment on
    # register_klu. Pinned so that adding one is a deliberate decision.
    k = ms.klu(unsymmetric())
    assert not hasattr(k, "L")
    assert not hasattr(k, "U")


# The stress program for TestConcurrentRefactor. It runs in a subprocess so
# that a use-after-free -- the other way this race fails -- shows up as a
# failing exit code rather than taking pytest down with it.
_RACE = r"""
import sys, threading
import numpy as np
import scipy.sparse as sp
import mtl5.sparse as ms

kind = sys.argv[1]
k = 60
eye = sp.eye(k, format="csr")
tri = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k), format="csr")
A1 = (sp.kron(eye, tri) + sp.kron(tri, eye)).tocsr()
A2 = A1.copy()
A2.data = A2.data * 4.0
M1, M2 = ms.from_scipy(A1), ms.from_scipy(A2)
make = {"cholesky": ms.cholesky, "ldlt": ms.ldlt, "splu": ms.splu,
        "supernodal_lu": ms.supernodal_lu}[kind]
attr = "U" if kind in ("splu", "supernodal_lu") else "L"

def snap(f):
    return ms.to_scipy(getattr(f, attr)).data.copy()

ref = [snap(make(A1)), snap(make(A2))]
b = A1 @ np.ones(A1.shape[0])
xs = [np.ones(A1.shape[0]), np.full(A1.shape[0], 0.25)]   # A2 = 4 * A1
f = make(A1)
stop = threading.Event()
torn = [0, 0]

def writer():
    for i in range(400):
        f.refactor(M2 if i % 2 else M1)
    stop.set()

def reader():
    while not stop.is_set():
        d = snap(f)
        if not any(d.shape == r.shape and np.allclose(d, r) for r in ref):
            torn[0] += 1
        x = f.solve(b).to_numpy()
        if not any(np.allclose(x, xr) for xr in xs):
            torn[1] += 1

threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("torn factor reads:", torn[0], "torn solves:", torn[1])
sys.exit(1 if any(torn) else 0)
"""


class TestConcurrentRefactor:
    """refactor() replaces the factor with the GIL released (CodeRabbit on #126).

    A getter, solve() or iterative_refine running in another thread must see
    either the old factor or the new one, never a mix. Before the per-wrapper
    lock, three reader threads against one refactoring writer saw torn factors
    in every run, for every factorization here.
    """

    @pytest.mark.parametrize("kind", ["cholesky", "ldlt", "splu", "supernodal_lu"])
    def test_readers_never_see_a_torn_factor(self, kind):
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-c", _RACE, kind], capture_output=True, text=True, timeout=300
        )
        assert r.returncode == 0, r.stdout + r.stderr
