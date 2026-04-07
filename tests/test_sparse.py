"""Tests for MTL5 sparse matrix interop with SciPy."""

import numpy as np
import numpy.testing as npt
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")
from scipy.sparse.linalg import cg, gmres  # noqa: E402

import mtl5  # noqa: E402
import mtl5.sparse as msp  # noqa: E402


@pytest.fixture
def sample_csr():
    """A 3x3 CSR matrix:
    [[1, 0, 2],
     [0, 3, 4],
     [5, 0, 0]]
    """
    indptr = np.array([0, 2, 4, 5], dtype=np.int64)
    indices = np.array([0, 2, 1, 2, 0], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    return data, indices, indptr, (3, 3)


class TestSparseConstruction:
    def test_csr_matrix_factory(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        A = msp.csr_matrix(data, indices, indptr, shape)
        assert isinstance(A, msp.SparseMatrix_f64)
        assert A.shape == (3, 3)
        assert A.nnz == 5
        assert A.dtype == "f64"

    def test_f32_dispatch(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        data_f32 = data.astype(np.float32)
        A = msp.csr_matrix(data_f32, indices, indptr, shape)
        assert isinstance(A, msp.SparseMatrix_f32)

    def test_invalid_indptr_length(self, sample_csr):
        data, indices, indptr, _ = sample_csr
        # indptr should be nrows+1 = 4, pass nrows=2 → expected length 3
        with pytest.raises(ValueError):
            msp.csr_matrix(data, indices, indptr, (2, 3))

    def test_repr(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        A = msp.csr_matrix(data, indices, indptr, shape)
        r = repr(A)
        assert "SparseMatrix_f64" in r
        assert "nnz=5" in r


class TestScipyRoundTrip:
    def test_dense_to_scipy_csr_to_mtl5_back(self):
        dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0], [5.0, 0.0, 0.0]])
        sp_orig = scipy_sparse.csr_matrix(dense)

        mtl5_sparse = msp.from_scipy(sp_orig)
        assert mtl5_sparse.shape == (3, 3)
        assert mtl5_sparse.nnz == sp_orig.nnz

        sp_back = msp.to_scipy(mtl5_sparse)
        assert sp_back.shape == sp_orig.shape
        npt.assert_array_equal(sp_back.toarray(), dense)

    def test_csc_input_converts(self):
        dense = np.array([[1.0, 2.0], [0.0, 3.0]])
        csc = scipy_sparse.csc_matrix(dense)
        mtl5_sparse = msp.from_scipy(csc)
        sp_back = msp.to_scipy(mtl5_sparse)
        npt.assert_array_equal(sp_back.toarray(), dense)

    def test_coo_input_converts(self):
        dense = np.array([[1.0, 0.0], [0.0, 4.0]])
        coo = scipy_sparse.coo_matrix(dense)
        mtl5_sparse = msp.from_scipy(coo)
        sp_back = msp.to_scipy(mtl5_sparse)
        npt.assert_array_equal(sp_back.toarray(), dense)

    def test_rectangular_matrix(self):
        dense = np.array([[1.0, 2.0, 0.0, 4.0], [0.0, 5.0, 6.0, 0.0]])
        sp_orig = scipy_sparse.csr_matrix(dense)
        mtl5_sparse = msp.from_scipy(sp_orig)
        assert mtl5_sparse.shape == (2, 4)
        sp_back = msp.to_scipy(mtl5_sparse)
        npt.assert_array_equal(sp_back.toarray(), dense)

    def test_1x1_matrix(self):
        sp_orig = scipy_sparse.csr_matrix(np.array([[7.0]]))
        mtl5_sparse = msp.from_scipy(sp_orig)
        assert mtl5_sparse.shape == (1, 1)
        sp_back = msp.to_scipy(mtl5_sparse)
        npt.assert_array_equal(sp_back.toarray(), [[7.0]])

    def test_empty_matrix(self):
        sp_orig = scipy_sparse.csr_matrix((3, 3))
        mtl5_sparse = msp.from_scipy(sp_orig)
        assert mtl5_sparse.nnz == 0
        sp_back = msp.to_scipy(mtl5_sparse)
        assert sp_back.nnz == 0

    def test_from_scipy_rejects_non_sparse(self):
        with pytest.raises(TypeError):
            msp.from_scipy(np.array([[1.0]]))


class TestSpMV:
    def test_matvec_correctness(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        A = msp.csr_matrix(data, indices, indptr, shape)
        x = mtl5.vector(np.array([1.0, 1.0, 1.0]))
        y = A.matvec(x)
        # A @ [1,1,1] for [[1,0,2],[0,3,4],[5,0,0]] = [3, 7, 5]
        npt.assert_allclose(y.to_numpy(), [3.0, 7.0, 5.0])

    def test_matvec_ndarray_input(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        A = msp.csr_matrix(data, indices, indptr, shape)
        y = A.matvec(np.array([1.0, 2.0, 3.0]))
        # Expected: [1*1 + 2*3, 3*2 + 4*3, 5*1] = [7, 18, 5]
        npt.assert_allclose(y.to_numpy(), [7.0, 18.0, 5.0])

    def test_matvec_matches_scipy(self, rng):
        n = 30
        sp_dense = rng.standard_normal((n, n))
        sp_dense[sp_dense < 0.7] = 0  # ~75% sparsity
        sp_csr = scipy_sparse.csr_matrix(sp_dense)
        A = msp.from_scipy(sp_csr)

        x = rng.standard_normal(n)
        y_mtl5 = A.matvec(x).to_numpy()
        y_scipy = sp_csr @ x
        npt.assert_allclose(y_mtl5, y_scipy, rtol=1e-12)

    def test_matvec_rectangular(self, rng):
        sp_csr = scipy_sparse.csr_matrix(rng.standard_normal((4, 6)))
        A = msp.from_scipy(sp_csr)
        x = rng.standard_normal(6)
        y_mtl5 = A.matvec(x).to_numpy()
        y_scipy = sp_csr @ x
        npt.assert_allclose(y_mtl5, y_scipy, rtol=1e-12)

    def test_matvec_dimension_mismatch(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        A = msp.csr_matrix(data, indices, indptr, shape)
        with pytest.raises(ValueError):
            A.matvec(np.array([1.0, 2.0]))  # too short


class TestLinearOperator:
    def test_as_linear_operator_returns_scipy_lo(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        A = msp.csr_matrix(data, indices, indptr, shape)
        lo = msp.as_linear_operator(A)
        from scipy.sparse.linalg import LinearOperator

        assert isinstance(lo, LinearOperator)
        assert lo.shape == (3, 3)

    def test_linear_operator_matvec(self, sample_csr):
        data, indices, indptr, shape = sample_csr
        A = msp.csr_matrix(data, indices, indptr, shape)
        lo = msp.as_linear_operator(A)
        y = lo @ np.array([1.0, 2.0, 3.0])
        npt.assert_allclose(y, [7.0, 18.0, 5.0])

    def test_cg_solve_via_linear_operator(self, rng):
        """Use scipy's CG with an MTL5 sparse matrix backing."""
        n = 50
        # Build an SPD sparse matrix
        A_dense = rng.standard_normal((n, n))
        A_spd = A_dense @ A_dense.T + n * np.eye(n)  # SPD by construction
        # Sparsify lightly to keep CG meaningful but well-conditioned
        A_spd[np.abs(A_spd) < 0.05] = 0
        # Restore diagonal so it's still SPD
        np.fill_diagonal(A_spd, np.diag(A_dense @ A_dense.T) + n)

        sp_csr = scipy_sparse.csr_matrix(A_spd)
        mtl5_A = msp.from_scipy(sp_csr)
        lo = msp.as_linear_operator(mtl5_A)

        b = rng.standard_normal(n)
        x_via_lo, info = cg(lo, b, rtol=1e-10, maxiter=200)
        assert info == 0, f"CG did not converge: info={info}"

        # Compare with scipy direct
        x_scipy, _ = cg(sp_csr, b, rtol=1e-10, maxiter=200)
        npt.assert_allclose(x_via_lo, x_scipy, rtol=1e-6)

    def test_gmres_via_linear_operator(self, rng):
        n = 30
        A_dense = rng.standard_normal((n, n)) + n * np.eye(n)
        A_dense[np.abs(A_dense) < 0.3] = 0
        np.fill_diagonal(A_dense, np.diag(A_dense) + n)  # well-conditioned

        sp_csr = scipy_sparse.csr_matrix(A_dense)
        lo = msp.as_linear_operator(msp.from_scipy(sp_csr))

        b = rng.standard_normal(n)
        x, info = gmres(lo, b, rtol=1e-10, maxiter=200)
        assert info == 0

        # Verify by computing residual via the LinearOperator
        residual = np.linalg.norm(lo @ x - b) / np.linalg.norm(b)
        assert residual < 1e-8
