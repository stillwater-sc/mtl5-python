"""Alternative sparse storage formats: COO and ELL, plus native transpose-SpMV.

The reference matrix is deliberately **non-symmetric**. A symmetric one would
make `A @ x` and `A.T @ x` agree, so an orientation mistake in any of these
formats would pass unnoticed — which is exactly the bug that makes CSC
unbindable here (see TestNoCSC).
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

sp = pytest.importorskip("scipy.sparse")
import mtl5.sparse as ms  # noqa: E402

# A = [[1,2,0],[0,3,4],[5,0,6]]        A @ x = [5, 18, 23]
# x = [1,2,3]                          A.T @ x = [16, 8, 26]
DENSE = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])
X = np.array([1.0, 2.0, 3.0])
ROW = [0, 0, 1, 1, 2, 2]
COL = [0, 1, 1, 2, 0, 2]
VAL = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

DTYPES = [(np.float32, "f32", 1e-6), (np.float64, "f64", 1e-12)]


@pytest.fixture(params=DTYPES, ids=[d[1] for d in DTYPES])
def dtype(request):
    return request.param


def csr(np_dt=np.float64):
    return ms.from_scipy(sp.csr_matrix(DENSE.astype(np_dt)))


class TestCOO:
    def test_construction_and_shape(self, dtype):
        np_dt, suffix, _ = dtype
        C = ms.coo_matrix(ROW, COL, np.array(VAL, dtype=np_dt), (3, 3))
        assert C.shape == (3, 3)
        assert C.nnz == 6
        assert C.dtype == suffix
        assert "SparseMatrixCOO" in repr(C)

    def test_matvec(self, dtype):
        np_dt, _, tol = dtype
        C = ms.coo_matrix(ROW, COL, np.array(VAL, dtype=np_dt), (3, 3))
        np.testing.assert_allclose(C.matvec(X.astype(np_dt)).to_numpy(), DENSE @ X, rtol=tol)

    def test_round_trip_through_scipy(self, dtype):
        np_dt, _, _ = dtype
        C = ms.coo_matrix(ROW, COL, np.array(VAL, dtype=np_dt), (3, 3))
        back = ms.to_scipy(C)
        assert back.format == "coo", "a COO round trip must not silently compress"
        np.testing.assert_allclose(back.toarray(), DENSE)

    def test_element_access(self):
        C = ms.coo_matrix(ROW, COL, VAL, (3, 3))
        assert C[2, 0] == 5.0
        assert C[0, 2] == 0.0, "an absent entry reads as zero, not an error"

    def test_insert_appends(self):
        C = ms.coo_matrix([0], [0], [1.0], (2, 2))
        C.insert(1, 1, 7.0)
        assert C.nnz == 2
        assert C[1, 1] == 7.0

    def test_sort(self):
        C = ms.coo_matrix([2, 0, 1], [0, 1, 1], [5.0, 2.0, 3.0], (3, 3))
        assert not C.is_sorted
        C.sort()
        assert C.is_sorted
        row, _, _ = C.to_coo_arrays()
        np.testing.assert_array_equal(row, [0, 1, 2])

    def test_to_coo_arrays(self):
        C = ms.coo_matrix(ROW, COL, VAL, (3, 3))
        row, col, data = C.to_coo_arrays()
        np.testing.assert_array_equal(row, ROW)
        np.testing.assert_array_equal(col, COL)
        np.testing.assert_allclose(data, VAL)

    def test_tocsr(self):
        C = ms.coo_matrix(ROW, COL, VAL, (3, 3))
        np.testing.assert_allclose(ms.to_scipy(C.tocsr()).toarray(), DENSE)

    def test_out_of_range_construction_raises(self):
        """coordinate2D only asserts, which is a no-op in a release build, so
        an out-of-range index would be stored and surface later as a wrong
        answer. The binding checks up front instead."""
        with pytest.raises(IndexError, match="row index 5"):
            ms.coo_matrix([5], [0], [1.0], (2, 2))
        with pytest.raises(IndexError, match="column index 9"):
            ms.coo_matrix([0], [9], [1.0], (2, 2))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            ms.coo_matrix([0, 1], [0], [1.0], (2, 2))

    def test_insert_out_of_range_raises(self):
        C = ms.coo_matrix([0], [0], [1.0], (2, 2))
        with pytest.raises(IndexError):
            C.insert(5, 0, 1.0)


class TestCOODuplicates:
    """Duplicates accumulate rather than overwrite, matching
    scipy.sparse.coo_matrix. They stay separate until tocsr() folds them."""

    def test_duplicates_sum_on_read(self):
        C = ms.coo_matrix([0, 0], [0, 0], [1.0, 2.0], (2, 2))
        assert C.nnz == 2, "both triplets are still stored"
        assert C[0, 0] == 3.0, "reading sums them"

    def test_tocsr_folds_them(self):
        C = ms.coo_matrix([0, 0], [0, 0], [1.0, 2.0], (2, 2))
        compressed = C.tocsr()
        assert compressed.nnz == 1
        assert ms.to_scipy(compressed).toarray()[0, 0] == 3.0

    def test_matches_scipy(self):
        C = ms.coo_matrix([0, 0], [0, 0], [1.0, 2.0], (2, 2))
        reference = sp.coo_matrix(([1.0, 2.0], ([0, 0], [0, 0])), shape=(2, 2))
        assert C.nnz == reference.nnz
        np.testing.assert_allclose(ms.to_scipy(C).toarray(), reference.toarray())

    def test_round_trip_preserves_duplicates(self):
        C = ms.coo_matrix([0, 0], [0, 0], [1.0, 2.0], (2, 2))
        assert ms.to_scipy(C).nnz == 2, "the round trip must not sum them early"


class TestELL:
    def test_from_csr(self, dtype):
        np_dt, suffix, _ = dtype
        E = ms.ell_matrix(sp.csr_matrix(DENSE.astype(np_dt)))
        assert E.shape == (3, 3)
        assert E.dtype == suffix
        assert E.max_width == 2, "the widest row has two entries"
        assert E.nnz == 6
        assert "SparseMatrixELL" in repr(E)

    def test_matvec(self, dtype):
        np_dt, _, tol = dtype
        E = ms.ell_matrix(sp.csr_matrix(DENSE.astype(np_dt)))
        np.testing.assert_allclose(E.matvec(X.astype(np_dt)).to_numpy(), DENSE @ X, rtol=tol)

    def test_round_trip(self):
        E = ms.ell_matrix(sp.csr_matrix(DENSE))
        np.testing.assert_allclose(ms.to_scipy(E).toarray(), DENSE)

    def test_accepts_an_mtl5_matrix_too(self):
        E = ms.ell_matrix(csr())
        np.testing.assert_allclose(ms.to_scipy(E).toarray(), DENSE)

    def test_element_access(self):
        E = ms.ell_matrix(sp.csr_matrix(DENSE))
        assert E[2, 0] == 5.0
        assert E[0, 2] == 0.0

    def test_to_ell_arrays_uses_minus_one_for_padding(self):
        """ell_matrix's own sentinel is size_type(-1), which would arrive in
        NumPy as a huge positive index. The binding maps it to -1."""
        ragged = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 0.0]]))
        E = ms.ell_matrix(ragged)
        indices, data = E.to_ell_arrays()
        assert indices.shape == (2, 2)
        assert indices.min() == -1, "padding is -1"
        assert data[1, 1] == 0.0, "padded data slots are zero"

    def test_padding_ratio_is_zero_for_uniform_rows(self):
        E = ms.ell_matrix(sp.csr_matrix(DENSE))
        assert E.padding_ratio == pytest.approx(0.0)
        assert E.stored == E.nnz

    def test_padding_ratio_exposes_a_bad_fit(self):
        """One long row forces every other row to carry empty slots. This is
        the number that says 'use CSR instead', so it has to be right."""
        ragged = np.zeros((4, 4))
        ragged[0, :] = 1.0  # a full row
        ragged[1:, 0] = [2.0, 3.0, 4.0]  # one entry each
        E = ms.ell_matrix(sp.csr_matrix(ragged))
        assert E.max_width == 4
        assert E.nnz == 7
        assert E.stored == 16
        assert E.padding_ratio == pytest.approx(9 / 16)

    def test_empty_matrix_padding_ratio_is_defined(self):
        E = ms.ell_matrix(sp.csr_matrix(np.zeros((3, 3))))
        assert E.padding_ratio == 0.0

    def test_the_two_ratios_quoted_in_the_readme(self):
        """The README uses these to say when ELL is worth it. Pin them so the
        advice cannot drift away from what the code does."""
        laplacian = ms.ell_matrix(mtl5.generators.laplacian_2d(32, 32))
        assert laplacian.padding_ratio == pytest.approx(0.025, abs=5e-4)

        n = 200
        arrow = sp.lil_matrix((n, n))
        arrow.setdiag(1.0)
        arrow[0, :] = 1.0
        arrow[:, 0] = 1.0
        assert ms.ell_matrix(arrow.tocsr()).padding_ratio == pytest.approx(0.985, abs=5e-4)


class TestTransposeSpMV:
    """`rmatvec` uses MTL5's transpose view rather than a scipy round trip.
    Checked against a non-symmetric matrix so a no-op transpose would fail."""

    def test_rmatvec(self, dtype):
        np_dt, _, tol = dtype
        M = csr(np_dt)
        np.testing.assert_allclose(M.rmatvec(X.astype(np_dt)).to_numpy(), DENSE.T @ X, rtol=tol)

    def test_rmatvec_differs_from_matvec(self):
        M = csr()
        assert not np.allclose(M.matvec(X).to_numpy(), M.rmatvec(X).to_numpy())

    def test_non_square(self):
        """The shapes have to swap, which a symmetric test would never catch."""
        rect = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
        M = ms.from_scipy(sp.csr_matrix(rect))
        z = np.array([1.0, 2.0])
        out = M.rmatvec(z).to_numpy()
        assert len(out) == 3
        np.testing.assert_allclose(out, rect.T @ z)

    def test_accepts_a_vector_view(self):
        M = csr()
        np.testing.assert_allclose(M.rmatvec(mtl5.vector(X)).to_numpy(), DENSE.T @ X)

    def test_length_mismatch_raises(self):
        M = ms.from_scipy(sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])))
        with pytest.raises(ValueError, match="num_rows"):
            M.rmatvec(np.ones(3))

    def test_linear_operator_uses_it(self):
        M = csr()
        lo = ms.as_linear_operator(M)
        np.testing.assert_allclose(lo @ X, DENSE @ X)
        np.testing.assert_allclose(lo.T @ X, DENSE.T @ X)

    def test_tocoo(self):
        C = csr().tocoo()
        assert C.nnz == 6
        assert C.is_sorted
        np.testing.assert_allclose(ms.to_scipy(C).toarray(), DENSE)


class TestNoCSC:
    """MTL5's compressed2D takes tag::col_major but ignores it — the inserter,
    element access and mult all treat the storage as row-major. There is
    therefore no CSC container to bind, and scipy CSC input is converted.
    """

    def test_csc_input_becomes_csr(self):
        M = ms.from_scipy(sp.csc_matrix(DENSE))
        assert "SparseMatrix_f64" in repr(M)
        np.testing.assert_allclose(ms.to_scipy(M).toarray(), DENSE)

    def test_no_csc_class_is_exported(self):
        """If MTL5 ever implements col_major and a CSC binding lands, this
        should fail and be replaced by real CSC coverage."""
        assert not [n for n in ms.__all__ if "CSC" in n.upper()]


class TestPublicSurface:
    def test_new_names_are_exported(self):
        for name in (
            "SparseMatrixCOO_f32",
            "SparseMatrixCOO_f64",
            "SparseMatrixELL_f32",
            "SparseMatrixELL_f64",
            "coo_matrix",
            "ell_matrix",
        ):
            assert name in ms.__all__, f"{name} missing from mtl5.sparse.__all__"
            assert hasattr(ms, name)

    def test_dtype_is_rejected_clearly(self):
        with pytest.raises(TypeError, match="float32, float64"):
            ms.coo_matrix([0], [0], np.array([1], dtype=np.int64), (2, 2))
