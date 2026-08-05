"""Matrix views.

Two of these carry the weight, because both are places a caller could reasonably
expect the opposite of what the code does:

  * `banded` takes **bandwidths**, not signed diagonal offsets. `banded(A, 1, 1)`
    is tridiagonal. Passing -1 for the lower bandwidth — which reads plausibly
    as "one subdiagonal" — asked upstream for `diff >= 1` and silently returned
    only the superdiagonal, so the binding refuses a negative.
  * `hermitian` is **not** the adjoint. It reads the upper triangle and mirrors
    it conjugated into the lower, discarding whatever was there. `mtl5.adjoint`
    is A^H. TestHermitianIsNotAdjoint pins the distinction.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

v = mtl5.view

A = np.arange(1.0, 10.0).reshape(3, 3)
DTYPES = [(np.float32, "f32"), (np.float64, "f64")]


def M(a=A, dt=np.float64):
    return mtl5.matrix(np.ascontiguousarray(a, dtype=dt))


class TestTriangular:
    @pytest.mark.parametrize("np_dt,_suffix", DTYPES)
    def test_lower(self, np_dt, _suffix):
        np.testing.assert_allclose(v.lower(M(dt=np_dt)).to_numpy(), np.tril(A))

    @pytest.mark.parametrize("np_dt,_suffix", DTYPES)
    def test_upper(self, np_dt, _suffix):
        np.testing.assert_allclose(v.upper(M(dt=np_dt)).to_numpy(), np.triu(A))

    def test_strict_lower(self):
        np.testing.assert_allclose(v.strict_lower(M()).to_numpy(), np.tril(A, -1))

    def test_strict_upper(self):
        np.testing.assert_allclose(v.strict_upper(M()).to_numpy(), np.triu(A, 1))

    def test_lower_plus_strict_upper_reconstructs(self):
        """They partition A, which is the property that makes them useful for
        splitting an operator."""
        out = v.lower(M()).to_numpy() + v.strict_upper(M()).to_numpy()
        np.testing.assert_allclose(out, A)

    def test_non_square_is_fine(self):
        rect = np.arange(1.0, 13.0).reshape(3, 4)
        np.testing.assert_allclose(v.lower(M(rect)).to_numpy(), np.tril(rect))


class TestTransposed:
    def test_matches_numpy(self):
        np.testing.assert_allclose(v.transposed(M()).to_numpy(), A.T)

    def test_non_square_swaps_the_shape(self):
        rect = np.arange(1.0, 13.0).reshape(3, 4)
        out = v.transposed(M(rect)).to_numpy()
        assert out.shape == (4, 3)
        np.testing.assert_allclose(out, rect.T)

    def test_complex_is_not_conjugated(self):
        """A plain transpose. mtl5.adjoint is the one that conjugates."""
        Z = np.array([[1 + 1j, 2 - 1j], [0 + 3j, 4 + 0j]])
        np.testing.assert_allclose(v.transposed(mtl5.matrix(Z)).to_numpy(), Z.T)


class TestBandedTakesBandwidths:
    """The parameters are counts, not signed diagonal offsets."""

    def test_tridiagonal(self):
        np.testing.assert_allclose(v.banded(M(), 1, 1).to_numpy(), np.tril(np.triu(A, -1), 1))

    def test_diagonal(self):
        np.testing.assert_allclose(v.banded(M(), 0, 0).to_numpy(), np.diag(np.diag(A)))

    def test_asymmetric_band(self):
        np.testing.assert_allclose(v.banded(M(), 0, 1).to_numpy(), np.tril(np.triu(A, 0), 1))
        np.testing.assert_allclose(v.banded(M(), 1, 0).to_numpy(), np.tril(np.triu(A, -1), 0))

    def test_a_wide_band_is_the_whole_matrix(self):
        np.testing.assert_allclose(v.banded(M(), 10, 10).to_numpy(), A)

    @pytest.mark.parametrize("lo,up", [(-1, 1), (1, -1), (-1, -1)])
    def test_negative_bandwidths_are_refused(self, lo, up):
        """Upstream reads them as `diff >= -lower`, so a negative lower asks for
        `diff >= 1` and quietly returns only the superdiagonal. Refuse instead
        of returning something that looks like a band and is not."""
        with pytest.raises(ValueError, match="bandwidths"):
            v.banded(M(), lo, up)

    def test_the_message_gives_the_right_call(self):
        with pytest.raises(ValueError, match=r"banded\(A, 1, 1\) is tridiagonal"):
            v.banded(M(), -1, 1)


class TestMap:
    def test_selects_rows_and_columns(self):
        out = v.map(M(), [2, 0], [1, 2]).to_numpy()
        np.testing.assert_allclose(out, A[np.ix_([2, 0], [1, 2])])

    def test_indices_may_repeat(self):
        out = v.map(M(), [1, 1], [0, 0, 0]).to_numpy()
        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, A[np.ix_([1, 1], [0, 0, 0])])

    def test_out_of_range_row(self):
        with pytest.raises(IndexError, match="row index 9"):
            v.map(M(), [9], [0])

    def test_out_of_range_column(self):
        with pytest.raises(IndexError, match="column index 7"):
            v.map(M(), [0], [7])


class TestHermitianIsNotAdjoint:
    """`hermitian_view` reconstructs a full Hermitian matrix from the upper
    triangle. `adjoint` transposes and conjugates. They agree only when the
    input is already Hermitian."""

    # The lower triangle is deliberately inconsistent, so anything that reads it
    # shows up.
    Z = np.array([[1 + 0j, 2 - 1j], [9 + 9j, 4 + 0j]])

    def test_the_result_is_hermitian(self):
        H = v.hermitian(mtl5.matrix(self.Z)).to_numpy()
        np.testing.assert_allclose(H, H.conj().T)

    def test_it_keeps_the_upper_triangle_and_discards_the_lower(self):
        H = v.hermitian(mtl5.matrix(self.Z)).to_numpy()
        assert H[0, 1] == self.Z[0, 1]
        assert H[1, 0] == np.conj(self.Z[0, 1])
        assert H[1, 0] != self.Z[1, 0], "the stored lower triangle must be ignored"

    def test_it_differs_from_adjoint(self):
        H = v.hermitian(mtl5.matrix(self.Z)).to_numpy()
        adj = mtl5.adjoint(mtl5.matrix(self.Z)).to_numpy()
        assert not np.allclose(H, adj)

    def test_they_agree_when_the_input_is_already_hermitian(self):
        """The one case where the two coincide — worth pinning, since it is the
        case a reader is most likely to test with and be misled by."""
        herm = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]])
        H = v.hermitian(mtl5.matrix(herm)).to_numpy()
        adj = mtl5.adjoint(mtl5.matrix(herm)).to_numpy()
        np.testing.assert_allclose(H, adj)
        np.testing.assert_allclose(H, herm)

    def test_a_complex_diagonal_is_refused(self):
        """The view leaves the diagonal alone, so a complex diagonal entry gives
        a result that is not actually Hermitian."""
        bad = np.array([[1 + 1j, 2 + 0j], [2 + 0j, 3 + 0j]])
        with pytest.raises(ValueError, match="imaginary part"):
            v.hermitian(mtl5.matrix(bad))

    def test_the_message_names_the_offending_entry(self):
        bad = np.array([[1 + 0j, 2 + 0j], [2 + 0j, 3 + 5j]])
        with pytest.raises(ValueError, match=r"A\[1, 1\]"):
            v.hermitian(mtl5.matrix(bad))

    def test_non_square_is_refused(self):
        rect = np.zeros((2, 3), dtype=np.complex128)
        with pytest.raises(ValueError, match="square"):
            v.hermitian(mtl5.matrix(rect))

    def test_real_input_works(self):
        """A real symmetric matrix is Hermitian; the diagonal check is a no-op."""
        sym = np.array([[2.0, 1.0], [1.0, 3.0]])
        np.testing.assert_allclose(v.hermitian(M(sym)).to_numpy(), sym)


class TestResultsAreIndependentMatrices:
    """These materialise rather than aliasing. Upstream's views hold a
    `const Matrix&`, which for a Python caller passing a temporary would
    dangle — the same shape as the bug that segfaulted pc::ssor."""

    def test_the_result_does_not_alias_the_source(self):
        a = np.arange(1.0, 10.0).reshape(3, 3)
        src = mtl5.matrix(a)
        out = v.lower(src)
        a[0, 0] = 99.0
        assert out.to_numpy()[0, 0] == 1.0

    def test_a_temporary_source_is_safe(self):
        """No reference is retained, so the source can go away immediately."""
        import gc

        out = v.upper(mtl5.matrix(np.arange(1.0, 10.0).reshape(3, 3)))
        gc.collect()
        np.testing.assert_allclose(out.to_numpy(), np.triu(A))

    def test_the_result_is_a_normal_dense_matrix(self):
        """Which is the point of materialising — it feeds every other binding."""
        out = v.lower(M())
        assert out.shape == (3, 3)
        np.testing.assert_allclose(mtl5.matmul(out, out).to_numpy(), np.tril(A) @ np.tril(A))


class TestPublicSurface:
    def test_view_is_exported(self):
        assert "view" in mtl5.__all__
        assert hasattr(mtl5, "view")

    def test_all_eight_are_listed_and_present(self):
        listed = v.views()
        assert len(listed) == 8
        for name in listed:
            assert hasattr(v, name), f"mtl5.view.{name} missing"

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_complex_dtypes(self, dt):
        # Genuinely Hermitian: the lower entry must be conj of the upper.
        Z = np.array([[1 + 0j, 2 - 1j], [2 + 1j, 3 + 0j]], dtype=dt)
        assert np.allclose(Z, Z.conj().T), "fixture must be Hermitian"
        np.testing.assert_allclose(v.lower(mtl5.matrix(Z)).to_numpy(), np.tril(Z))
        np.testing.assert_allclose(v.hermitian(mtl5.matrix(Z)).to_numpy(), Z)
