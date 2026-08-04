"""Complex element types (c64 = complex64, c128 = complex128).

Three of these tests exist because the operation compiles for complex but does
not mean what a caller would assume, and a compile-only check would have passed
every one of them:

  * `dot` is Hermitian (numpy.vdot), `dot_real` is not (numpy.dot)
  * `.T` does not conjugate; `.H` does
  * `ldlt` is LDL^T, so it is wrong on Hermitian input and the binding refuses it

The rest pin the boundary: what MTL5 supports for complex, and what it must
refuse loudly rather than approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

# A Hermitian matrix (A == A^H) and a complex symmetric one (A == A^T). For
# complex these are different properties, and which one you have decides which
# solver is correct — that distinction is the subject of half this file.
HERMITIAN = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]])
SYMMETRIC = np.array([[2 + 1j, 1 - 1j], [1 - 1j, 3 + 2j]])
X_TRUE = np.array([1 + 0j, 0 + 1j])

DTYPES = [(np.complex64, "c64", 1e-5), (np.complex128, "c128", 1e-12)]


@pytest.fixture(params=DTYPES, ids=[d[1] for d in DTYPES])
def cdtype(request):
    """(numpy dtype, mtl5 suffix, tolerance) for each complex precision."""
    return request.param


class TestContainers:
    def test_dtype_and_shape(self, cdtype):
        np_dt, suffix, _ = cdtype
        v = mtl5.vector(np.array([1 + 2j, 3 - 1j], dtype=np_dt))
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        assert v.dtype == suffix
        assert A.dtype == suffix
        assert A.shape == (2, 2)
        assert len(v) == 2

    def test_vector_is_a_zero_copy_view(self, cdtype):
        np_dt, _, _ = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        v = mtl5.vector(a)
        assert v.is_view
        v[0] = 9 - 9j
        assert a[0] == 9 - 9j, "writing through the view must reach the NumPy buffer"
        assert v.to_numpy().dtype == np_dt

    def test_matrix_is_a_zero_copy_view(self, cdtype):
        np_dt, _, _ = cdtype
        a = HERMITIAN.astype(np_dt)
        A = mtl5.matrix(a)
        A[0, 0] = 7 + 7j
        assert a[0, 0] == 7 + 7j
        np.testing.assert_array_equal(A.to_numpy(), a)

    def test_copy_does_not_alias(self, cdtype):
        np_dt, _, _ = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        w = mtl5.vector(a).copy()
        w[0] = 0j
        assert a[0] == 1 + 2j
        assert not w.is_view

    def test_vector_copy_factory_owns_its_data(self, cdtype):
        np_dt, _, _ = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        v = mtl5.vector_copy(a)
        a[0] = 0j
        assert v[0] == 1 + 2j

    def test_real_and_imag(self, cdtype):
        np_dt, _, _ = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        v = mtl5.vector(a)
        np.testing.assert_allclose(v.real.to_numpy(), a.real)
        np.testing.assert_allclose(v.imag.to_numpy(), a.imag)

        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        np.testing.assert_allclose(A.real.to_numpy(), HERMITIAN.real)
        np.testing.assert_allclose(A.imag.to_numpy(), HERMITIAN.imag)

    def test_out_of_range_raises(self, cdtype):
        np_dt, _, _ = cdtype
        v = mtl5.vector(np.array([1 + 1j], dtype=np_dt))
        with pytest.raises(IndexError):
            v[5]
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        with pytest.raises(IndexError):
            A[9, 0]

    def test_repr_names_the_type(self, cdtype):
        np_dt, suffix, _ = cdtype
        v = mtl5.vector(np.array([1 + 2j, 3 - 1j], dtype=np_dt))
        assert f"DenseVector_{suffix}" in repr(v)
        assert "1+2j" in repr(v) and "3-1j" in repr(v)
        assert f"DenseMatrix_{suffix}" in repr(mtl5.matrix(HERMITIAN.astype(np_dt)))


class TestDotIsHermitian:
    """`dot` conjugates its FIRST argument, so it is numpy.vdot and not
    numpy.dot. Both are exposed under names that say which is which."""

    def test_dot_matches_vdot(self, cdtype):
        np_dt, _, tol = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        b = np.array([2 + 0j, 1 + 1j], dtype=np_dt)
        assert mtl5.dot(mtl5.vector(a), mtl5.vector(b)) == pytest.approx(np.vdot(a, b), rel=tol)

    def test_dot_real_matches_numpy_dot(self, cdtype):
        np_dt, _, tol = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        b = np.array([2 + 0j, 1 + 1j], dtype=np_dt)
        assert mtl5.dot_real(mtl5.vector(a), mtl5.vector(b)) == pytest.approx(np.dot(a, b), rel=tol)

    def test_the_two_actually_differ(self):
        """If these ever coincide the fixture has stopped testing anything."""
        a = np.array([1 + 2j, 3 - 1j])
        b = np.array([2 + 0j, 1 + 1j])
        va, vb = mtl5.vector(a), mtl5.vector(b)
        assert mtl5.dot(va, vb) != mtl5.dot_real(va, vb)

    def test_dot_with_self_is_real_and_positive(self, cdtype):
        """conj(a).a == |a|^2, so a Hermitian product with self has no
        imaginary part. This is the property that makes it the right default."""
        np_dt, _, tol = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        v = mtl5.vector(a)
        d = mtl5.dot(v, v)
        assert d.imag == pytest.approx(0.0, abs=tol)
        assert d.real == pytest.approx(np.sum(np.abs(a) ** 2), rel=tol)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mtl5.dot(mtl5.vector(np.array([1j])), mtl5.vector(np.array([1j, 2j])))


class TestNumpyArraysDoNotLoseTheImaginaryPart:
    """`norm` and `dot` already had float64 ndarray overloads. Without a complex
    overload registered, a complex array could reach those through nanobind's
    converting pass and silently arrive as its real part. These are the
    regression: the answers below differ from the real-part-only answers."""

    def test_norm_of_a_complex_ndarray(self):
        a = np.array([3 + 4j])
        assert mtl5.norm(a, 2) == pytest.approx(5.0)  # not 3.0

    def test_dot_of_complex_ndarrays(self):
        a = np.array([1 + 2j, 3 - 1j])
        b = np.array([2 + 0j, 1 + 1j])
        assert mtl5.dot(a, b) == pytest.approx(np.vdot(a, b))
        assert mtl5.dot_real(a, b) == pytest.approx(np.dot(a, b))

    def test_matrix_factory_keeps_complex(self):
        A = mtl5.matrix(HERMITIAN)
        assert A.dtype == "c128"
        np.testing.assert_array_equal(A.to_numpy(), HERMITIAN)


class TestNorms:
    """Every norm of a complex container is real — a float, not a complex with
    a zero imaginary part."""

    def test_vector_norms(self, cdtype):
        np_dt, _, tol = cdtype
        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        v = mtl5.vector(a)
        assert isinstance(mtl5.norm(v, 2), float)
        assert mtl5.norm(v, 1) == pytest.approx(np.linalg.norm(a, 1), rel=tol)
        assert mtl5.norm(v, 2) == pytest.approx(np.linalg.norm(a, 2), rel=tol)
        assert mtl5.norm(v, -1) == pytest.approx(np.linalg.norm(a, np.inf), rel=tol)

    def test_one_norm_sums_magnitudes(self, cdtype):
        """sum|z|, not sum(|re| + |im|) — those differ for any non-axis value."""
        np_dt, _, tol = cdtype
        a = np.array([3 + 4j], dtype=np_dt)
        assert mtl5.norm(mtl5.vector(a), 1) == pytest.approx(5.0, rel=tol)

    def test_frobenius_norm(self, cdtype):
        np_dt, _, tol = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        ref = np.linalg.norm(HERMITIAN, "fro")
        assert mtl5.frobenius_norm(A) == pytest.approx(ref, rel=tol)
        assert mtl5.norm(A, 2) == pytest.approx(ref, rel=tol)

    def test_bad_ord_raises(self):
        with pytest.raises(ValueError, match="ord must be"):
            mtl5.norm(mtl5.vector(np.array([1j])), 3)


class TestTransposeVersusAdjoint:
    """MTL5's `trans` does not conjugate. For real elements .T and .H agree, so
    the difference only shows up here — pin it."""

    def test_T_is_a_plain_transpose(self, cdtype):
        np_dt, _, _ = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        np.testing.assert_allclose(A.T.to_numpy(), HERMITIAN.T)

    def test_H_is_the_conjugate_transpose(self, cdtype):
        np_dt, _, _ = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        np.testing.assert_allclose(A.H.to_numpy(), HERMITIAN.conj().T)

    def test_T_and_H_differ(self):
        A = mtl5.matrix(HERMITIAN)
        assert not np.allclose(A.T.to_numpy(), A.H.to_numpy())

    def test_free_functions_agree_with_the_properties(self):
        A = mtl5.matrix(HERMITIAN)
        np.testing.assert_allclose(mtl5.transpose(A).to_numpy(), A.T.to_numpy())
        np.testing.assert_allclose(mtl5.adjoint(A).to_numpy(), A.H.to_numpy())

    def test_conj(self, cdtype):
        np_dt, _, _ = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        np.testing.assert_allclose(A.conj().to_numpy(), HERMITIAN.conj())
        np.testing.assert_allclose(mtl5.conj(A).to_numpy(), HERMITIAN.conj())

        a = np.array([1 + 2j, 3 - 1j], dtype=np_dt)
        np.testing.assert_allclose(mtl5.vector(a).conj().to_numpy(), a.conj())
        np.testing.assert_allclose(mtl5.conj(mtl5.vector(a)).to_numpy(), a.conj())

    def test_hermitian_is_not_symmetric_for_complex(self):
        H = mtl5.matrix(HERMITIAN)
        assert mtl5.is_hermitian(H)
        assert not mtl5.is_symmetric(H)

        S = mtl5.matrix(SYMMETRIC)
        assert mtl5.is_symmetric(S)
        assert not mtl5.is_hermitian(S)


class TestProducts:
    def test_matvec(self, cdtype):
        np_dt, _, tol = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        x = mtl5.vector(X_TRUE.astype(np_dt))
        np.testing.assert_allclose(mtl5.matvec(A, x).to_numpy(), HERMITIAN @ X_TRUE, rtol=tol)

    def test_matmul(self, cdtype):
        np_dt, _, tol = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        np.testing.assert_allclose(mtl5.matmul(A, A).to_numpy(), HERMITIAN @ HERMITIAN, rtol=tol)

    def test_matmul_operator(self, cdtype):
        np_dt, _, tol = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        x = mtl5.vector(X_TRUE.astype(np_dt))
        np.testing.assert_allclose((A @ A).to_numpy(), HERMITIAN @ HERMITIAN, rtol=tol)
        np.testing.assert_allclose((A @ x).to_numpy(), HERMITIAN @ X_TRUE, rtol=tol)

    def test_shape_mismatch_raises(self):
        A = mtl5.matrix(np.ones((2, 3), dtype=np.complex128))
        with pytest.raises(ValueError, match="num_cols"):
            mtl5.matmul(A, A)
        with pytest.raises(ValueError, match="num_cols"):
            mtl5.matvec(A, mtl5.vector(np.ones(2, dtype=np.complex128)))


class TestSolve:
    def test_solve_recovers_the_known_solution(self, cdtype):
        np_dt, _, tol = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        b = mtl5.vector((HERMITIAN @ X_TRUE).astype(np_dt))
        np.testing.assert_allclose(mtl5.solve(A, b).to_numpy(), X_TRUE, rtol=tol, atol=tol)

    def test_solve_on_a_nonhermitian_system(self, cdtype):
        """LU is general — it must not need any symmetry."""
        np_dt, _, tol = cdtype
        rng = np.random.default_rng(0)
        M = (rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))).astype(np_dt)
        x = (rng.standard_normal(6) + 1j * rng.standard_normal(6)).astype(np_dt)
        b = (M @ x).astype(np_dt)
        got = mtl5.solve(mtl5.matrix(M), mtl5.vector(b)).to_numpy()
        np.testing.assert_allclose(got, np.linalg.solve(M, b), rtol=1e-4, atol=1e-4)

    def test_lu_factor_is_reusable(self, cdtype):
        np_dt, suffix, tol = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        f = mtl5.lu(A)
        assert f.dtype == suffix
        assert f.shape == (2, 2)
        assert f"LUFactor_{suffix}" in repr(f)
        for x in (X_TRUE, np.array([2 + 0j, 1 - 3j])):
            b = mtl5.vector((HERMITIAN @ x).astype(np_dt))
            np.testing.assert_allclose(f.solve(b).to_numpy(), x, rtol=tol, atol=tol)

    def test_inv(self, cdtype):
        np_dt, _, tol = cdtype
        A = mtl5.matrix(HERMITIAN.astype(np_dt))
        np.testing.assert_allclose(
            mtl5.inv(A).to_numpy(), np.linalg.inv(HERMITIAN), rtol=tol, atol=tol
        )

    def test_singular_matrix_raises(self):
        A = mtl5.matrix(np.zeros((2, 2), dtype=np.complex128))
        with pytest.raises(RuntimeError, match="[Ss]ingular"):
            mtl5.solve(A, mtl5.vector(np.ones(2, dtype=np.complex128)))

    def test_non_square_raises(self):
        A = mtl5.matrix(np.ones((2, 3), dtype=np.complex128))
        with pytest.raises(ValueError, match="square"):
            mtl5.inv(A)


class TestLdltIsTransposeNotHermitian:
    """MTL5's ldlt has no conjugation in it, so it computes A = L D L^T. That is
    right for a complex symmetric matrix and silently wrong for a Hermitian one
    — it returns info=0 and a bad answer. The binding refuses that input."""

    def test_correct_on_complex_symmetric(self, cdtype):
        np_dt, _, tol = cdtype
        A = mtl5.matrix(SYMMETRIC.astype(np_dt))
        b = mtl5.vector((SYMMETRIC @ X_TRUE).astype(np_dt))
        np.testing.assert_allclose(mtl5.ldlt_solve(A, b).to_numpy(), X_TRUE, rtol=tol, atol=tol)

    def test_hermitian_input_is_refused(self):
        A = mtl5.matrix(HERMITIAN)
        b = mtl5.vector(HERMITIAN @ X_TRUE)
        with pytest.raises(ValueError, match="LDL\\^T, not LDL\\^H"):
            mtl5.ldlt_solve(A, b)

    def test_the_refusal_is_not_theoretical(self):
        """The wrong answer LDL^T would have given is genuinely wrong, so the
        guard is load-bearing rather than defensive."""
        assert not np.allclose(
            np.linalg.solve(HERMITIAN, HERMITIAN @ X_TRUE),
            np.linalg.solve(HERMITIAN.T, HERMITIAN @ X_TRUE),
        )

    def test_general_matrix_is_refused(self):
        A = mtl5.matrix(np.array([[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]]))
        b = mtl5.vector(np.ones(2, dtype=np.complex128))
        with pytest.raises(ValueError, match="symmetric"):
            mtl5.ldlt_solve(A, b)


class TestUnsupportedOperationsRefuse:
    """Cholesky does not compile for complex in MTL5 (it compares a complex
    against a complex where a magnitude is meant), `ldlt` is LDL^T rather than
    LDL^H, and there is no complex eigensolver or SVD. Complex input must earn a
    clear message that names a working alternative, not a confusing one from
    deep inside.

    QR and LQ are deliberately absent from this list: MTL5 gained a complex
    Householder, and they are bound — see TestComplexQRAndLQ.
    """

    @pytest.mark.parametrize("fn", ["cholesky", "ldlt", "bunch_kaufman"])
    def test_factorizations_reject_complex_clearly(self, fn):
        with pytest.raises(TypeError, match="complex128"):
            getattr(mtl5, fn)(HERMITIAN)

    @pytest.mark.parametrize("fn", ["cholesky", "ldlt", "bunch_kaufman"])
    def test_an_mtl5_complex_matrix_is_refused_too(self, fn):
        """Not just NumPy input — a DenseMatrix_c128 takes a different branch."""
        with pytest.raises(TypeError, match="c128"):
            getattr(mtl5, fn)(mtl5.matrix(HERMITIAN))

    def test_the_message_points_somewhere_useful(self):
        with pytest.raises(TypeError, match="mtl5.solve"):
            mtl5.cholesky(HERMITIAN)


class TestComplexQRAndLQ:
    """MTL5 gained a complex Householder (upstream aa3b52c), so QR and LQ work
    for complex. Verified for the answer, not just the compile: the least-squares
    residual must be orthogonal to range(A), which is what distinguishes a solve
    that applies Q^H from one that applies Q^T. A Q^T implementation would
    reconstruct A = QR perfectly and still solve the wrong problem.
    """

    # 4x2 overdetermined, full column rank.
    A = np.array([[1 + 1j, 2 - 1j], [0 + 2j, 1 + 0j], [3 + 0j, 0 + 1j], [-1 + 1j, 2 + 2j]])

    def test_qr_reconstructs_a(self, cdtype):
        np_dt, suffix, tol = cdtype
        f = mtl5.qr(self.A.astype(np_dt))
        assert f.dtype == suffix
        assert f.shape == (4, 2)
        Q, R = f.Q.to_numpy(), f.R.to_numpy()
        np.testing.assert_allclose(Q @ R, self.A, rtol=1e-4, atol=1e-4)

    def test_qr_q_is_unitary(self, cdtype):
        """Unitary (Q^H Q = I), not merely orthogonal (Q^T Q = I) — for complex
        those are different matrices."""
        np_dt, _, _ = cdtype
        Q = mtl5.qr(self.A.astype(np_dt)).Q.to_numpy()
        np.testing.assert_allclose(Q.conj().T @ Q, np.eye(4), rtol=1e-4, atol=1e-4)

    def test_qr_solves_a_consistent_system_exactly(self, cdtype):
        np_dt, _, _ = cdtype
        x_true = np.array([1 - 2j, 0 + 1j])
        b = self.A @ x_true
        f = mtl5.qr(self.A.astype(np_dt))
        x = f.solve(mtl5.vector(b.astype(np_dt))).to_numpy()
        np.testing.assert_allclose(x, x_true, rtol=1e-4, atol=1e-4)

    def test_qr_least_squares_residual_is_orthogonal(self):
        """The load-bearing check. For an inconsistent system the residual must
        satisfy A^H (Ax - b) = 0; using Q^T would leave it non-orthogonal."""
        b = np.array([1 + 0j, 0 + 1j, 2 - 1j, -1 + 3j])
        x = mtl5.qr(self.A).solve(mtl5.vector(b)).to_numpy()
        residual = self.A @ x - b
        np.testing.assert_allclose(self.A.conj().T @ residual, np.zeros(2), atol=1e-12)

    def test_qr_least_squares_matches_numpy(self):
        b = np.array([1 + 0j, 0 + 1j, 2 - 1j, -1 + 3j])
        x = mtl5.qr(self.A).solve(mtl5.vector(b)).to_numpy()
        np.testing.assert_allclose(x, np.linalg.lstsq(self.A, b, rcond=None)[0], atol=1e-12)

    def test_qr_accepts_an_mtl5_matrix(self):
        f = mtl5.qr(mtl5.matrix(self.A))
        np.testing.assert_allclose(f.Q.to_numpy() @ f.R.to_numpy(), self.A, atol=1e-12)

    def test_qr_still_needs_rows_ge_cols(self):
        with pytest.raises(ValueError, match="num_rows >= num_cols"):
            mtl5.qr(self.A.T)

    def test_lq_reconstructs_a(self, cdtype):
        np_dt, suffix, _ = cdtype
        B = self.A.conj().T  # 2x4, underdetermined
        g = mtl5.lq(B.astype(np_dt))
        assert g.dtype == suffix
        L, Q = g.L.to_numpy(), g.Q.to_numpy()
        np.testing.assert_allclose(L @ Q, B, rtol=1e-4, atol=1e-4)

    def test_lq_q_is_unitary(self, cdtype):
        np_dt, _, _ = cdtype
        Q = mtl5.lq(self.A.conj().T.astype(np_dt)).Q.to_numpy()
        np.testing.assert_allclose(Q.conj().T @ Q, np.eye(4), rtol=1e-4, atol=1e-4)

    def test_lq_l_is_lower_trapezoidal(self):
        L = mtl5.lq(self.A.conj().T).L.to_numpy()
        np.testing.assert_allclose(np.triu(L, 1), np.zeros_like(np.triu(L, 1)), atol=1e-14)

    def test_the_factor_classes_are_exported(self):
        for name in ("QRFactor_c64", "QRFactor_c128", "LQFactor_c64", "LQFactor_c128"):
            assert name in mtl5.__all__, f"{name} missing from __all__"
            assert hasattr(mtl5, name)


class TestPublicSurface:
    def test_complex_names_are_exported(self):
        for name in (
            "DenseVector_c64",
            "DenseVector_c128",
            "DenseMatrix_c64",
            "DenseMatrix_c128",
            "LUFactor_c64",
            "LUFactor_c128",
            "adjoint",
            "conj",
            "dot_real",
            "frobenius_norm",
            "ldlt_solve",
        ):
            assert name in mtl5.__all__, f"{name} missing from __all__"
            assert hasattr(mtl5, name)

    def test_complex_is_not_advertised_by_dtypes(self):
        """dtypes() names what convert() accepts, and convert() has no complex
        target. Listing c128 there would send people to a TypeError."""
        assert "c128" not in mtl5.dtypes()
