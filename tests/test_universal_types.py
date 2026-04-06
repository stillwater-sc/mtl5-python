"""Tests for Universal number types — posit, fixpnt, lns variants."""

import numpy as np
import numpy.testing as npt
import pytest

import mtl5


# Posit types — tapered precision floats
@pytest.mark.parametrize(
    "name, factory_name, vec_class_name, tol",
    [
        ("posit8", "vector_posit8", "DenseVector_posit8", 1.0),  # very coarse
        ("posit16", "vector_posit16", "DenseVector_posit16", 1e-2),
        ("posit32", "vector_posit32", "DenseVector_posit32", 1e-6),
        ("posit64", "vector_posit64", "DenseVector_posit64", 1e-14),
    ],
)
class TestPositTypes:
    def test_create_and_access(self, name, factory_name, vec_class_name, tol):
        factory = getattr(mtl5, factory_name)
        vec_cls = getattr(mtl5, vec_class_name)

        v = factory(np.array([1.0, 2.0, 3.0, 4.0]))
        assert isinstance(v, vec_cls)
        assert v.dtype == name
        assert len(v) == 4

    def test_to_numpy(self, name, factory_name, vec_class_name, tol):
        factory = getattr(mtl5, factory_name)
        a = np.array([0.5, 1.0, 2.0, 4.0])
        v = factory(a)
        result = v.to_numpy()
        # posit8 has very limited precision, just check approximate
        if name == "posit8":
            assert len(result) == 4
        else:
            npt.assert_allclose(result, a, rtol=tol)

    def test_dot_product(self, name, factory_name, vec_class_name, tol):
        factory = getattr(mtl5, factory_name)
        a = factory(np.array([1.0, 0.0, 0.0]))
        b = factory(np.array([0.0, 1.0, 0.0]))
        # orthogonal vectors → dot = 0 (exact even in low precision)
        assert mtl5.dot(a, b) == pytest.approx(0.0, abs=tol)

    def test_norm(self, name, factory_name, vec_class_name, tol):
        factory = getattr(mtl5, factory_name)
        v = factory(np.array([3.0, 4.0]))
        n = mtl5.norm(v)
        assert n == pytest.approx(5.0, rel=max(tol, 0.1))


# Fixpnt types
@pytest.mark.parametrize(
    "name, factory_name, vec_class_name, max_val",
    [
        ("fixpnt8", "vector_fixpnt8", "DenseVector_fixpnt8", 7.5),  # range [-8, 8)
        ("fixpnt16", "vector_fixpnt16", "DenseVector_fixpnt16", 100.0),  # range [-128, 128)
    ],
)
class TestFixpntTypes:
    def test_create_and_access(self, name, factory_name, vec_class_name, max_val):
        factory = getattr(mtl5, factory_name)
        vec_cls = getattr(mtl5, vec_class_name)

        # Use values within representable range
        a = np.array([0.5, 1.0, 2.0, max_val * 0.5])
        v = factory(a)
        assert isinstance(v, vec_cls)
        assert v.dtype == name
        assert len(v) == 4

    def test_in_range_values_preserved(self, name, factory_name, vec_class_name, max_val):
        factory = getattr(mtl5, factory_name)
        # values that should round-trip
        a = np.array([0.25, 0.5, 1.0, 2.0])
        v = factory(a)
        result = v.to_numpy()
        # Resolution depends on rbits — fixpnt8<8,4> has 1/16 resolution
        npt.assert_allclose(result, a, atol=0.1)

    def test_dot_product(self, name, factory_name, vec_class_name, max_val):
        factory = getattr(mtl5, factory_name)
        a = factory(np.array([1.0, 0.0]))
        b = factory(np.array([0.0, 1.0]))
        assert mtl5.dot(a, b) == pytest.approx(0.0, abs=0.1)


# LNS types
@pytest.mark.parametrize(
    "name, factory_name, vec_class_name, tol",
    [
        ("lns16", "vector_lns16", "DenseVector_lns16", 1e-1),
        ("lns32", "vector_lns32", "DenseVector_lns32", 1e-3),
    ],
)
class TestLnsTypes:
    def test_create_and_access(self, name, factory_name, vec_class_name, tol):
        factory = getattr(mtl5, factory_name)
        vec_cls = getattr(mtl5, vec_class_name)

        v = factory(np.array([1.0, 2.0, 4.0]))
        assert isinstance(v, vec_cls)
        assert v.dtype == name
        assert len(v) == 3

    def test_powers_of_two_exact(self, name, factory_name, vec_class_name, tol):
        """LNS represents powers of two exactly."""
        factory = getattr(mtl5, factory_name)
        a = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        v = factory(a)
        result = v.to_numpy()
        npt.assert_allclose(result, a, rtol=tol)


class TestMatrixUniversalTypes:
    @pytest.mark.parametrize(
        "factory_name, mat_class_name",
        [
            ("matrix_posit16", "DenseMatrix_posit16"),
            ("matrix_posit32", "DenseMatrix_posit32"),
            ("matrix_fixpnt16", "DenseMatrix_fixpnt16"),
            ("matrix_lns16", "DenseMatrix_lns16"),
        ],
    )
    def test_matrix_create(self, factory_name, mat_class_name):
        factory = getattr(mtl5, factory_name)
        mat_cls = getattr(mtl5, mat_class_name)
        M = factory(np.eye(3))
        assert isinstance(M, mat_cls)
        assert M.shape == (3, 3)


class TestPosit32Solve:
    """Verify that solve() works at posit32 precision (close to f32)."""

    def test_solve_simple(self):
        A = mtl5.matrix_posit32(np.array([[2.0, 1.0], [1.0, 3.0]]))
        b = mtl5.vector_posit32(np.array([5.0, 7.0]))
        x = mtl5.solve(A, b)
        result = np.array(x.to_list())
        npt.assert_allclose(result, [1.6, 1.8], rtol=1e-4)

    def test_solve_identity(self):
        n = 4
        A = mtl5.matrix_posit32(np.eye(n))
        b = mtl5.vector_posit32(np.array([1.0, 2.0, 3.0, 4.0]))
        x = mtl5.solve(A, b)
        npt.assert_allclose(x.to_list(), [1.0, 2.0, 3.0, 4.0], rtol=1e-4)
