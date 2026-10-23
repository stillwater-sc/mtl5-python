"""np.asarray() on the dense containers (#17).

What has to hold is the difference between the two kinds of container: a
native one hands NumPy a view of its own storage, a Universal one a float64
copy, and asking the second for no copy is an error rather than a silent copy.

The ``copy=False`` tests call ``__array__`` directly: ``np.asarray`` gained
``copy=`` only in NumPy 2.0, and the package supports NumPy 1.24.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

X = np.arange(1.0, 7.0).reshape(2, 3)

NATIVE = [np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]


class TestNative:
    @pytest.mark.parametrize("dt", NATIVE)
    def test_matrix_is_a_view_of_the_borrowed_array(self, dt):
        X_ = X.astype(dt)
        a = np.asarray(mtl5.matrix(X_))
        assert a.dtype == dt and a.shape == X.shape
        assert np.shares_memory(a, X_)

    @pytest.mark.parametrize("dt", NATIVE)
    def test_vector_is_a_view_of_the_borrowed_array(self, dt):
        x = X[0].astype(dt)
        a = np.asarray(mtl5.vector(x))
        assert a.dtype == dt
        assert np.shares_memory(a, x)

    def test_owning_matrix_converts(self):
        """A container that owns its storage, here a product, converts too."""
        C = mtl5.matrix(X) @ mtl5.matrix(X.T.copy())
        np.testing.assert_allclose(np.asarray(C), X @ X.T)

    def test_writes_through_the_view_reach_the_container(self):
        M = mtl5.matrix_copy(X)
        np.asarray(M)[0, 0] = -1.0
        assert M[0, 0] == -1.0

    def test_copy_true_detaches(self):
        X_ = X.copy()
        a = np.array(mtl5.matrix(X_), copy=True)
        assert not np.shares_memory(a, X_)

    def test_copy_false_is_honoured(self):
        X_ = X.copy()
        assert np.shares_memory(mtl5.matrix(X_).__array__(copy=False), X_)

    def test_dtype_request_converts(self):
        a = np.asarray(mtl5.matrix(X), dtype=np.float32)
        assert a.dtype == np.float32
        np.testing.assert_array_equal(a, X.astype(np.float32))

    def test_dtype_request_without_copy_raises(self):
        with pytest.raises(ValueError, match="needs a copy"):
            mtl5.matrix(X).__array__(np.float32, copy=False)


class TestUniversal:
    @pytest.mark.parametrize("name", ["posit16", "fp16", "lns32"])
    def test_matrix_converts_to_float64(self, name):
        P = mtl5.convert(X, name)
        a = np.asarray(P)
        assert a.dtype == np.float64 and a.shape == X.shape
        np.testing.assert_array_equal(a, P.to_numpy())

    def test_vector_converts_to_float64(self):
        p = mtl5.convert(X[0].copy(), "posit16")
        np.testing.assert_array_equal(np.asarray(p), p.to_numpy())

    def test_copy_false_raises(self):
        with pytest.raises(ValueError, match="to_universal"):
            mtl5.convert(X, "posit16").__array__(copy=False)

    def test_dtype_request(self):
        assert np.asarray(mtl5.convert(X, "posit16"), dtype=np.float32).dtype == np.float32


def test_every_dense_class_has_the_protocol():
    names = [n for n in dir(mtl5._core) if n.startswith(("DenseVector_", "DenseMatrix_"))]
    assert names
    assert all(hasattr(getattr(mtl5._core, n), "__array__") for n in names)


def test_sparse_does_not_densify():
    """np.asarray on a sparse matrix would densify it without saying so."""
    sp = pytest.importorskip("scipy.sparse")
    import mtl5.sparse as msp

    S = msp.from_scipy(sp.random(5, 4, density=0.4, format="csr", random_state=0))
    assert not hasattr(S, "__array__")
