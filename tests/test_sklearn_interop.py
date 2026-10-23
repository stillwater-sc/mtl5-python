"""scikit-learn estimators on MTL5 containers (#17).

scikit-learn has no dtype concept of its own: every estimator validates its
input with ``check_array``, which calls ``np.asarray``. So what decides whether
an MTL5 container works is the array protocol (``mtl5/_array_protocol.py``), not
the estimator. Each test compares against the same estimator fitted on the
NumPy array the container came from.

What does NOT work, and why:

* **MTL5 sparse matrices.** They have no ``__array__`` on purpose, since it
  would densify them silently, so scikit-learn rejects them. Pass
  ``mtl5.sparse.to_scipy(S)``, which every sparse-aware estimator takes.
* **Computing in a Universal type.** A posit or fp16 container is accepted, but
  scikit-learn computes on the float64 copy. The storage is reduced-precision;
  the arithmetic is not.
* **float32 kept as float32, without help.** An estimator that takes either
  float type (Ridge) computes a float32 container in float64, because the
  container's ``dtype`` is a name, not a NumPy dtype. ``np.asarray(M)`` first
  is zero-copy and keeps float32.

Estimators that call LAPACK (PCA, SVC) are not a failure case. The issue
expected them to be, but they never see a posit: by the time LAPACK runs, the
data is float64.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import pairwise_distances  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

import mtl5  # noqa: E402

rng = np.random.default_rng(17)
X = rng.standard_normal((60, 4))
W = np.array([1.5, -2.0, 0.5, 3.0])
y = X @ W + 0.25 + 0.01 * rng.standard_normal(60)


@pytest.fixture(params=[np.float64, np.float32], ids=["f64", "f32"])
def data(request):
    dt = request.param
    return X.astype(dt), y.astype(dt)


class TestRegression:
    def test_linear_regression_fit_and_predict(self, data):
        Xd, yd = data
        ref = LinearRegression().fit(Xd, yd)
        got = LinearRegression().fit(mtl5.matrix(Xd), mtl5.vector(yd))
        np.testing.assert_array_equal(got.coef_, ref.coef_)
        assert got.intercept_ == ref.intercept_
        np.testing.assert_array_equal(got.predict(mtl5.matrix(Xd)), ref.predict(Xd))
        np.testing.assert_allclose(got.coef_, W, atol=1e-2)

    def test_ridge(self):
        ref = Ridge(alpha=0.5).fit(X, y)
        got = Ridge(alpha=0.5).fit(mtl5.matrix(X), mtl5.vector(y))
        np.testing.assert_array_equal(got.coef_, ref.coef_)

    def test_float32_is_upcast_unless_converted_first(self):
        """The container's ``dtype`` is a name ("f32"), not a NumPy dtype, so
        check_array cannot see that the input is float32. An estimator that
        accepts either float type then picks float64. ``np.asarray`` first is
        zero-copy and keeps float32."""
        X32, y32 = X.astype(np.float32), y.astype(np.float32)
        upcast = Ridge(alpha=0.5).fit(mtl5.matrix(X32), y32)
        assert upcast.coef_.dtype == np.float64
        np.testing.assert_array_equal(
            upcast.coef_, Ridge(alpha=0.5).fit(X32.astype(np.float64), y32).coef_
        )
        kept = Ridge(alpha=0.5).fit(np.asarray(mtl5.matrix(X32)), y32)
        np.testing.assert_array_equal(kept.coef_, Ridge(alpha=0.5).fit(X32, y32).coef_)

    def test_vector_target_with_numpy_features(self):
        got = LinearRegression().fit(X, mtl5.vector(y))
        np.testing.assert_array_equal(got.coef_, LinearRegression().fit(X, y).coef_)

    def test_fit_leaves_the_borrowed_array_alone(self):
        Xc = X.copy()
        make_pipeline(StandardScaler(), Ridge()).fit(mtl5.matrix(Xc), y)
        np.testing.assert_array_equal(Xc, X)

    def test_pipeline(self):
        ref = make_pipeline(StandardScaler(), Ridge()).fit(X, y)
        got = make_pipeline(StandardScaler(), Ridge()).fit(mtl5.matrix(X), mtl5.vector(y))
        np.testing.assert_array_equal(got.predict(mtl5.matrix(X)), ref.predict(X))


class TestDistances:
    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_pairwise_distances(self, metric):
        np.testing.assert_array_equal(
            pairwise_distances(mtl5.matrix(X), metric=metric), pairwise_distances(X, metric=metric)
        )

    def test_pairwise_distances_between_two_containers(self):
        A, B = mtl5.matrix(X[:10].copy()), mtl5.matrix(X[10:25].copy())
        np.testing.assert_array_equal(
            pairwise_distances(A, B), pairwise_distances(X[:10], X[10:25])
        )


class TestLapackBound:
    """These run: LAPACK sees float64, never a posit."""

    def test_pca(self):
        ref = PCA(n_components=2).fit(X)
        got = PCA(n_components=2).fit(mtl5.matrix(X))
        np.testing.assert_allclose(got.explained_variance_ratio_, ref.explained_variance_ratio_)

    def test_svc(self):
        labels = (y > 0).astype(int)
        ref = SVC().fit(X, labels)
        got = SVC().fit(mtl5.matrix(X), labels)
        np.testing.assert_array_equal(got.predict(mtl5.matrix(X)), ref.predict(X))


class TestUniversalStorage:
    def test_posit16_features_fit_on_the_rounded_values(self):
        """Accepted, and identical to fitting the float64 values of the posits."""
        P = mtl5.convert(X, "posit16")
        got = LinearRegression().fit(P, y)
        ref = LinearRegression().fit(P.to_numpy(), y)
        np.testing.assert_array_equal(got.coef_, ref.coef_)
        np.testing.assert_allclose(got.coef_, W, atol=1e-2)


class TestNotSupported:
    def test_mtl5_sparse_is_rejected(self):
        sp = pytest.importorskip("scipy.sparse")
        import mtl5.sparse as msp

        S = msp.from_scipy(sp.random(60, 4, density=0.5, format="csr", random_state=0))
        with pytest.raises(TypeError):
            Ridge().fit(S, y)

    def test_mtl5_sparse_through_scipy(self):
        sp = pytest.importorskip("scipy.sparse")
        import mtl5.sparse as msp

        Ssp = sp.random(60, 4, density=0.5, format="csr", random_state=0)
        got = Ridge().fit(msp.to_scipy(msp.from_scipy(Ssp)), y)
        np.testing.assert_allclose(got.coef_, Ridge().fit(Ssp, y).coef_)
