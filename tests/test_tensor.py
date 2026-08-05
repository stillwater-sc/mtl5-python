"""Index-notation tensor algebra.

`TestContractionDispatch` is the load-bearing part. MTL5's `contract` takes its
index names as compile-time template parameters, so a runtime index string
cannot reach it. The binding enumerates the four rank2×rank2 patterns (the
repeated index sits in one of two positions on each side) and dispatches — which
means the risk is not that contraction is wrong, but that the *dispatch* picks
the wrong one of four correct kernels. Every pattern is therefore checked
against an explicit NumPy reference that distinguishes it from the other three.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5
import mtl5.tensor as mtt

# Deliberately non-symmetric, so transposing either operand changes the answer
# and a mis-dispatch cannot hide.
A = np.arange(1.0, 10.0).reshape(3, 3)
B = ((np.arange(3).reshape(3, 1) + 1) * (np.arange(3) + 2)).astype(float)
X = np.array([1.0, 2.0, 3.0])


class TestContainer:
    @pytest.mark.parametrize("dim", [2, 3, 4])
    @pytest.mark.parametrize("rank", [1, 2])
    def test_round_trip(self, rank, dim):
        a = np.arange(float(dim**rank)).reshape((dim,) * rank)
        np.testing.assert_allclose(mtt.asarray(a).to_numpy(), a)

    def test_shape_and_metadata(self):
        t = mtt.asarray(A)
        assert t.shape == [3, 3]
        assert t.dtype == "f64"
        assert t.size == 9

    def test_element_access(self):
        t = mtt.asarray(A)
        assert t[1, 2] == A[1, 2]
        t[0, 0] = 99.0
        assert t[0, 0] == 99.0

    def test_out_of_range(self):
        t = mtt.asarray(A)
        with pytest.raises(IndexError):
            t[3, 0]

    def test_float32(self):
        assert mtt.asarray(A.astype(np.float32)).dtype == "f32"

    def test_zeros(self):
        z = mtt.zeros(2, 4)
        assert z.shape == [4, 4]
        np.testing.assert_allclose(z.to_numpy(), np.zeros((4, 4)))

    def test_rank_4_exists_only_as_an_outer_product(self):
        """Nothing takes a rank-4 tensor as input, so it need only be
        constructible as a result — which is why rank 3 is absent entirely."""
        t = mtt.outer(mtt.asarray(A), mtt.asarray(A))
        assert t.shape == [3, 3, 3, 3]


class TestAsarrayValidation:
    def test_ragged_shape_is_refused(self):
        """A tensor has one dimension shared by every index."""
        with pytest.raises(ValueError, match="every extent must be equal"):
            mtt.asarray(np.zeros((3, 4)))

    def test_rank_3_is_refused_with_a_reason(self):
        with pytest.raises(ValueError, match=r"rank must be one of \(1, 2, 4\)"):
            mtt.asarray(np.zeros((3, 3, 3)))

    def test_unsupported_dimension(self):
        with pytest.raises(ValueError, match="dimension must be one of"):
            mtt.asarray(np.zeros((5, 5)))

    def test_integer_dtype_is_refused(self):
        with pytest.raises(TypeError, match="float32 or float64"):
            mtt.asarray(np.eye(3, dtype=np.int64))


class TestContractionDispatch:
    """Four correct kernels; the risk is picking the wrong one. Each reference
    below differs from the other three, so a mis-dispatch fails."""

    @pytest.mark.parametrize(
        ("sa", "sb", "ref"),
        [
            ("ij", "jk", A @ B),
            ("ij", "kj", A @ B.T),
            ("ji", "jk", A.T @ B),
            ("ji", "kj", A.T @ B.T),
        ],
    )
    def test_rank2_rank2(self, sa, sb, ref):
        got = mtt.contract(mtt.asarray(A), sa, mtt.asarray(B), sb).to_numpy()
        np.testing.assert_allclose(got, ref)

    def test_the_four_references_are_all_different(self):
        """Otherwise the parametrisation above would not distinguish them."""
        refs = [A @ B, A @ B.T, A.T @ B, A.T @ B.T]
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                assert not np.allclose(refs[i], refs[j])

    @pytest.mark.parametrize(("sa", "ref"), [("ij", A @ X), ("ji", A.T @ X)])
    def test_rank2_rank1(self, sa, ref):
        got = mtt.contract(mtt.asarray(A), sa, mtt.asarray(X), "j").to_numpy()
        np.testing.assert_allclose(got, ref)

    def test_index_letters_are_arbitrary(self):
        """Only which positions are shared matters, not the letters used."""
        a, b = mtt.asarray(A), mtt.asarray(B)
        np.testing.assert_allclose(
            mtt.contract(a, "pq", b, "qr").to_numpy(),
            mtt.contract(a, "ij", b, "jk").to_numpy(),
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_every_dimension(self, dim):
        a = np.arange(1.0, dim * dim + 1).reshape(dim, dim)
        got = mtt.contract(mtt.asarray(a), "ij", mtt.asarray(a), "jk").to_numpy()
        np.testing.assert_allclose(got, a @ a)


class TestContractionRejections:
    def test_no_shared_index(self):
        a, b = mtt.asarray(A), mtt.asarray(B)
        with pytest.raises(ValueError, match="share no index"):
            mtt.contract(a, "ij", b, "kl")

    def test_a_trace_is_named_as_such(self):
        """'ii' also reads as a doubled shared index, so the check order matters
        — 'this is a trace' is the more useful message."""
        a, b = mtt.asarray(A), mtt.asarray(B)
        with pytest.raises(ValueError, match="trace"):
            mtt.contract(a, "ii", b, "ij")

    def test_two_shared_indices(self):
        a, b = mtt.asarray(A), mtt.asarray(B)
        with pytest.raises(ValueError, match="more than one index"):
            mtt.contract(a, "ij", b, "ji")

    def test_index_string_length_must_match_rank(self):
        a = mtt.asarray(A)
        with pytest.raises(ValueError, match="match the ranks"):
            mtt.contract(a, "i", a, "ij")


class TestOuter:
    def test_rank1_gives_rank2(self):
        x = mtt.asarray(X)
        np.testing.assert_allclose(mtt.outer(x, x).to_numpy(), np.outer(X, X))

    def test_rank2_gives_rank4(self):
        a = mtt.asarray(A)
        got = mtt.outer(a, a).to_numpy()
        assert got.shape == (3, 3, 3, 3)
        np.testing.assert_allclose(got, np.einsum("ij,kl->ijkl", A, A))

    def test_outer_is_not_symmetric_in_its_arguments(self):
        x, y = mtt.asarray(X), mtt.asarray(np.array([4.0, 5.0, 6.0]))
        assert not np.allclose(mtt.outer(x, y).to_numpy(), mtt.outer(y, x).to_numpy())


class TestMetric:
    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_euclidean_is_the_identity(self, dim):
        np.testing.assert_allclose(mtt.euclidean_metric(dim).to_numpy(), np.eye(dim))

    def test_minkowski_signature(self):
        """Signature (-, +, +, +): the time component flips sign."""
        g = mtt.minkowski_metric().to_numpy()
        np.testing.assert_allclose(np.diag(g), [-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(g, np.diag(np.diag(g)))

    def test_lowering_with_euclidean_changes_nothing(self):
        x = mtt.asarray(X)
        got = mtt.lower_index(x, mtt.euclidean_metric(3)).to_numpy()
        np.testing.assert_allclose(got, X)

    def test_lowering_with_minkowski_flips_the_time_component(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        got = mtt.lower_index(mtt.asarray(v), mtt.minkowski_metric()).to_numpy()
        np.testing.assert_allclose(got, [-1.0, 2.0, 3.0, 4.0])

    def test_raise_then_lower_is_the_identity_for_minkowski(self):
        """The Minkowski metric is its own inverse, so applying it twice returns
        the original — which is what makes raise/lower inverse operations."""
        v = np.array([1.0, 2.0, 3.0, 4.0])
        g = mtt.minkowski_metric()
        once = mtt.lower_index(mtt.asarray(v), g)
        twice = mtt.raise_index(once, g).to_numpy()
        np.testing.assert_allclose(twice, v)

    def test_rank2_index_operations(self):
        g = mtt.minkowski_metric()
        t = np.arange(16.0).reshape(4, 4)
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(mtt.lower_first(mtt.asarray(t), g).to_numpy(), eta @ t)
        np.testing.assert_allclose(mtt.lower_second(mtt.asarray(t), g).to_numpy(), t @ eta)


class TestProperties:
    def test_symmetric(self):
        sym = np.array([[0.0, 1, 2], [1, 0, 3], [2, 3, 0]])
        assert mtt.is_symmetric(mtt.asarray(sym))
        assert not mtt.is_symmetric(mtt.asarray(A))

    def test_antisymmetric(self):
        asym = np.array([[0.0, 1, 2], [-1, 0, 3], [-2, -3, 0]])
        assert mtt.is_antisymmetric(mtt.asarray(asym))
        assert not mtt.is_antisymmetric(mtt.asarray(A))

    def test_antisymmetry_requires_a_vanishing_diagonal(self):
        near = np.array([[1.0, 1, 2], [-1, 0, 3], [-2, -3, 0]])
        assert not mtt.is_antisymmetric(mtt.asarray(near))

    def test_tolerance(self):
        almost = np.array([[0.0, 1, 2], [1 + 1e-9, 0, 3], [2, 3, 0]])
        assert not mtt.is_symmetric(mtt.asarray(almost))
        assert mtt.is_symmetric(mtt.asarray(almost), tol=1e-6)


class TestSymmetricStorage:
    def test_packed_size(self):
        """D(D+1)/2 stored against D^2 logical — the point of the type."""
        core = mtl5._core.tensor
        for dim, stored in ((2, 3), (3, 6), (4, 10)):
            S = getattr(core, f"SymmetricTensor_d{dim}_f64")()
            assert S.num_stored == stored
            assert S.dimension == dim

    def test_writing_one_triangle_sets_both(self):
        S = mtl5._core.tensor.SymmetricTensor_d3_f64()
        S[0, 1] = 5.0
        assert S[1, 0] == 5.0

    def test_to_dense_is_symmetric(self):
        S = mtl5._core.tensor.SymmetricTensor_d3_f64()
        S[0, 1] = 5.0
        S[0, 2] = -2.0
        dense = S.to_dense().to_numpy()
        np.testing.assert_allclose(dense, dense.T)
        assert dense[1, 0] == 5.0

    def test_out_of_range(self):
        S = mtl5._core.tensor.SymmetricTensor_d3_f64()
        with pytest.raises(IndexError):
            S[3, 0]


class TestTheDocstringExamplesRun:
    """The module docstring shipped with a `lower_index(x_3d, minkowski_metric())`
    call, which cannot work — the metric is 4-D. Caught in review rather than by
    the suite, so the examples are executed here."""

    def test_module_docstring_examples(self):
        A_ = mtt.asarray(np.arange(9.0).reshape(3, 3))
        x_ = mtt.asarray(np.array([1.0, 2.0, 3.0]))
        mtt.contract(A_, "ij", x_, "j")
        mtt.contract(A_, "ji", x_, "j")
        mtt.lower_index(x_, mtt.euclidean_metric(3))

        v = mtt.asarray(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(
            mtt.lower_index(v, mtt.minkowski_metric()).to_numpy(), [-1.0, 2, 3, 4]
        )

    def test_a_metric_of_the_wrong_dimension_is_refused(self):
        x_ = mtt.asarray(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(TypeError):
            mtt.lower_index(x_, mtt.minkowski_metric())


class TestPublicSurface:
    def test_tensor_is_exported(self):
        assert "tensor" in mtl5.__all__
        assert hasattr(mtl5, "tensor")

    def test_ranks_and_dimensions_are_reported(self):
        assert mtt.ranks() == [1, 2, 4]
        assert mtt.dimensions() == [2, 3, 4]

    def test_all_names_resolve(self):
        for name in mtt.__all__:
            assert hasattr(mtt, name), f"mtl5.tensor.{name} missing"
