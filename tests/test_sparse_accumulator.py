"""Accumulator policy on the sparse direct factorizations.

`sparse_lu_numeric`, `native_klu_factor`, `supernodal_lu_numeric` and
`supernodal_ldlt_numeric` each take an `Accumulator` template parameter typing
their dense numeric workspace. A float32 factor can therefore accumulate its
updates in float64: the factor stays narrow, but the arithmetic that produced it
is wider.

The load-bearing test here is `TestItActuallyChangesTheAnswer`. A parameter that
is plumbed through but makes no numerical difference is worse than no parameter
at all — it reads as a capability while being decoration. These tests fail if
the accumulator ever stops reaching the kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

sp = pytest.importorskip("scipy.sparse")
import mtl5.sparse as ms  # noqa: E402

FACTORIES = ["splu", "klu", "supernodal_lu", "supernodal_ldlt"]


def hard_matrix(n: int = 400):
    """A badly scaled sparse matrix — ill-conditioned enough that intermediate
    rounding in the update chains is visible in the answer."""
    rng = np.random.default_rng(1)
    M = sp.random(n, n, density=0.02, random_state=2, format="csr") + sp.eye(n) * 1e-3
    D = sp.diags(10.0 ** rng.uniform(-4, 4, n))
    A = (D @ M).tocsr()
    x_true = rng.standard_normal(n)
    return A, x_true, A.astype(np.float64) @ x_true


def spd_matrix(k: int = 12):
    """Symmetric positive definite, for the LDL^T path."""
    return ms.to_scipy(mtl5.generators.laplacian_2d(k, k))


def as_f32(csr):
    return ms.from_scipy(csr.astype(np.float32))


def factor(name, A32, **kw):
    return getattr(ms, name)(A32, **kw)


class TestTheParameterIsAccepted:
    @pytest.mark.parametrize("name", FACTORIES)
    def test_default_is_reported(self, name):
        A32 = as_f32(spd_matrix())
        assert factor(name, A32).accumulator == "default"

    @pytest.mark.parametrize("name", FACTORIES)
    @pytest.mark.parametrize("acc", ["f32", "f64", "fma32", "fma64"])
    def test_each_accumulator_is_accepted_and_reported(self, name, acc):
        A32 = as_f32(spd_matrix())
        assert factor(name, A32, accumulator=acc).accumulator == acc

    @pytest.mark.parametrize("name", FACTORIES)
    def test_solves_correctly_with_a_wider_accumulator(self, name):
        """Whatever the policy, the answer must still be a solution."""
        csr = spd_matrix()
        A64 = csr.astype(np.float64)
        rng = np.random.default_rng(0)
        x_true = rng.standard_normal(csr.shape[0])
        b = A64 @ x_true
        x = factor(name, as_f32(csr), accumulator="f64").solve(b)
        np.testing.assert_allclose(x, x_true, rtol=1e-3, atol=1e-3)


class TestItActuallyChangesTheAnswer:
    """The point of the parameter. If these stop failing to differ, the
    accumulator has stopped reaching the kernel."""

    @pytest.mark.parametrize("order", ["colamd", "amd", "natural"])
    def test_f64_accumulation_beats_f32_on_an_ill_conditioned_matrix(self, order):
        """The direction is the robust claim, and it holds for every ordering.
        The magnitude is not: it ranges from about 1.3x to 3.3x depending on
        ordering and platform, so only a modest margin is asserted here."""
        A, x_true, b = hard_matrix()
        A32 = as_f32(A)
        errs = {}
        for acc in ("f32", "f64"):
            x = ms.splu(A32, ordering=order, accumulator=acc).solve(b)
            errs[acc] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        assert errs["f64"] < errs["f32"], errs

    def test_the_improvement_is_substantial_somewhere(self):
        """Direction alone could be noise. Somewhere across the orderings the
        gain must be unmistakable — it is 1.3x-3.3x depending on platform, so
        require the best of the three to clear 1.3x."""
        A, x_true, b = hard_matrix()
        A32 = as_f32(A)
        gains = []
        for order in ("colamd", "amd", "natural"):
            e = {}
            for acc in ("f32", "f64"):
                x = ms.splu(A32, ordering=order, accumulator=acc).solve(b)
                e[acc] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
            gains.append(e["f32"] / e["f64"])
        assert max(gains) > 1.3, f"no ordering showed a clear gain: {gains}"

    def test_the_two_factors_are_not_bit_identical(self):
        """The narrowest statement of 'it reached the kernel'."""
        A, _, b = hard_matrix()
        A32 = as_f32(A)
        x32 = ms.splu(A32, ordering="natural", accumulator="f32").solve(b)
        x64 = ms.splu(A32, ordering="natural", accumulator="f64").solve(b)
        assert not np.array_equal(x32, x64)

    def test_default_matches_the_element_type(self):
        """default == accumulate in the element type, so f32 elements give the
        same answer as an explicit 'f32'."""
        A, _, b = hard_matrix()
        A32 = as_f32(A)
        x_def = ms.splu(A32, ordering="natural").solve(b)
        x_f32 = ms.splu(A32, ordering="natural", accumulator="f32").solve(b)
        np.testing.assert_array_equal(x_def, x_f32)

    def test_fma64_tracks_f64(self):
        """They agree to well within the float32 factor's own accuracy. Not
        bitwise: a fused multiply-add rounds once where separate operations
        round twice, so an FMA accumulator is entitled to differ slightly and
        still be correct."""
        A, _, b = hard_matrix()
        A32 = as_f32(A)
        x_f64 = ms.splu(A32, ordering="natural", accumulator="f64").solve(b)
        x_fma = ms.splu(A32, ordering="natural", accumulator="fma64").solve(b)
        np.testing.assert_allclose(x_fma, x_f64, rtol=1e-6, atol=1e-9)

    def test_it_helps_iterative_refinement_converge(self):
        """The actual use case: factor narrow, refine in double. A better factor
        needs fewer corrections and lands closer."""
        A, x_true, b = hard_matrix()
        A32 = as_f32(A)
        A64 = ms.from_scipy(A.astype(np.float64))
        out = {}
        for acc in ("f32", "f64"):
            f = ms.splu(A32, ordering="natural", accumulator=acc)
            x, info = mtl5.mixed.iterative_refine(A64, f, b, max_iter=50, rel_tol=1e-14)
            out[acc] = (
                info["iters"],
                np.linalg.norm(x - x_true) / np.linalg.norm(x_true),
            )
        assert out["f64"][0] <= out["f32"][0], f"iterations should not increase: {out}"
        assert out["f64"][1] < out["f32"][1], f"refined error should improve: {out}"


class TestAccumulatorNamesAreCanonical:
    """`parse_acc` accepts aliases; the stored name must be the canonical one,
    or two spellings of the same policy behave differently — 'none' used to be
    stored verbatim and then rejected by refactor's default-only check."""

    @pytest.mark.parametrize(
        ("given", "canonical"),
        [
            ("none", "default"),
            ("default", "default"),
            ("float32", "f32"),
            ("float64", "f64"),
            ("fma", "fma64"),
            ("fma64", "fma64"),
        ],
    )
    def test_aliases_normalize(self, given, canonical):
        A32 = as_f32(spd_matrix())
        assert ms.splu(A32, accumulator=given).accumulator == canonical

    def test_none_is_the_default_policy_for_refactor_too(self):
        A32 = as_f32(spd_matrix())
        f = ms.splu(A32, accumulator="none")
        f.refactor(A32)  # must not raise: 'none' IS the default policy
        assert f.accumulator == "default"


class TestRejections:
    @pytest.mark.parametrize("name", FACTORIES)
    @pytest.mark.parametrize("acc", ["f32", "fma32"])
    def test_a_narrower_accumulator_than_the_element_is_refused(self, name, acc):
        """Accumulating float64 data in float32 loses precision rather than
        gaining it, so it is a mistake worth naming."""
        A64 = ms.from_scipy(spd_matrix().astype(np.float64))
        with pytest.raises(ValueError, match="narrower than the float64"):
            factor(name, A64, accumulator=acc)

    @pytest.mark.parametrize("name", FACTORIES)
    def test_quire_is_refused_with_a_reason(self, name):
        """SparseMatrix is float32/float64 only, and Universal defines a quire
        only for its own number systems."""
        A32 = as_f32(spd_matrix())
        with pytest.raises(ValueError, match="quire.*not available"):
            factor(name, A32, accumulator="quire")

    @pytest.mark.parametrize("name", FACTORIES)
    def test_an_unknown_name_lists_the_valid_ones(self, name):
        A32 = as_f32(spd_matrix())
        with pytest.raises(ValueError, match="unknown accumulator"):
            factor(name, A32, accumulator="float128")

    def test_f64_accumulator_on_f64_elements_is_fine(self):
        A64 = ms.from_scipy(spd_matrix().astype(np.float64))
        assert ms.splu(A64, accumulator="f64").accumulator == "f64"
        assert ms.splu(A64, accumulator="fma64").accumulator == "fma64"


class TestRefactor:
    """refactor() replays a stored pivot sequence through an upstream entry
    point that takes no accumulator. Silently dropping back to element precision
    would undo what the caller asked for, so it refuses instead — except for
    supernodal_ldlt, whose refactor re-runs the full numeric factorization."""

    @pytest.mark.parametrize("name", ["splu", "klu", "supernodal_lu"])
    def test_refactor_refuses_a_non_default_accumulator(self, name):
        A32 = as_f32(spd_matrix())
        f = factor(name, A32, accumulator="f64")
        with pytest.raises(ValueError, match="accumulator='f64'"):
            f.refactor(A32)

    @pytest.mark.parametrize("name", ["splu", "klu", "supernodal_lu"])
    def test_refactor_still_works_by_default(self, name):
        """The fast path must be untouched for everyone not using the policy."""
        csr = spd_matrix()
        A32 = as_f32(csr)
        f = factor(name, A32)
        f.refactor(A32)
        assert f.accumulator == "default"

    def test_supernodal_ldlt_refactor_carries_the_policy(self):
        A32 = as_f32(spd_matrix())
        f = ms.supernodal_ldlt(A32, accumulator="f64")
        f.refactor(A32)
        assert f.accumulator == "f64"

    def test_supernodal_ldlt_refactor_gives_the_accumulated_answer(self):
        """Not just that it runs — that it produces the f64-accumulated factor
        rather than quietly reverting."""
        csr = spd_matrix()
        A32 = as_f32(csr)
        A64 = csr.astype(np.float64)
        rng = np.random.default_rng(3)
        x_true = rng.standard_normal(csr.shape[0])
        b = A64 @ x_true

        fresh = ms.supernodal_ldlt(A32, accumulator="f64")
        reused = ms.supernodal_ldlt(A32, accumulator="f64")
        reused.refactor(A32)
        np.testing.assert_array_equal(fresh.solve(b), reused.solve(b))


class TestDefaultsAreUnchanged:
    """Everything above is additive; the existing behaviour must not move."""

    @pytest.mark.parametrize("name", FACTORIES)
    def test_positional_signature_still_works(self, name):
        A32 = as_f32(spd_matrix())
        assert factor(name, A32).accumulator == "default"

    def test_solve_is_unchanged_without_the_parameter(self):
        csr = spd_matrix()
        A64 = csr.astype(np.float64)
        rng = np.random.default_rng(5)
        x_true = rng.standard_normal(csr.shape[0])
        b = A64 @ x_true
        np.testing.assert_allclose(ms.splu(ms.from_scipy(A64)).solve(b), x_true, rtol=1e-9)
