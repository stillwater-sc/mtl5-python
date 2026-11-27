"""NumPy input to the dense solvers keeps its precision, or is refused (#152).

nanobind's converting pass used to hand an integer, float16 or non-contiguous
array to the first overload that accepts a cast, the float32 one. An int64
matrix, and a float64 matrix in Fortran order, both came back in single
precision with no warning. Now any layout is made contiguous with the dtype
kept, and a dtype the C++ overloads do not take is refused, as the
factorizations (qr, ldlt, ...) already refuse it.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5

EPS = 2.0**-30  # survives float64, vanishes in float32
A = np.array([[4.0 + EPS, 1.0], [1.0, 3.0]])  # SPD
B = np.array([1.0, 2.0])

# name -> call with (A, b); every one takes a NumPy matrix
CALLS = {
    "solve": lambda a, b: mtl5.solve(a, b),
    "inv": lambda a, b: mtl5.inv(a),
    "lu": lambda a, b: mtl5.lu(a),
    "cholesky": lambda a, b: mtl5.cholesky(a),
    "matmul": lambda a, b: mtl5.matmul(a, a),
    "det": lambda a, b: mtl5.det(a),
}


def _precision(result):
    """'f64' or 'f32' for what each entry point returns."""
    if isinstance(result, np.ndarray):
        return {np.float64: "f64", np.float32: "f32"}[result.dtype.type]
    if isinstance(result, float):
        return None  # det returns a Python float either way; checked by value
    return type(result).__name__.rsplit("_", 1)[1]


class TestIntegerInputIsRefused:
    @pytest.mark.parametrize("name", list(CALLS))
    @pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint8, np.bool_, np.float16])
    def test_refused_with_the_cast_to_make(self, name, dtype):
        msg = rf"^{name}: NumPy arrays must be float32 or float64"
        with pytest.raises(TypeError, match=msg) as exc:
            CALLS[name](A.astype(dtype), B.astype(dtype))
        assert "astype(np.float64)" in str(exc.value)

    def test_the_issue_example(self):
        """#152's report: int64 A and b gave a float32 x, 1.1e-8 off."""
        A_int = np.array([[4, 1], [1, 3]], dtype=np.int64)
        with pytest.raises(TypeError, match="got int64"):
            mtl5.solve(A_int, np.array([1, 2], dtype=np.int64))
        x = mtl5.solve(A_int.astype(np.float64), np.array([1.0, 2.0]))
        assert x.dtype == np.float64
        np.testing.assert_allclose(x, [1 / 11, 7 / 11], rtol=1e-15)

    def test_integer_b_alone_is_refused(self):
        with pytest.raises(TypeError, match="got int64"):
            mtl5.solve(A, B.astype(np.int64))

    def test_the_compiled_overloads_do_not_convert(self):
        """noconvert in C++: a direct _core call fails instead of narrowing."""
        with pytest.raises(TypeError):
            mtl5._core.solve(A.astype(np.int64), B.astype(np.int64))
        with pytest.raises(TypeError):
            mtl5._core.inv(np.asfortranarray(A))

    def test_a_universal_dtypes_array_is_pointed_at_mtl5_matrix(self):
        ud = pytest.importorskip("universal_dtypes")
        with pytest.raises(TypeError, match=r"mtl5\.matrix\(a\)"):
            mtl5.solve(A.astype(ud.posit16), B.astype(ud.posit16))


class TestLayoutKeepsThePrecision:
    @pytest.mark.parametrize("name", list(CALLS))
    @pytest.mark.parametrize(
        "layout",
        [np.asfortranarray, lambda a: np.repeat(a, 2, axis=1)[:, ::2]],
        ids=["fortran", "strided"],
    )
    def test_float64_stays_float64(self, name, layout):
        a = layout(A)
        assert not a.flags.c_contiguous
        got = CALLS[name](a, B)
        if name == "det":
            assert got == pytest.approx(np.linalg.det(A), rel=1e-14)
            assert got - 11.0 > 1e-9  # det = 11 + 3 EPS; float32 rounds it to 11
        else:
            assert _precision(got) == "f64"

    def test_a_strided_b_stays_float64(self):
        b = np.repeat(B, 2)[::2]
        x = mtl5.solve(A, b)
        assert x.dtype == np.float64
        np.testing.assert_allclose(x, np.linalg.solve(A, B), rtol=1e-15)

    def test_the_result_matches_numpy_in_float64(self):
        np.testing.assert_allclose(
            mtl5.solve(np.asfortranarray(A), B), np.linalg.solve(A, B), rtol=1e-15
        )
        np.testing.assert_allclose(
            mtl5.inv(np.asfortranarray(A)).to_numpy(), np.linalg.inv(A), rtol=1e-14
        )

    @pytest.mark.parametrize("name", list(CALLS))
    def test_float32_stays_float32(self, name):
        got = CALLS[name](np.asfortranarray(A.astype(np.float32)), B.astype(np.float32))
        if name != "det":
            assert _precision(got) == "f32"

    def test_a_float32_float64_pair_goes_to_float64(self):
        for a, b in ((A.astype(np.float32), B), (A, B.astype(np.float32))):
            assert mtl5.solve(a, b).dtype == np.float64
        assert mtl5.matmul(A.astype(np.float32), A).dtype == "f64"


class TestNormAndDot:
    """Strided float64 vectors lost the same precision in norm and dot."""

    x = np.repeat(np.array([1.0 + EPS, 1.0]), 2)[::2]

    def test_norm_of_a_strided_float64_vector(self):
        assert not self.x.flags.c_contiguous
        assert mtl5.norm(self.x, 1) == 2.0 + EPS
        assert mtl5.norm(self.x, -1) == 1.0 + EPS

    def test_dot_of_strided_float64_vectors(self):
        assert mtl5.dot(self.x, self.x) == pytest.approx((1 + EPS) ** 2 + 1, rel=1e-15)
        assert mtl5.dot(self.x, self.x) != float(np.float32((1 + EPS) ** 2 + 1))

    def test_integer_dot_is_unchanged(self):
        """The widening integer path is separate and stays as it was."""
        a = np.array([100, 100], dtype=np.int8)
        assert mtl5.dot(a, a) == 20000


class TestOtherInputIsUnchanged:
    def test_mtl5_containers_pass_through(self):
        M = mtl5.matrix(A)
        assert type(mtl5.inv(M)).__name__ == "DenseMatrix_f64"
        x = mtl5.solve(M, mtl5.vector(B))
        np.testing.assert_allclose(x.to_numpy(), np.linalg.solve(A, B), rtol=1e-15)

    def test_keyword_arguments(self):
        np.testing.assert_allclose(mtl5.solve(A=A, b=B), np.linalg.solve(A, B), rtol=1e-15)
        with pytest.raises(TypeError, match="got int64"):
            mtl5.solve(A=A.astype(np.int64), b=B)

    def test_a_list_is_still_refused(self):
        with pytest.raises(TypeError):
            mtl5.solve(A.tolist(), B.tolist())

    def test_the_docstring_is_the_compiled_one(self):
        assert mtl5.solve.__name__ == "solve"
        assert "ndarray[dtype=float64" in mtl5.solve.__doc__
