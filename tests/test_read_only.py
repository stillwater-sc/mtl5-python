"""Read-only NumPy input (#116).

pandas 3's copy-on-write makes ``Series.to_numpy()`` and ``DataFrame.to_numpy()``
read-only views, and ``np.broadcast_to`` or ``a.flags.writeable = False`` give
the same thing. Every entry point that only reads or copies its input has to
take one. The zero-copy views (``vector``, ``matrix``, ``array.asarray``) alias
the array and can write through it, so they still refuse, but they have to say
why rather than dump nanobind's overload list.
"""

from __future__ import annotations

import numpy as np
import pytest

import mtl5
import mtl5.sparse  # noqa: F401  (a submodule; import mtl5 does not load it)


def ro(a):
    a = np.array(a)
    a.flags.writeable = False
    return a


A = ro([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])  # SPD
b = ro([1.0, 2.0, 3.0])
X = ro(np.arange(1.0, 7.0).reshape(2, 3))


def _sparse():
    import mtl5.sparse as msp

    return msp.csr_matrix(
        ro([4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0]),
        ro([0, 1, 0, 1, 2, 1, 2]),
        ro([0, 2, 5, 7]),
        (3, 3),
    )


# Each case calls one entry point with read-only input only. A case that needs
# another dtype wraps its astype() in ro() again, because astype() always copies.
# The expected value is the same call on a writable copy, so what is tested is
# that read-only input is accepted and changes nothing, not the arithmetic.
CASES = {
    "vector_copy": lambda a: mtl5.vector_copy(a).to_numpy(),
    "matrix_copy": lambda A: mtl5.matrix_copy(A).to_numpy(),
    "vector_copy_f32": lambda a: mtl5.vector_copy(ro(a.astype(np.float32))).to_numpy(),
    "norm": lambda a: mtl5.norm(a),
    "norm_ord1": lambda a: mtl5.norm(a, 1),
    "dot": lambda a: mtl5.dot(a, a),
    "solve": lambda A, b: mtl5.solve(A, b),
    "matmul": lambda A: mtl5.matmul(A, A).to_numpy(),
    "det": lambda A: mtl5.det(A),
    "inv": lambda A: mtl5.inv(A).to_numpy(),
    "lu_solve": lambda A, b: mtl5.lu(A).solve(b).to_numpy(),
    "cholesky_solve": lambda A, b: mtl5.cholesky(A).solve(b).to_numpy(),
    # qr/ldlt run A through a Python helper; their solve() takes an MTL5 vector.
    "qr_solve": lambda A, b: mtl5.qr(A).solve(mtl5.vector_copy(b)).to_numpy(),
    "ldlt_solve": lambda A, b: mtl5.ldlt(A).solve(mtl5.vector_copy(b)).to_numpy(),
    "convert_vec": lambda a: mtl5.convert(a, "posit16").to_numpy(),
    "convert_mat": lambda A: mtl5.convert(A, "posit16").to_numpy(),
    "vector_posit16": lambda a: mtl5.vector_posit16(a).to_numpy(),
    "mixed_dot": lambda a: mtl5.mixed.dot(a, a),
    "mixed_matmul_i8": lambda A: mtl5.mixed.matmul(
        ro(A.astype(np.int8)), ro(A.astype(np.int8)), accumulator="i32"
    ).to_numpy(),
    "mixed_lu_iterative_refine": lambda A, b: (
        mtl5.mixed.lu_iterative_refine(A, b, working="f32")[0]
    ),
    "mixed_backward_error": lambda A, b: mtl5.mixed.backward_error(A, b, b),
    "complex_norm": lambda a: mtl5.norm(ro(a.astype(np.complex128))),
    "complex_dot": lambda a: mtl5.dot(ro(a.astype(np.complex128)), ro(a.astype(np.complex128))),
    "complex_vector_copy": lambda a: mtl5.vector_copy(ro(a.astype(np.complex128))).to_numpy(),
    "tensor": lambda A: mtl5.tensor.asarray(A).to_numpy(),
    "sparse_csr_matvec": lambda b: _sparse().matvec(b).to_numpy(),
    "sparse_coo": lambda b: mtl5.sparse.coo_matrix(ro([0, 1, 2]), ro([0, 1, 2]), b, (3, 3))
    .matvec(b)
    .to_numpy(),
    "sparse_cg": lambda b: mtl5.sparse.cg(_sparse(), b)[0],
    "sparse_splu": lambda b: mtl5.sparse.splu(_sparse()).solve(b).to_numpy(),
}

ARGS = {"a": b, "b": b, "A": A}


def _call(name, make_writable=False):
    f = CASES[name]
    names = f.__code__.co_varnames[: f.__code__.co_argcount]
    args = [ARGS[n] for n in names]
    if make_writable:
        args = [np.array(x) for x in args]
    return f(*args)


@pytest.mark.parametrize("name", sorted(CASES))
def test_read_only_input_is_accepted(name):
    if name.startswith("sparse_"):
        pytest.importorskip("scipy")
    got = _call(name)
    want = _call(name, make_writable=True)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_inputs_were_really_read_only():
    """Guard the sweep: if ro() stopped working every case would pass trivially."""
    assert not A.flags.writeable and not b.flags.writeable and not X.flags.writeable


class TestViewsRefuseWithAReason:
    @pytest.mark.parametrize(
        "factory, arg, copy_fn",
        [
            (mtl5.vector, b, "vector_copy"),
            (mtl5.matrix, X, "matrix_copy"),
            (mtl5.vector, b.astype(np.float32), "vector_copy"),
            (mtl5.matrix, X.astype(np.int64), "matrix_copy"),
            (mtl5.vector, b.astype(np.complex128), "vector_copy"),
            (mtl5.matrix, X.astype(np.complex64), "matrix_copy"),
        ],
    )
    def test_vector_and_matrix(self, factory, arg, copy_fn):
        arg = ro(arg)
        with pytest.raises(TypeError, match=rf"read-only.*mtl5\.{copy_fn}\(a\)"):
            factory(arg)

    def test_asarray(self):
        with pytest.raises(TypeError, match=r"mtl5\.array\.asarray\(\) is a zero-copy view"):
            mtl5.array.asarray(X)

    def test_writable_input_still_views(self):
        """The fallback overload must not shadow the view for writable arrays."""
        x = np.array(b)
        assert np.shares_memory(mtl5.vector(x).to_numpy(), x)
        assert mtl5.vector(x).is_view

    def test_a_dtype_mismatch_is_not_blamed_on_read_only(self):
        """The fallback is noconvert, so an unbound dtype keeps nanobind's own error."""
        with pytest.raises(TypeError) as e:
            mtl5.vector(ro(np.array([1, 2], dtype=np.uint64)))
        assert "read-only" not in str(e.value)


def test_pandas3_dataframe_columns():
    """The case that found #116: a DataFrame column straight into mtl5."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": [3.0, 0.0], "y": [4.0, 1.0]})
    col = df["x"].to_numpy()
    assert mtl5.norm(df["y"].to_numpy()) == pytest.approx(np.hypot(4.0, 1.0))
    assert mtl5.dot(col, col) == pytest.approx(9.0)
    # On pandas 3 col is read-only (copy-on-write); on pandas 2 this proves less.
    np.testing.assert_array_equal(mtl5.convert(col, "posit16").to_numpy(), col)
