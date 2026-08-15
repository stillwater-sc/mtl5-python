"""MTL5 Python bindings — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    import mtl5._core as _core
except ImportError as _exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        f"mtl5: the compiled extension mtl5._core could not be imported ({_exc}). "
        "If you launched Python from the repository root, the source 'mtl5/' "
        "directory shadows the installed package but has no compiled extension — "
        "run from another directory. Otherwise build/install it: 'pip install .' "
        "(or 'pip install -e .' for a development checkout)."
    ) from _exc

# Single source of truth: pyproject.toml → installed package metadata.
# The C++ module also reads this at build time via CMake.
# Falls back to the C++ extension's version when running from a source tree
# without installed metadata (e.g. an in-place build without pip install).
try:
    __version__ = _pkg_version("mtl5")
except _PackageNotFoundError:
    try:
        from mtl5._core import __version__  # noqa: F811
    except ImportError:
        __version__ = "0.0.0-dev"

# Guard against a compiled extension that is older than this Python layer — the
# classic "updated the source, forgot to rebuild the C++ extension", most often
# with an editable `pip install -e .` where the .py files advance while _core
# stays at the last build.
#
# Every _core import below is wrapped, and the ImportError is reframed rather
# than checked for in advance. The previous form probed a fixed tuple of
# submodules ("tensor", "view", "mg", ...) and so only noticed staleness that
# happened to remove one of those seven. It missed the real case: #69-#73 added
# CLASSES (DenseMatrix_cfloat32 and friends), all seven submodules still
# existed, the guard passed, and a user got the bare
# "cannot import name 'DenseMatrix_cfloat32' from 'mtl5._core'" it exists to
# prevent. Extending the tuple would only re-arm that trap for the next symbol;
# catching the failure covers every name with no list to maintain.
#
# A version-string compare would NOT work here: with an editable install the
# metadata and _core versions both stay at the old build while only the source
# advances.
try:
    from mtl5 import tensor  # noqa: F401  -- submodule, imported for mtl5.tensor
    from mtl5._core import (
        # Cholesky factorization objects
        CholeskyFactor_c64,
        CholeskyFactor_c128,
        CholeskyFactor_f32,
        CholeskyFactor_f64,
        # Complex types (zero-copy views over complex64/complex128)
        DenseMatrix_c64,
        DenseMatrix_c128,
        # Extended number types (#69) — emulated binary32, takum, and the float
        # cascades. isort keeps this block alphabetical, so these land split across
        # the family comments rather than together; the comment above each name is
        # the one that applies to it.
        DenseMatrix_cfloat32,
        DenseMatrix_dd_cascade,
        # Native IEEE types (zero-copy views)
        DenseMatrix_f32,
        DenseMatrix_f64,
        # Universal fixpnt types
        DenseMatrix_fixpnt8,
        DenseMatrix_fixpnt16,
        # Universal cfloat types
        DenseMatrix_fp8,
        DenseMatrix_fp16,
        DenseMatrix_i32,
        DenseMatrix_i64,
        # Universal lns types
        DenseMatrix_lns16,
        DenseMatrix_lns32,
        # Universal posit types
        DenseMatrix_posit8,
        DenseMatrix_posit16,
        DenseMatrix_posit32,
        DenseMatrix_posit64,
        # Extended number types, continued
        DenseMatrix_qd_cascade,
        DenseMatrix_takum32,
        DenseMatrix_td_cascade,
        DenseVector_c64,
        DenseVector_c128,
        DenseVector_cfloat32,
        DenseVector_dd_cascade,
        DenseVector_f32,
        DenseVector_f64,
        DenseVector_fixpnt8,
        DenseVector_fixpnt16,
        DenseVector_fp8,
        DenseVector_fp16,
        DenseVector_i32,
        DenseVector_i64,
        DenseVector_lns16,
        DenseVector_lns32,
        DenseVector_posit8,
        DenseVector_posit16,
        DenseVector_posit32,
        DenseVector_posit64,
        DenseVector_qd_cascade,
        DenseVector_takum32,
        DenseVector_td_cascade,
        # QR / LQ factorization objects — complex only; the real ones are reached
        # through mtl5.qr()/mtl5.lq(), which dispatch on dtype.
        LQFactor_c64,
        LQFactor_c128,
        # LU factorization objects
        LUFactor_c64,
        LUFactor_c128,
        LUFactor_f32,
        LUFactor_f64,
        QRFactor_c64,
        QRFactor_c128,
        # Complex operations
        adjoint,
        # N-dimensional array layer (mtl/array)
        array,
        # Backend management
        backends,
        # Build introspection
        build_info,
        cholesky,
        condition_number,
        conj,
        # Mixed precision
        convert,
        # Device management
        devices,
        # Operations
        dot,
        dot_real,
        dtypes,
        eig,
        eigh,
        eigvals,
        eigvalsh,
        frobenius_norm,
        ger,
        get_backend,
        # Threading
        get_num_threads,
        has_inf,
        has_nan,
        inertia,
        inv,
        is_banded,
        is_diagonal,
        is_diagonally_dominant,
        is_empty,
        is_finite,
        is_hermitian,
        is_indefinite,
        is_invertible,
        is_lower_triangular,
        is_nonsingular,
        is_normal,
        is_normalized,
        is_orthogonal,
        is_orthogonal_to,
        is_positive_definite,
        is_singular,
        is_spd,
        is_square,
        is_symmetric,
        is_triangular,
        is_unit,
        is_unitary,
        is_upper_triangular,
        is_zero,
        ldlt_solve,
        lu,
        matmul,
        matrix,
        matrix_cfloat32,
        matrix_copy,
        matrix_dd_cascade,
        matrix_fixpnt8,
        matrix_fixpnt16,
        matrix_fp8,
        matrix_fp16,
        matrix_lns16,
        matrix_lns32,
        matrix_posit8,
        matrix_posit16,
        matrix_posit32,
        matrix_posit64,
        matrix_qd_cascade,
        matrix_takum32,
        matrix_td_cascade,
        matvec,
        # Smoothers, grid transfer and multigrid
        mg,
        # Mixed-precision submodule (element / accumulator / result precisions)
        mixed,
        norm,
        nullity,
        numerical_rank,
        rcond,
        set_backend,
        set_num_threads,
        solve,
        spectral_radius,
        svd,
        svdvals,
        symm,
        symv,
        syr2k,
        syrk,
        transpose,
        trmm,
        trmv,
        trsm,
        trsv,
        vector,
        vector_cfloat32,
        vector_copy,
        vector_dd_cascade,
        vector_fixpnt8,
        vector_fixpnt16,
        vector_fp8,
        vector_fp16,
        vector_lns16,
        vector_lns32,
        vector_posit8,
        vector_posit16,
        vector_posit32,
        vector_posit64,
        vector_qd_cascade,
        vector_takum32,
        vector_td_cascade,
        # Matrix views
        view,
    )
    from mtl5._core import arange as _arange
    from mtl5._core import det as _det
    from mtl5._core import generators as _generators
    from mtl5._core import geomspace as _geomspace
    from mtl5._core import linspace as _linspace
    from mtl5._core import logspace as _logspace
except ImportError as _exc:  # pragma: no cover - depends on a stale build
    # Only reframe a failure that is actually about _core. An ImportError from
    # somewhere else (numpy missing inside tensor.py, say) must surface as
    # itself rather than be mislabelled a stale build.
    if "_core" not in str(_exc):
        raise

    try:
        _pkg = _pkg_version("mtl5")
    except _PackageNotFoundError:
        _pkg = "<no installed metadata>"

    raise ImportError(
        "mtl5: the compiled extension (mtl5._core) is out of sync with the "
        f"Python package — {_exc}.\n"
        f"  Python package : {__file__}\n"
        f"  compiled _core : {getattr(_core, '__file__', '<unknown>')}\n"
        f"  versions       : package metadata {_pkg!r}, compiled _core "
        f"{getattr(_core, '__version__', '<unknown>')!r}\n"
        "This usually means the C++ extension was not rebuilt after the source "
        "was updated (common with an editable 'pip install -e .').\n"
        "Rebuild or reinstall:\n"
        "  pip install -e . --force-reinstall --no-deps   # rebuild a dev install\n"
        "  # or a clean wheel install, from outside the repository:\n"
        "  pip uninstall -y mtl5 && pip install mtl5\n"
        "Also avoid launching Python from the repository root, where the source "
        "'mtl5/' directory shadows the installed package."
    ) from _exc

_UNIVERSAL_DTYPES = (
    "fp8",
    "fp16",
    "posit8",
    "posit16",
    "posit32",
    "posit64",
    "fixpnt8",
    "fixpnt16",
    "lns16",
    "lns32",
    "cfloat32",
    "takum32",
    "dd_cascade",
    "td_cascade",
    "qd_cascade",
)


_NUMPY_TO_MTL5 = {
    "float32": "f32",
    "float64": "f64",
    "complex64": "c64",
    "complex128": "c128",
}

_COMPLEX_SUFFIXES = ("c64", "c128")


def _reject_complex(name: str, prefix: str, suffix: str, shown: str = "") -> None:
    """Raise for complex input to a factorization that has no complex form.

    QR and LQ do have one — MTL5 gained a complex Householder — so the test is
    whether the instantiation exists rather than a fixed list. Cholesky still
    does not compile for complex upstream, and `ldlt` is LDL^T rather than
    LDL^H; without this the caller would get nanobind's overload dump instead of
    somewhere to go.
    """
    if suffix not in _COMPLEX_SUFFIXES:
        return
    if hasattr(_core, f"{prefix}_{suffix}"):
        return
    raise TypeError(
        f"{name}: not available for {shown or suffix} — MTL5 has no complex {name}. "
        "Complex is supported for qr/lq/solve/lu/inv/matmul/matvec/norm/dot; "
        "use mtl5.solve() for a general complex system, or mtl5.ldlt_solve() "
        "for a complex symmetric one."
    )


def _as_mtl5_matrix(name: str, prefix: str, A):
    """Normalize A to (mtl5_matrix, dtype_suffix).

    Accepts an MTL5 matrix from mtl5.matrix()/mtl5.convert(), or a NumPy
    float32/float64 array, which is wrapped zero-copy — matching what
    mtl5.cholesky() has always accepted.

    `name` is the public function ("qr", "ldlt", ...) and is what the errors
    quote; `prefix` is the internal class prefix and never reaches the user.
    """
    import numpy as _np

    if isinstance(A, _np.ndarray):
        suffix = _NUMPY_TO_MTL5.get(A.dtype.name)
        if suffix is None:
            # Only suggest convert() for the factorizations that actually have
            # Universal instantiations, or the user earns a second TypeError.
            #
            # Since #73 every bound factorization has them, so the else branch
            # has no caller today. It stays for the next factorization added
            # ahead of its instantiations — this probe reads coverage off _core
            # rather than a hardcoded list, which is why it tracked qr and lu
            # gaining theirs in #69, then lq/cholesky/ldlt/bunch_kaufman in #73,
            # without an edit either time.
            hint = (
                " For another number system, convert first: mtl5.convert(a, 'posit32')."
                if _has_universal_support(prefix)
                else f" {name}() supports float32 and float64 only."
            )
            raise TypeError(
                f"{name}: NumPy arrays must be float32 or float64, got {A.dtype.name}.{hint}"
            )
        _reject_complex(name, prefix, suffix, A.dtype.name)
        return matrix(_np.ascontiguousarray(A)), suffix

    dt = getattr(A, "dtype", None)
    if not isinstance(dt, str):
        raise TypeError(
            f"{name}: expected an MTL5 matrix (from mtl5.matrix() or "
            f"mtl5.convert()) or a NumPy array, got {type(A).__name__}"
        )
    _reject_complex(name, prefix, dt)
    return A, dt


def _has_universal_support(prefix: str) -> bool:
    """Whether `prefix` is instantiated beyond float32/float64."""
    return hasattr(_core, f"{prefix}_posit32")


def _factor(name: str, prefix: str, A):
    """Construct the Factor class matching A's element type."""
    mat, dt = _as_mtl5_matrix(name, prefix, A)
    cls = getattr(_core, f"{prefix}_{dt}", None)
    if cls is None:
        available = "float32/float64"
        if _has_universal_support(prefix):
            available += " and the Universal dtypes"
        raise TypeError(f"{name} is not available for dtype '{dt}' — it supports {available}.")
    return cls(mat)


def qr(A):
    """Householder QR of a tall-or-square matrix (num_rows >= num_cols).

    Accepts float32/float64 (and complex), plus every Universal dtype via
    `mtl5.convert()`. Returns a factorization exposing `.solve(b)` (least
    squares), `.Q` and `.R`.

    Above 16 bits the Universal types factorize usefully. Below that, read the
    result before trusting it: for `fp8` and `fixpnt8` every Householder
    reflector rounds to zero, so `.Q` comes back as the exact identity and `.R`
    as the upper triangle of A. Such a factorization scores a perfect
    orthogonality check while reconstructing nothing but `triu(A)` — its error
    is exactly the weight of A's discarded subdiagonal, so how bad it looks
    depends on the input rather than on the format. It does not raise, because
    a genuinely upper-triangular input has zero reflectors too and that case is
    correct.
    """
    return _factor("qr", "QRFactor", A)


def lq(A):
    """Householder LQ: A = L Q, with L lower trapezoidal and Q orthogonal.

    Accepts any shape. For a wide or square A, Q is `num_cols x num_cols`; for
    a tall A the factorization is the economy form, with Q `num_cols x
    num_cols` and L `num_rows x num_cols`. Unlike `qr`, which needs
    num_rows >= num_cols, there is no shape restriction.

    Accepts float32/float64 (and complex), plus every Universal dtype via
    `mtl5.convert()`. Being the row-space counterpart of `qr`, it shares that
    machinery and its limits: `fp8` and `fixpnt8` degenerate to `Q == I` in the
    same way, and `fixpnt16` saturates here even though `qr` tolerates it at
    the same width. Read the result before trusting a narrow format.

    Returns a factorization exposing `.L` and `.Q`.
    """
    return _factor("lq", "LQFactor", A)


def bunch_kaufman(A):
    """Bunch-Kaufman LDL^T: symmetric indefinite with 1x1/2x2 block pivoting.

    The pivoting variant of `ldlt`, so it handles a symmetric matrix that plain
    `ldlt` rejects on a zero pivot. Available for float32/float64 and every
    Universal dtype via `mtl5.convert()`.

    Returns a factorization exposing `.solve(b)` and `.ipiv()`.
    """
    return _factor("bunch_kaufman", "BunchKaufmanFactor", A)


def ldlt(A):
    """LDL^T factorization of a symmetric matrix.

    Takes no square roots, so unlike `cholesky` it tolerates an indefinite
    matrix; `.diagonal()` returns D, whose signs give the inertia. It does not
    pivot, so a zero pivot raises.

    Available for float32/float64 and all ten Universal dtypes, which is what
    makes a Cholesky-versus-LDL^T comparison across number systems possible.
    The integer element types (i32/i64) are not supported.
    """
    return _factor("ldlt", "LDLTFactor", A)


# Convenience aliases — default to f64
DenseVector = DenseVector_f64
DenseMatrix = DenseMatrix_f64
LUFactor = LUFactor_f64
CholeskyFactor = CholeskyFactor_f64


# Re-export det (avoids shadowing the builtin namespace)
def det(A):
    """Compute the determinant of a matrix via LU factorization."""
    return _det(A)


_native_cholesky = cholesky


def cholesky(A):  # noqa: F811
    """Cholesky factorization A = L L^T of a symmetric positive definite matrix.

    Accepts float32/float64 matrices and NumPy arrays, and — since Phase 3 —
    the Universal dtypes as well, so it can be compared against `ldlt` in the
    same precision.
    """
    dt = getattr(A, "dtype", None)
    if isinstance(dt, str) and dt in _UNIVERSAL_DTYPES:
        return getattr(_core, f"CholeskyFactor_{dt}")(A)
    if isinstance(dt, str):
        _reject_complex("cholesky", "CholeskyFactor", dt)
    else:
        shown = getattr(dt, "name", "")
        _reject_complex("cholesky", "CholeskyFactor", _NUMPY_TO_MTL5.get(shown, ""), shown)
    return _native_cholesky(A)


# ---------------------------------------------------------------------------
# Generators and range vectors
#
# The C++ side produces float64. `dtype=` rounds to the requested element type,
# which is the right semantics for a test matrix: the definitions are over the
# reals, so you want the correctly rounded representation of the exact entry.
# ---------------------------------------------------------------------------
def _retype(obj, dtype: str):
    """Round a float64 generator result to `dtype`, or pass it through."""
    if dtype == "f64":
        return obj
    if not hasattr(obj, "to_numpy"):
        raise TypeError(
            "dtype= is not supported for the sparse generators; they are "
            "float64 CSR. Convert with mtl5.sparse.from_scipy(...astype(...))."
        )
    return convert(obj.to_numpy(), dtype)


def _make_generator(name):
    fn = getattr(_generators, name)

    def wrapper(*args, dtype: str = "f64", **kwargs):
        return _retype(fn(*args, **kwargs), dtype)

    wrapper.__name__ = name
    doc = fn.__doc__ or ""
    wrapper.__doc__ = (
        doc + "\n\nGenerated in float64; pass dtype= to round to another "
        "element type (see mtl5.dtypes())."
    )
    return wrapper


class _Generators:
    """Named test matrices with known pathologies.

    Every dense generator takes `dtype=` — `mtl5.generators.hilbert(8,
    dtype="posit16")` gives the correctly rounded posit16 Hilbert matrix, which
    is the usual starting point for a conditioning experiment.
    """

    _DENSE = (
        "clement frank pascal wilkinson rosser magic hilbert lehmer lotkin minij "
        "ones forsythe kahan moler companion vandermonde randorth randspd randsym "
        "randsvd"
    ).split()
    # Sparse generators still go through the wrapper so `dtype=` produces the
    # explanatory TypeError rather than nanobind's arity error.
    _SPARSE = ("laplacian_1d", "laplacian_2d", "poisson2d")
    # Pure data — no matrix, so no dtype.
    _PASSTHROUGH = ("testsuite_names", "testsuite_kappa")


for _n in [*_Generators._DENSE, *_Generators._SPARSE]:
    setattr(_Generators, _n, staticmethod(_make_generator(_n)))
for _n in _Generators._PASSTHROUGH:
    setattr(_Generators, _n, staticmethod(getattr(_generators, _n)))

generators = _Generators()


def _range_fn(fn, name):
    def wrapper(*args, dtype: str = "f64", **kwargs):
        return _retype(fn(*args, **kwargs), dtype)

    wrapper.__name__ = name
    wrapper.__doc__ = (fn.__doc__ or "") + (
        "\n\nGenerated in float64; pass dtype= for another element type."
    )
    return wrapper


arange = _range_fn(_arange, "arange")
linspace = _range_fn(_linspace, "linspace")
logspace = _range_fn(_logspace, "logspace")
geomspace = _range_fn(_geomspace, "geomspace")


# `io` is a Python module wrapping _core.io — it adds NumPy coercion and
# documents the two behaviours that differ from scipy.io. Importing it here
# binds mtl5.io to the wrapper rather than the bare extension submodule.
from mtl5 import io  # noqa: E402

# Optional pandas extension types — only loaded if pandas is installed
try:
    from mtl5.pandas_ext import HAS_PANDAS  # noqa: F401

    if HAS_PANDAS:
        from mtl5.pandas_ext import Posit16Array, Posit16Dtype  # noqa: F401
except ImportError:
    HAS_PANDAS = False


__all__ = [
    "__version__",
    # Typed vector classes — IEEE
    "DenseVector",
    # Complex
    "DenseVector_c64",
    "DenseVector_c128",
    "DenseVector_f32",
    "DenseVector_f64",
    "DenseVector_i32",
    "DenseVector_i64",
    # Typed vector classes — Universal cfloat
    "DenseVector_fp8",
    "DenseVector_fp16",
    # Typed vector classes — Universal posit
    "DenseVector_posit8",
    "DenseVector_posit16",
    "DenseVector_posit32",
    "DenseVector_posit64",
    # Typed vector classes — Universal fixpnt
    "DenseVector_fixpnt8",
    "DenseVector_fixpnt16",
    # Typed vector classes — Universal lns
    "DenseVector_lns16",
    "DenseVector_lns32",
    "DenseVector_cfloat32",
    "DenseVector_takum32",
    "DenseVector_dd_cascade",
    "DenseVector_td_cascade",
    "DenseVector_qd_cascade",
    # Typed matrix classes — IEEE
    "DenseMatrix",
    "DenseMatrix_c64",
    "DenseMatrix_c128",
    "DenseMatrix_f32",
    "DenseMatrix_f64",
    "DenseMatrix_i32",
    "DenseMatrix_i64",
    # Typed matrix classes — Universal cfloat
    "DenseMatrix_fp8",
    "DenseMatrix_fp16",
    # Typed matrix classes — Universal posit
    "DenseMatrix_posit8",
    "DenseMatrix_posit16",
    "DenseMatrix_posit32",
    "DenseMatrix_posit64",
    # Typed matrix classes — Universal fixpnt
    "DenseMatrix_fixpnt8",
    "DenseMatrix_fixpnt16",
    # Typed matrix classes — Universal lns
    "DenseMatrix_lns16",
    "DenseMatrix_lns32",
    "DenseMatrix_cfloat32",
    "DenseMatrix_takum32",
    "DenseMatrix_dd_cascade",
    "DenseMatrix_td_cascade",
    "DenseMatrix_qd_cascade",
    # Factorization classes
    "LQFactor_c64",
    "LQFactor_c128",
    "QRFactor_c64",
    "QRFactor_c128",
    "LUFactor",
    "LUFactor_c64",
    "LUFactor_c128",
    "LUFactor_f32",
    "LUFactor_f64",
    "CholeskyFactor",
    "CholeskyFactor_c64",
    "CholeskyFactor_c128",
    "CholeskyFactor_f32",
    "CholeskyFactor_f64",
    # Device & backend management
    "backends",
    "devices",
    "get_backend",
    "set_backend",
    # Mixed precision
    "array",
    "mg",
    "tensor",
    "view",
    "mixed",
    "convert",
    "dtypes",
    # Build introspection & threading
    "build_info",
    "get_num_threads",
    "set_num_threads",
    # Eigenvalues, SVD, BLAS L2/L3, and property predicates
    "condition_number",
    "nullity",
    "numerical_rank",
    "rcond",
    "svd",
    "svdvals",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "ger",
    "has_inf",
    "has_nan",
    "inertia",
    "is_banded",
    "is_diagonal",
    "is_diagonally_dominant",
    "is_empty",
    "is_finite",
    "is_hermitian",
    "is_indefinite",
    "is_invertible",
    "is_lower_triangular",
    "is_nonsingular",
    "is_normal",
    "is_normalized",
    "is_orthogonal",
    "is_orthogonal_to",
    "is_positive_definite",
    "is_singular",
    "is_spd",
    "is_square",
    "is_symmetric",
    "is_triangular",
    "is_unit",
    "is_unitary",
    "is_upper_triangular",
    "is_zero",
    "spectral_radius",
    "symm",
    "symv",
    "syr2k",
    "syrk",
    "trmm",
    "trmv",
    "trsm",
    "trsv",
    # Operations
    # I/O
    "io",
    # Generators and range vectors
    "generators",
    "arange",
    "geomspace",
    "linspace",
    "logspace",
    "bunch_kaufman",
    "cholesky",
    "ldlt",
    "ldlt_solve",
    "lq",
    "qr",
    "adjoint",
    "conj",
    "det",
    "dot",
    "dot_real",
    "frobenius_norm",
    "inv",
    "lu",
    "matmul",
    "matrix",
    "matrix_copy",
    "matvec",
    "norm",
    "solve",
    "transpose",
    "vector",
    "vector_copy",
    # Universal type factories — cfloat
    "matrix_fp8",
    "matrix_fp16",
    "vector_fp8",
    "vector_fp16",
    # Universal type factories — posit
    "matrix_posit8",
    "matrix_posit16",
    "matrix_posit32",
    "matrix_posit64",
    "vector_posit8",
    "vector_posit16",
    "vector_posit32",
    "vector_posit64",
    # Universal type factories — fixpnt
    "matrix_fixpnt8",
    "matrix_fixpnt16",
    "vector_fixpnt8",
    "vector_fixpnt16",
    # Universal type factories — lns
    "matrix_lns16",
    "matrix_lns32",
    "matrix_cfloat32",
    "matrix_takum32",
    "matrix_dd_cascade",
    "matrix_td_cascade",
    "matrix_qd_cascade",
    "vector_lns16",
    "vector_lns32",
    "vector_cfloat32",
    "vector_takum32",
    "vector_dd_cascade",
    "vector_td_cascade",
    "vector_qd_cascade",
]


# The pandas ExtensionDtype surface only exists when pandas is installed, so it
# joins __all__ here rather than being listed statically — a static entry would
# break `from mtl5 import *` on an install without pandas.
__all__ += ["HAS_PANDAS"]
if HAS_PANDAS:
    __all__ += ["Posit16Array", "Posit16Dtype"]
