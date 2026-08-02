// mtl5-python -- the rest of the dense surface: eigen/SVD, BLAS levels 2 and 3,
// and the property predicates (MTL5 epics #202, #229, #244).
//
// The eigen and SVD entry points mirror numpy.linalg deliberately -- eigvals /
// eig / eigvalsh / eigh with the same return shapes -- because
// that is the vocabulary a caller already has. They return NumPy arrays rather
// than MTL5 containers: these are analysis results you read, not operands you
// chain, and the general eigenproblem is complex, which the MTL5 container
// bindings do not cover.
//
// Element types are float32/float64 here. The eigensolvers are available for
// custom number types upstream (the in-house QR path serves them), but the
// tolerance defaults and convergence behaviour want their own validation pass
// before being exposed for posit and friends.
//
// SVD IS NOT BOUND. mtl::svd / detail::singular_values returns all-NaN
// singular values for roughly 30% of ordinary symmetric matrices, and is off by
// up to 143% on sigma_max for many of the rest -- verified against the
// sigma_max == spectral_radius identity, which needs no reference
// implementation. Filed as stillwater-sc/mtl5#337. That also takes out
// condition_number, rcond, numerical_rank and nullity, which are all computed
// from it. The eigensolvers on the same matrices are accurate (worst relative
// error 3.7e-15 symmetric, 1.1e-9 general over 120 matrices), so eigvals / eig
// / eigvalsh / eigh do ship.

#include "mtl5_types.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

// svd.hpp forms products with operator*, which lives here rather than being
// pulled in by svd.hpp itself.
#include <mtl/operation/operators.hpp>
#include <mtl/mat/operators.hpp>

#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/matrix_properties.hpp>
#include <mtl/operation/vector_properties.hpp>
#include <mtl/operation/factorization_properties.hpp>
#include <mtl/operation/spectral_properties.hpp>
#include <mtl/operation/ger.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/operation/trsv.hpp>
#include <mtl/operation/trmm.hpp>
#include <mtl/operation/trsm.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/operation/syr2k.hpp>

#include <complex>
#include <optional>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

// ---------------------------------------------------------------------------
// NumPy handoff helpers. Each allocates, fills, and hands ownership to a
// capsule; the GIL is held throughout, so no nogil regions here.
// ---------------------------------------------------------------------------
nb::ndarray<nb::numpy, double, nb::ndim<1>> to_numpy_1d(const std::vector<double>& v) {
    double* buf = new double[v.size()];
    for (std::size_t i = 0; i < v.size(); ++i) buf[i] = v[i];
    std::size_t shape[1] = { v.size() };
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
}

nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>
to_numpy_1d(const std::vector<std::complex<double>>& v) {
    auto* buf = new std::complex<double>[v.size()];
    for (std::size_t i = 0; i < v.size(); ++i) buf[i] = v[i];
    std::size_t shape[1] = { v.size() };
    nb::capsule owner(buf, [](void* p) noexcept {
        delete[] static_cast<std::complex<double>*>(p);
    });
    return nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>(buf, 1, shape, owner);
}

template <typename Scalar, typename Fill>
nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>
make_numpy_2d(std::size_t rows, std::size_t cols, Fill&& fill) {
    Scalar* buf = new Scalar[rows * cols];
    fill(buf);
    std::size_t shape[2] = { rows, cols };
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<Scalar*>(p); });
    return nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(buf, 2, shape, owner);
}

// ===========================================================================
// Eigenvalues and SVD
// ===========================================================================
template <typename T>
void register_eigen(nb::module_& m) {
    using MV = MatrixView<T>;

    auto square = [](const MV& A, const char* who) {
        if (A.mat.num_rows() != A.mat.num_cols())
            throw std::invalid_argument(std::string(who) + ": matrix must be square");
    };

    m.def("eigvals", [square](const MV& A_mv, double tol) {
        square(A_mv, "eigvals");
        const std::size_t n = A_mv.mat.num_rows();
        std::vector<std::complex<double>> out(n);
        {
            nogil guard;
            auto eigs = mtl::eigenvalue(A_mv.mat, static_cast<T>(tol));
            for (std::size_t i = 0; i < n; ++i)
                out[i] = std::complex<double>(static_cast<double>(eigs[i].real()),
                                              static_cast<double>(eigs[i].imag()));
        }
        return to_numpy_1d(out);
    }, "A"_a, "tol"_a = 1e-10,
       "Eigenvalues of a general square matrix, as a complex array");

    m.def("eig", [square](const MV& A_mv, double tol) {
        square(A_mv, "eig");
        const std::size_t n = A_mv.mat.num_rows();
        std::vector<std::complex<double>> w(n);
        std::vector<std::complex<double>> V(n * n);
        {
            nogil guard;
            auto res = mtl::eigen(A_mv.mat, static_cast<T>(tol));
            for (std::size_t i = 0; i < n; ++i)
                w[i] = std::complex<double>(static_cast<double>(res.eigenvalues[i].real()),
                                            static_cast<double>(res.eigenvalues[i].imag()));
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t c = 0; c < n; ++c)
                    V[r * n + c] = std::complex<double>(
                        static_cast<double>(res.eigenvectors(r, c).real()),
                        static_cast<double>(res.eigenvectors(r, c).imag()));
        }
        auto Vn = make_numpy_2d<std::complex<double>>(n, n, [&](std::complex<double>* buf) {
            for (std::size_t i = 0; i < n * n; ++i) buf[i] = V[i];
        });
        return std::make_pair(to_numpy_1d(w), Vn);
    }, "A"_a, "tol"_a = 1e-10,
       "Eigenvalues and right eigenvectors of a general square matrix, as "
       "(w, V) complex arrays with V's columns the eigenvectors");

    m.def("eigvalsh", [square](const MV& A_mv, double tol) {
        square(A_mv, "eigvalsh");
        const std::size_t n = A_mv.mat.num_rows();
        std::vector<double> out(n);
        {
            nogil guard;
            auto eigs = mtl::eigenvalue_symmetric(A_mv.mat, static_cast<T>(tol));
            for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(eigs[i]);
        }
        return to_numpy_1d(out);
    }, "A"_a, "tol"_a = 1e-10,
       "Eigenvalues of a symmetric matrix, real and ascending");

    m.def("eigh", [square](const MV& A_mv, double tol) {
        square(A_mv, "eigh");
        const std::size_t n = A_mv.mat.num_rows();
        std::vector<double> w(n), V(n * n);
        {
            nogil guard;
            auto res = mtl::eigen_symmetric(A_mv.mat, static_cast<T>(tol));
            for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<double>(res.eigenvalues[i]);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t c = 0; c < n; ++c)
                    V[r * n + c] = static_cast<double>(res.eigenvectors(r, c));
        }
        auto Vn = make_numpy_2d<double>(n, n, [&](double* buf) {
            for (std::size_t i = 0; i < n * n; ++i) buf[i] = V[i];
        });
        return std::make_pair(to_numpy_1d(w), Vn);
    }, "A"_a, "tol"_a = 1e-10,
       "Eigenvalues and eigenvectors of a symmetric matrix, as (w, Q) with "
       "A = Q diag(w) Q^T");

    // svd() and svdvals() are NOT bound — see the note at the head of this file.
}

// ===========================================================================
// BLAS levels 2 and 3
//
// Every one writes into a caller-supplied output, matching the BLAS contract
// (and MTL5's signatures) rather than allocating. That is the point of these:
// accumulating into an existing C is what makes them useful.
// ===========================================================================
template <typename T>
void register_blas(nb::module_& m) {
    using MV = MatrixView<T>;
    using VV = VectorView<T>;

    m.def("ger", [](double alpha, const VV& x, const VV& y, MV& A) {
        if (A.mat.num_rows() != x.vec.size() || A.mat.num_cols() != y.vec.size())
            throw std::invalid_argument("ger: A must be len(x) x len(y)");
        nogil guard;
        mtl::ger(static_cast<T>(alpha), x.vec, y.vec, A.mat);
    }, "alpha"_a, "x"_a, "y"_a, "A"_a,
       "Rank-1 update in place: A += alpha * x y^T");

    m.def("symv", [](double alpha, const MV& A, const VV& x, double beta, VV& y) {
        if (A.mat.num_rows() != A.mat.num_cols())
            throw std::invalid_argument("symv: A must be square");
        if (A.mat.num_cols() != x.vec.size() || A.mat.num_rows() != y.vec.size())
            throw std::invalid_argument("symv: dimension mismatch");
        nogil guard;
        mtl::symv(static_cast<T>(alpha), A.mat, x.vec, static_cast<T>(beta), y.vec);
    }, "alpha"_a, "A"_a, "x"_a, "beta"_a, "y"_a,
       "Symmetric matrix-vector product in place: y = alpha*A*x + beta*y");

    m.def("trmv", [](const MV& A, VV& x, bool upper, bool unit_diag) {
        if (A.mat.num_rows() != A.mat.num_cols() || A.mat.num_cols() != x.vec.size())
            throw std::invalid_argument("trmv: A must be square and match len(x)");
        nogil guard;
        mtl::trmv(A.mat, x.vec, upper, unit_diag);
    }, "A"_a, "x"_a, "upper"_a = true, "unit_diag"_a = false,
       "Triangular matrix-vector product in place: x = A*x");

    m.def("trsv", [](const MV& A, VV& x, bool upper, bool unit_diag) {
        if (A.mat.num_rows() != A.mat.num_cols() || A.mat.num_cols() != x.vec.size())
            throw std::invalid_argument("trsv: A must be square and match len(x)");
        nogil guard;
        mtl::trsv(A.mat, x.vec, upper, unit_diag);
    }, "A"_a, "x"_a, "upper"_a = true, "unit_diag"_a = false,
       "Triangular solve in place: x = A^-1 x");

    m.def("trmm", [](double alpha, const MV& A, MV& B, bool upper, bool unit_diag) {
        if (A.mat.num_rows() != A.mat.num_cols() || A.mat.num_cols() != B.mat.num_rows())
            throw std::invalid_argument("trmm: A must be square with num_cols == B.num_rows");
        nogil guard;
        mtl::trmm(static_cast<T>(alpha), A.mat, B.mat, upper, unit_diag);
    }, "alpha"_a, "A"_a, "B"_a, "upper"_a = true, "unit_diag"_a = false,
       "Triangular matrix-matrix product in place: B = alpha*A*B");

    m.def("trsm", [](double alpha, const MV& A, MV& B, bool upper, bool unit_diag) {
        if (A.mat.num_rows() != A.mat.num_cols() || A.mat.num_cols() != B.mat.num_rows())
            throw std::invalid_argument("trsm: A must be square with num_cols == B.num_rows");
        nogil guard;
        mtl::trsm(static_cast<T>(alpha), A.mat, B.mat, upper, unit_diag);
    }, "alpha"_a, "A"_a, "B"_a, "upper"_a = true, "unit_diag"_a = false,
       "Triangular solve with multiple RHS in place: solve A X = alpha*B");

    m.def("symm", [](double alpha, const MV& A, const MV& B, double beta, MV& C) {
        if (A.mat.num_rows() != A.mat.num_cols())
            throw std::invalid_argument("symm: A must be square");
        if (A.mat.num_cols() != B.mat.num_rows() ||
            C.mat.num_rows() != A.mat.num_rows() || C.mat.num_cols() != B.mat.num_cols())
            throw std::invalid_argument("symm: dimension mismatch");
        nogil guard;
        mtl::symm(static_cast<T>(alpha), A.mat, B.mat, static_cast<T>(beta), C.mat);
    }, "alpha"_a, "A"_a, "B"_a, "beta"_a, "C"_a,
       "Symmetric matrix-matrix product in place: C = alpha*A*B + beta*C");

    m.def("syrk", [](double alpha, const MV& A, double beta, MV& C) {
        if (C.mat.num_rows() != A.mat.num_rows() || C.mat.num_cols() != A.mat.num_rows())
            throw std::invalid_argument("syrk: C must be A.num_rows x A.num_rows");
        nogil guard;
        mtl::syrk(static_cast<T>(alpha), A.mat, static_cast<T>(beta), C.mat);
    }, "alpha"_a, "A"_a, "beta"_a, "C"_a,
       "Symmetric rank-k update in place: C = alpha*A*A^T + beta*C "
       "(both triangles written)");

    m.def("syr2k", [](double alpha, const MV& A, const MV& B, double beta, MV& C) {
        if (A.mat.num_rows() != B.mat.num_rows() || A.mat.num_cols() != B.mat.num_cols())
            throw std::invalid_argument("syr2k: A and B must have the same shape");
        if (C.mat.num_rows() != A.mat.num_rows() || C.mat.num_cols() != A.mat.num_rows())
            throw std::invalid_argument("syr2k: C must be A.num_rows x A.num_rows");
        nogil guard;
        mtl::syr2k(static_cast<T>(alpha), A.mat, B.mat, static_cast<T>(beta), C.mat);
    }, "alpha"_a, "A"_a, "B"_a, "beta"_a, "C"_a,
       "Symmetric rank-2k update in place: C = alpha*(A*B^T + B*A^T) + beta*C");
}

// ===========================================================================
// Property predicates
//
// Cheap structural checks and the factorization/spectral-backed ones, which
// are not cheap -- the spectral ones run an eigensolve. The docstrings say
// which is which so a caller does not put an O(n^3) query inside a loop by
// accident.
// ===========================================================================
template <typename T>
void register_predicates(nb::module_& m) {
    using MV = MatrixView<T>;
    using VV = VectorView<T>;

    // -- structural, O(n^2) or less ------------------------------------------
    m.def("is_square", [](const MV& A) { return mtl::is_square(A.mat); }, "A"_a);
    m.def("is_empty",  [](const MV& A) { return mtl::is_empty(A.mat); }, "A"_a);
    m.def("is_symmetric", [](const MV& A, double tol) {
        nogil guard; return mtl::is_symmetric(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);
    m.def("is_hermitian", [](const MV& A, double tol) {
        nogil guard; return mtl::is_hermitian(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);
    m.def("is_upper_triangular", [](const MV& A, double tol) {
        nogil guard; return mtl::is_upper_triangular(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);
    m.def("is_lower_triangular", [](const MV& A, double tol) {
        nogil guard; return mtl::is_lower_triangular(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);
    m.def("is_triangular", [](const MV& A, double tol) {
        nogil guard; return mtl::is_triangular(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);
    m.def("is_diagonal", [](const MV& A, double tol) {
        nogil guard; return mtl::is_diagonal(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);
    m.def("is_banded", [](const MV& A, std::size_t kl, std::size_t ku, double tol) {
        nogil guard; return mtl::is_banded(A.mat, kl, ku, static_cast<T>(tol));
    }, "A"_a, "kl"_a, "ku"_a, "tol"_a = 0.0);
    m.def("is_diagonally_dominant", [](const MV& A, bool strict) {
        nogil guard; return mtl::is_diagonally_dominant(A.mat, strict);
    }, "A"_a, "strict"_a = false);

    // -- orthogonality, O(n^3): each forms a product ------------------------
    m.def("is_orthogonal", [](const MV& A, double tol) {
        nogil guard; return mtl::is_orthogonal(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = -1.0, "A^T A == I. O(n^3): forms the product.");
    m.def("is_unitary", [](const MV& A, double tol) {
        nogil guard; return mtl::is_unitary(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = -1.0, "A^H A == I. O(n^3).");
    m.def("is_normal", [](const MV& A, double tol) {
        nogil guard; return mtl::is_normal(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = -1.0, "A A^H == A^H A. O(n^3).");

    // -- factorization-backed, O(n^3) ---------------------------------------
    m.def("is_spd", [](const MV& A, double sym_tol) {
        nogil guard; return mtl::is_spd(A.mat, static_cast<T>(sym_tol));
    }, "A"_a, "sym_tol"_a = 0.0,
       "Symmetric positive definite. O(n^3): attempts a Cholesky on a copy.");
    m.def("is_positive_definite", [](const MV& A, double sym_tol) {
        nogil guard; return mtl::is_positive_definite(A.mat, static_cast<T>(sym_tol));
    }, "A"_a, "sym_tol"_a = 0.0);
    m.def("is_singular", [](const MV& A, double tol) {
        nogil guard; return mtl::is_singular(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0, "O(n^3): runs an LU on a copy.");
    m.def("is_nonsingular", [](const MV& A, double tol) {
        nogil guard; return mtl::is_nonsingular(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);
    m.def("is_invertible", [](const MV& A, double tol) {
        nogil guard; return mtl::is_invertible(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = 0.0);

    // -- spectral, O(n^3) via an eigensolve ---------------------------------
    //
    // condition_number, rcond, numerical_rank and nullity are absent: all four
    // are computed from detail::singular_values, and MTL5's SVD returns NaN for
    // ~30% of symmetric inputs (stillwater-sc/mtl5#337). spectral_radius,
    // inertia and is_indefinite go through the eigensolvers instead, which
    // check out, so they are bound.
    m.def("spectral_radius", [](const MV& A) {
        nogil guard; return static_cast<double>(mtl::spectral_radius(A.mat));
    }, "A"_a, "Largest |eigenvalue|. O(n^3).");
    m.def("is_indefinite", [](const MV& A, double tol) {
        nogil guard; return mtl::is_indefinite(A.mat, static_cast<T>(tol));
    }, "A"_a, "tol"_a = -1.0);
    m.def("inertia", [](const MV& A, double tol) {
        mtl::inertia_t in;
        {
            nogil guard;
            in = mtl::inertia(A.mat, static_cast<T>(tol));
        }
        nb::dict d;
        d["positive"] = in.positive;
        d["negative"] = in.negative;
        d["zero"]     = in.zero;
        return d;
    }, "A"_a, "tol"_a = -1.0,
       "Sylvester inertia as {positive, negative, zero} eigenvalue counts. "
       "O(n^3): runs the symmetric eigensolver.");

    // -- vector predicates ---------------------------------------------------
    m.def("is_zero", [](const VV& v, double tol) {
        nogil guard; return mtl::is_zero(v.vec, static_cast<T>(tol));
    }, "x"_a, "tol"_a = 0.0);
    m.def("is_finite", [](const VV& v) {
        nogil guard; return mtl::is_finite(v.vec);
    }, "x"_a);
    m.def("has_nan", [](const VV& v) {
        nogil guard; return mtl::has_nan(v.vec);
    }, "x"_a);
    m.def("has_inf", [](const VV& v) {
        nogil guard; return mtl::has_inf(v.vec);
    }, "x"_a);
    // These three take a RELATIVE tolerance defaulting to 128*eps upstream, not
    // the -1 "auto" sentinel the matrix-side predicates use. Passing -1 here
    // would make every comparison `<= -1` and so always false. Omitting the
    // argument defers to MTL5's own default.
    m.def("is_normalized", [](const VV& v, std::optional<double> tol) {
        nogil guard;
        return tol ? mtl::is_normalized(v.vec, static_cast<T>(*tol))
                   : mtl::is_normalized(v.vec);
    }, "x"_a, "tol"_a = nb::none(),
       "Unit two-norm to within tol (default: 128 * epsilon, relative)");
    m.def("is_unit", [](const VV& v, std::optional<double> tol) {
        nogil guard;
        return tol ? mtl::is_unit(v.vec, static_cast<T>(*tol)) : mtl::is_unit(v.vec);
    }, "x"_a, "tol"_a = nb::none());
    m.def("is_orthogonal_to", [](const VV& u, const VV& v, std::optional<double> tol) {
        if (u.vec.size() != v.vec.size())
            throw std::invalid_argument("is_orthogonal_to: vectors must have the same length");
        nogil guard;
        return tol ? mtl::is_orthogonal_to(u.vec, v.vec, static_cast<T>(*tol))
                   : mtl::is_orthogonal_to(u.vec, v.vec);
    }, "u"_a, "v"_a, "tol"_a = nb::none());
}

}  // namespace

// ===========================================================================
void register_dense_ops(nb::module_& m) {
    // float64 first so it wins overload resolution for a plain Python float
    // argument; the float32 overloads then bind the f32 containers.
    register_eigen<double>(m);
    register_eigen<float>(m);
    register_blas<double>(m);
    register_blas<float>(m);
    register_predicates<double>(m);
    register_predicates<float>(m);
}
