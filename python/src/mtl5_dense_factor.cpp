// mtl5-python -- dense factorizations beyond LU and Cholesky.
//
// QR and LQ (Householder) and LDL^T. All follow the
// factorization-object shape the rest of the package uses: construct once,
// solve many times.
//
// Type coverage is deliberately uneven:
//
//   QR / LQ                   float32, float64
//   LDL^T / Cholesky          float32, float64, and every Universal type
//
// LDL^T and Cholesky reach into the Universal types because that is the
// comparison mtl5-python#18 asks for -- Cholesky versus LDL^T for a UKF
// covariance update across number systems, where the question is which one
// survives a matrix that drifts out of positive-definiteness in low precision.
//
// Bunch-Kaufman is deliberately absent; see the note further down.

#include "mtl5_types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <mtl/operation/qr.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/cholesky.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace nb::literals;

namespace {

/// Native types hand back the zero-copy view wrapper the rest of the package
/// returns; the Universal types are bound as bare containers, so they hand back
/// the container. Keeps `.solve()` consistent with `mtl5.solve` for each family.
template <typename T>
auto wrap_vector(mtl::vec::dense_vector<T>&& v) {
    if constexpr (std::is_arithmetic_v<T>)
        return VectorView<T>(std::move(v));
    else
        return std::move(v);
}

template <typename T>
auto wrap_matrix(mtl::mat::dense2D<T>&& m) {
    if constexpr (std::is_arithmetic_v<T>)
        return MatrixView<T>(std::move(m));
    else
        return std::move(m);
}

/// Copy a bound matrix (view wrapper or bare container) into an owning dense2D.
template <typename T, typename Src>
mtl::mat::dense2D<T> owned_copy(const Src& src) {
    const auto& A = [&]() -> const mtl::mat::dense2D<T>& {
        if constexpr (std::is_arithmetic_v<T>) return src.mat;
        else                                   return src;
    }();
    mtl::mat::dense2D<T> out(A.num_rows(), A.num_cols());
    for (std::size_t r = 0; r < A.num_rows(); ++r)
        for (std::size_t c = 0; c < A.num_cols(); ++c)
            out(r, c) = A(r, c);
    return out;
}

/// The Python-facing argument type for a matrix of element type T.
template <typename T>
using MatArg = std::conditional_t<std::is_arithmetic_v<T>, MatrixView<T>, mtl::mat::dense2D<T>>;

/// ...and for a vector.
template <typename T>
using VecArg = std::conditional_t<std::is_arithmetic_v<T>,
                                  VectorView<T>, mtl::vec::dense_vector<T>>;

template <typename T>
const mtl::vec::dense_vector<T>& as_vector(const VecArg<T>& v) {
    if constexpr (std::is_arithmetic_v<T>) return v.vec;
    else                                   return v;
}

// ===========================================================================
// QR — Householder, for least squares and orthogonalization
// ===========================================================================
template <typename T>
struct QRFactor {
    mtl::mat::dense2D<T> QR;                  // Householder vectors + R
    mtl::vec::dense_vector<T> tau;
    std::size_t nrows, ncols;
};

template <typename T>
void register_qr(nb::module_& m) {
    using Wrap = QRFactor<T>;
    const std::string name = std::string("QRFactor_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const MatArg<T>& A_in) {
            auto A = owned_copy<T>(A_in);
            const std::size_t m_ = A.num_rows(), n_ = A.num_cols();
            if (m_ < n_)
                throw std::invalid_argument(
                    "qr: needs num_rows >= num_cols (overdetermined or square)");
            mtl::vec::dense_vector<T> tau;
            {
                nogil guard;
                mtl::qr_factor(A, tau);
            }
            new (self) Wrap{std::move(A), std::move(tau), m_, n_};
        }, "A"_a, "Householder QR factorization A = Q R")
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.nrows, s.ncols);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("Q", [](const Wrap& s) {
            mtl::mat::dense2D<T> Q(s.nrows, s.nrows);
            {
                nogil guard;
                Q = mtl::qr_extract_Q(s.QR, s.tau);
            }
            return wrap_matrix<T>(std::move(Q));
        }, "The orthogonal factor Q, formed explicitly from the reflectors")
        .def_prop_ro("R", [](const Wrap& s) {
            mtl::mat::dense2D<T> R(s.nrows, s.ncols);
            {
                nogil guard;
                R = mtl::qr_extract_R(s.QR);
            }
            return wrap_matrix<T>(std::move(R));
        }, "The upper triangular factor R")
        .def("solve", [](const Wrap& s, const VecArg<T>& b_in) {
            const auto& b = as_vector<T>(b_in);
            if (b.size() != s.nrows)
                throw std::invalid_argument(
                    "qr.solve: RHS length " + std::to_string(b.size()) +
                    " does not match the row count " + std::to_string(s.nrows));
            mtl::vec::dense_vector<T> x(s.ncols);
            {
                nogil guard;
                mtl::qr_solve(s.QR, s.tau, x, b);
            }
            return wrap_vector<T>(std::move(x));
        }, "b"_a, "Least-squares solution of min ||A x - b||_2")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5." + name + "(shape=(" + std::to_string(s.nrows) + ", " +
                   std::to_string(s.ncols) + "))";
        });
}

// ===========================================================================
// LQ — the row-space counterpart of QR
// ===========================================================================
template <typename T>
struct LQFactor {
    mtl::mat::dense2D<T> LQ;
    mtl::vec::dense_vector<T> tau;
    std::size_t nrows, ncols;
};

template <typename T>
void register_lq(nb::module_& m) {
    using Wrap = LQFactor<T>;
    const std::string name = std::string("LQFactor_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const MatArg<T>& A_in) {
            auto A = owned_copy<T>(A_in);
            const std::size_t m_ = A.num_rows(), n_ = A.num_cols();
            mtl::vec::dense_vector<T> tau;
            {
                nogil guard;
                mtl::lq_factor(A, tau);
            }
            new (self) Wrap{std::move(A), std::move(tau), m_, n_};
        }, "A"_a, "Householder LQ factorization A = L Q")
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.nrows, s.ncols);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("L", [](const Wrap& s) {
            mtl::mat::dense2D<T> L(s.nrows, s.ncols);
            {
                nogil guard;
                L = mtl::lq_extract_L(s.LQ);
            }
            return wrap_matrix<T>(std::move(L));
        }, "The lower triangular factor L")
        .def_prop_ro("Q", [](const Wrap& s) {
            mtl::mat::dense2D<T> Q(s.ncols, s.ncols);
            {
                nogil guard;
                Q = mtl::lq_extract_Q(s.LQ, s.tau);
            }
            return wrap_matrix<T>(std::move(Q));
        }, "The orthogonal factor Q")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5." + name + "(shape=(" + std::to_string(s.nrows) + ", " +
                   std::to_string(s.ncols) + "))";
        });
}

// ===========================================================================
// LDL^T — symmetric, no square roots, so it tolerates a matrix that Cholesky
// rejects. The diagonal D carries the inertia.
// ===========================================================================
template <typename T>
struct LDLTFactor {
    mtl::mat::dense2D<T> LD;   // strict lower = L (unit diagonal), diagonal = D
    std::size_t n;
};

template <typename T>
void register_ldlt(nb::module_& m) {
    using Wrap = LDLTFactor<T>;
    const std::string name = std::string("LDLTFactor_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const MatArg<T>& A_in) {
            auto A = owned_copy<T>(A_in);
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("ldlt: matrix must be square");
            const std::size_t n_ = A.num_rows();
            int info = 0;
            {
                nogil guard;
                info = mtl::ldlt_factor(A);
            }
            if (info != 0)
                throw std::runtime_error(
                    "ldlt: zero pivot in D at index " + std::to_string(info - 1) +
                    ". LDL^T does not pivot. The pivoting variant (Bunch-Kaufman) "
                    "is not exposed yet — MTL5's ldlt_bk is incorrect whenever it "
                    "interchanges, see stillwater-sc/mtl5#335.");
            new (self) Wrap{std::move(A), n_};
        }, "A"_a,
           "LDL^T factorization of a symmetric matrix. Unlike Cholesky it takes "
           "no square roots, so it tolerates an indefinite matrix — but it does "
           "not pivot, so a zero pivot raises.")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def("diagonal", [](const Wrap& s) {
            // D as float64 regardless of element type: this is the number you
            // inspect for inertia, and a Python float is the useful form.
            double* buf = new double[s.n];
            for (std::size_t i = 0; i < s.n; ++i)
                buf[i] = static_cast<double>(s.LD(i, i));
            std::size_t shape[1] = { s.n };
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
        }, "The diagonal D. Its signs are the inertia: any negative entry means "
           "A is indefinite, which is exactly the case Cholesky cannot handle.")
        .def("solve", [](const Wrap& s, const VecArg<T>& b_in) {
            const auto& b = as_vector<T>(b_in);
            if (b.size() != s.n)
                throw std::invalid_argument(
                    "ldlt.solve: RHS length " + std::to_string(b.size()) +
                    " does not match factor size " + std::to_string(s.n));
            mtl::vec::dense_vector<T> x(s.n);
            {
                nogil guard;
                mtl::ldlt_solve(s.LD, x, b);
            }
            return wrap_vector<T>(std::move(x));
        }, "b"_a, "Solve A x = b via the LDL^T factor")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5." + name + "(n=" + std::to_string(s.n) + ")";
        });
}

// ===========================================================================
// Bunch-Kaufman LDL^T — NOT BOUND
//
// mtl::ldlt_bk returns a wrong solution whenever the factorization applies a
// pivot interchange, and reports info == 0 while doing so: normwise backward
// error ~1e-1 instead of ~1e-16. Classifying 400 random symmetric matrices
// (n = 4..12) by pivot pattern: correct in 14/14 cases with no permutation,
// wrong in 37/47 with a 1x1 interchange and 278/339 with a 2x2 block. MTL5's
// own tests cover n = 2 and n = 3, which sit entirely in the regime that works.
//
// Filed as stillwater-sc/mtl5#335. Binding it would mean shipping a solver that
// returns a 60% backward error under a success code, which is worse than not
// shipping it, so the wrapper is withheld until the fix lands. When it does,
// restore this from git history — plain LDL^T below is unaffected.
// ===========================================================================

// ===========================================================================
// Cholesky for the Universal types
//
// float32/float64 Cholesky is already bound in mtl5_module.cpp; this adds the
// same factorization for the Universal element types so #18 can put Cholesky
// and LDL^T side by side in the same precision.
// ===========================================================================
template <typename T>
void register_universal_cholesky(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    using Vec = mtl::vec::dense_vector<T>;
    struct Wrap {
        Mat L;
        std::size_t n;
    };
    const std::string name = std::string("CholeskyFactor_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const Mat& A_in) {
            if (A_in.num_rows() != A_in.num_cols())
                throw std::invalid_argument("cholesky: matrix must be square");
            const std::size_t n_ = A_in.num_rows();
            Mat L(n_, n_);
            int info = 0;
            {
                nogil guard;
                for (std::size_t r = 0; r < n_; ++r)
                    for (std::size_t c = 0; c < n_; ++c) L(r, c) = A_in(r, c);
                info = mtl::cholesky_factor(L);
            }
            if (info != 0)
                throw std::runtime_error(
                    "cholesky: matrix is not symmetric positive definite "
                    "(failure at row " + std::to_string(info - 1) + ")");
            new (self) Wrap{std::move(L), n_};
        }, "A"_a, "Cholesky factorization of an SPD matrix")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def("solve", [](const Wrap& s, const Vec& b) {
            if (b.size() != s.n)
                throw std::invalid_argument("cholesky.solve: dimension mismatch");
            Vec x(s.n);
            {
                nogil guard;
                mtl::cholesky_solve(s.L, x, b);
            }
            return x;
        }, "b"_a, "Solve A x = b using the Cholesky factor")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5." + name + "(n=" + std::to_string(s.n) + ")";
        });
}

template <typename T>
void register_float_only(nb::module_& m) {
    register_qr<T>(m);
    register_lq<T>(m);
}

}  // namespace

// ===========================================================================
void register_dense_factorizations(nb::module_& m) {
    register_float_only<float>(m);
    register_float_only<double>(m);

    // LDL^T everywhere — the point of #18 is comparing number systems.
    register_ldlt<float>(m);
    register_ldlt<double>(m);
    register_ldlt<fp8>(m);
    register_ldlt<fp16>(m);
    register_ldlt<posit8>(m);
    register_ldlt<posit16>(m);
    register_ldlt<posit32>(m);
    register_ldlt<posit64>(m);
    register_ldlt<fixpnt8>(m);
    register_ldlt<fixpnt16>(m);
    register_ldlt<lns16>(m);
    register_ldlt<lns32>(m);

    // Cholesky for the Universal types only; float32/float64 already have it.
    register_universal_cholesky<fp8>(m);
    register_universal_cholesky<fp16>(m);
    register_universal_cholesky<posit8>(m);
    register_universal_cholesky<posit16>(m);
    register_universal_cholesky<posit32>(m);
    register_universal_cholesky<posit64>(m);
    register_universal_cholesky<fixpnt8>(m);
    register_universal_cholesky<fixpnt16>(m);
    register_universal_cholesky<lns16>(m);
    register_universal_cholesky<lns32>(m);
}
