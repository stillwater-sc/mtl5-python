// mtl5-python -- matrix views (mtl/mat/view).
//
// Upstream these are lazy read-only accessors: each holds a `const Matrix&` and
// computes an element on demand. Both halves of that are wrong for Python.
//
// The reference is a lifetime hazard. It is the same shape as the bug that
// segfaulted pc::ssor through the standalone solve path -- a caller passing a
// NumPy array or a converted scipy matrix hands over a temporary, and a view
// that borrows it dangles the moment the call returns.
//
// The laziness has nothing to buy. A lazy accessor pays for itself by feeding
// an expression template so the elements are never materialised; nothing on the
// Python side consumes one that way, so the first thing any caller does is read
// it out. These therefore materialise, which is also what makes the result a
// normal DenseMatrix that every other binding already accepts.
//
// `hermitian` is the one with a semantic trap. `hermitian_view` is not the
// adjoint: it reads the UPPER triangle as stored and mirrors it conjugated into
// the lower, i.e. "interpret this matrix as Hermitian". mtl5.adjoint(A) is A^H.
// Upstream also does not force the diagonal real, so a source with a complex
// diagonal entry yields a matrix that is not in fact Hermitian -- checked here
// rather than passed through.

#include "mtl5_types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mtl/mat/view/banded_view.hpp>
#include <mtl/mat/view/hermitian_view.hpp>
#include <mtl/mat/view/lower_view.hpp>
#include <mtl/mat/view/map_view.hpp>
#include <mtl/mat/view/strict_lower_view.hpp>
#include <mtl/mat/view/strict_upper_view.hpp>
#include <mtl/mat/view/transposed_view.hpp>
#include <mtl/mat/view/upper_view.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

namespace mv = mtl::mat::view;

/// Read a view out into an owning matrix.
template <typename T, typename View>
MatrixView<T> materialise(const View& v) {
    mtl::mat::dense2D<T> out(v.num_rows(), v.num_cols());
    {
        nogil guard;
        for (std::size_t i = 0; i < v.num_rows(); ++i)
            for (std::size_t j = 0; j < v.num_cols(); ++j)
                out(i, j) = v(i, j);
    }
    return MatrixView<T>(std::move(out));
}

/// The imaginary part, or 0 for a real element type — so the Hermitian
/// diagonal check reads the same for both.
template <typename T>
double imag_of(const T& x) {
    if constexpr (is_complex_v<T>) return static_cast<double>(x.imag());
    else                           { (void)x; return 0.0; }
}

template <typename T>
void register_views_for(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    using MV  = MatrixView<T>;

    m.def("lower", [](const MV& A) {
        return materialise<T>(mv::lower_view<Mat>(A.mat));
    }, "A"_a, "Lower triangle including the diagonal; zero above");

    m.def("upper", [](const MV& A) {
        return materialise<T>(mv::upper_view<Mat>(A.mat));
    }, "A"_a, "Upper triangle including the diagonal; zero below");

    m.def("strict_lower", [](const MV& A) {
        return materialise<T>(mv::strict_lower_view<Mat>(A.mat));
    }, "A"_a, "Lower triangle excluding the diagonal");

    m.def("strict_upper", [](const MV& A) {
        return materialise<T>(mv::strict_upper_view<Mat>(A.mat));
    }, "A"_a, "Upper triangle excluding the diagonal");

    m.def("transposed", [](const MV& A) {
        return materialise<T>(mv::transposed_view<Mat>(A.mat));
    }, "A"_a,
       "Plain transpose. For complex this does NOT conjugate -- mtl5.adjoint(A) "
       "is A^H.");

    m.def("banded", [](const MV& A, std::ptrdiff_t lower, std::ptrdiff_t upper) {
        if (lower < 0 || upper < 0)
            throw nb::value_error(
                "banded: lower and upper are bandwidths and must be >= 0. "
                "banded(A, 1, 1) is tridiagonal and banded(A, 0, 0) the "
                "diagonal; they are counts, not signed diagonal offsets.");
        return materialise<T>(mv::banded_view<Mat>(A.mat, lower, upper));
    }, "A"_a, "lower"_a, "upper"_a,
       "Keep the band from `lower` subdiagonals to `upper` superdiagonals and "
       "zero the rest. Both are bandwidths: banded(A, 1, 1) is tridiagonal.");

    m.def("map", [](const MV& A, const std::vector<std::size_t>& rows,
                    const std::vector<std::size_t>& cols) {
        for (std::size_t r : rows)
            if (r >= A.mat.num_rows())
                throw nb::index_error(
                    ("map: row index " + std::to_string(r) + " out of range").c_str());
        for (std::size_t c : cols)
            if (c >= A.mat.num_cols())
                throw nb::index_error(
                    ("map: column index " + std::to_string(c) + " out of range").c_str());
        return materialise<T>(mv::map_view<Mat>(A.mat, rows, cols));
    }, "A"_a, "rows"_a, "cols"_a,
       "Arbitrary row/column selection: out[i, j] = A[rows[i], cols[j]]. "
       "Indices may repeat and need not be sorted.");

    m.def("hermitian", [](const MV& A) {
        const std::size_t n = A.mat.num_rows();
        if (A.mat.num_cols() != n)
            throw std::invalid_argument("hermitian: matrix must be square");
        // The view mirrors the upper triangle into the lower conjugated, but
        // leaves the diagonal alone. A complex diagonal entry would therefore
        // produce a matrix that is not Hermitian, which is worth refusing
        // rather than returning.
        for (std::size_t i = 0; i < n; ++i) {
            const double im = imag_of<T>(A.mat(i, i));
            if (std::abs(im) > 0.0)
                throw nb::value_error(
                    ("hermitian: A[" + std::to_string(i) + ", " + std::to_string(i) +
                     "] has a non-zero imaginary part, so mirroring the upper "
                     "triangle would not give a Hermitian matrix. A Hermitian "
                     "matrix has a real diagonal.").c_str());
        }
        return materialise<T>(mv::hermitian_view<Mat>(A.mat));
    }, "A"_a,
       "Read A as Hermitian: keep the upper triangle and mirror it conjugated "
       "into the lower.\n\n"
       "This is NOT the adjoint -- it reconstructs a full Hermitian matrix from "
       "one stored triangle, discarding whatever was in the lower. For A^H use "
       "mtl5.adjoint(A). Refuses a non-real diagonal, which the result would "
       "otherwise carry through and not actually be Hermitian.");
}

}  // namespace

// ===========================================================================
void register_views(nb::module_& m) {
    nb::module_ v = m.def_submodule(
        "view", "Triangular, banded, transposed, mapped and Hermitian views");

    register_views_for<float>(v);
    register_views_for<double>(v);
    register_views_for<c64>(v);
    register_views_for<c128>(v);

    v.def("views", []() {
        return std::vector<std::string>{
            "lower", "upper", "strict_lower", "strict_upper",
            "transposed", "banded", "map", "hermitian"};
    }, "The views available in this submodule");
}
