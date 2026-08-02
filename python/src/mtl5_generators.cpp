// mtl5-python -- test-matrix generators and NumPy-style range vectors
// (MTL5 #272/#293, ported from Universal's sw::blas).
//
// A named catalog of matrices with known pathologies: Hilbert and Lotkin for
// catastrophic conditioning, Frank and Wilkinson for clustered eigenvalues,
// Kahan and Moler for QR/Cholesky stress, Clement and Rosser for exact known
// spectra. These are the inputs a mixed-precision experiment actually wants,
// and hand-rolling them in Python is both tedious and easy to get subtly wrong.
//
// Everything here generates in float64. The Python layer's `dtype=` argument
// then rounds to the requested element type, which is the right semantics for
// a test matrix: the definitions are over the reals, so you want the correctly
// rounded representation of the exact entry rather than the result of
// evaluating the formula in low-precision arithmetic. (For the common case
// they coincide -- 1/(i+j+1) with an exactly representable denominator rounds
// identically either way -- but the convert route is well-defined for every
// generator, including the ones that build via a factorization.)
//
// Doing it that way also keeps one instantiation per generator instead of one
// per (generator, dtype) pair.

#include "mtl5_types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mtl/generators/clement.hpp>
#include <mtl/generators/companion.hpp>
#include <mtl/generators/forsythe.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/hilbert.hpp>
#include <mtl/generators/kahan.hpp>
#include <mtl/generators/laplacian.hpp>
#include <mtl/generators/lehmer.hpp>
#include <mtl/generators/lotkin.hpp>
#include <mtl/generators/magic.hpp>
#include <mtl/generators/minij.hpp>
#include <mtl/generators/moler.hpp>
#include <mtl/generators/ones.hpp>
#include <mtl/generators/pascal.hpp>
#include <mtl/generators/poisson.hpp>
#include <mtl/generators/randorth.hpp>
#include <mtl/generators/randspd.hpp>
#include <mtl/generators/randsvd.hpp>
#include <mtl/generators/randsym.hpp>
#include <mtl/generators/ranges.hpp>
#include <mtl/generators/rosser.hpp>
#include <mtl/generators/testsuite.hpp>
#include <mtl/generators/vandermonde.hpp>
#include <mtl/generators/wilkinson.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

namespace gen = mtl::generators;

/// Several generators (hilbert, lehmer, lotkin, minij, ones) are lazy
/// expression types exposing operator()(r,c) rather than dense2D. Evaluate one
/// into an owning matrix.
template <typename Expr>
MatrixView<double> materialize(const Expr& e) {
    const std::size_t rows = e.num_rows(), cols = e.num_cols();
    mtl::mat::dense2D<double> out(rows, cols);
    {
        nogil guard;
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < cols; ++c)
                out(r, c) = static_cast<double>(e(r, c));
    }
    return MatrixView<double>(std::move(out));
}

MatrixView<double> wrap(mtl::mat::dense2D<double>&& m) {
    return MatrixView<double>(std::move(m));
}

VectorView<double> wrapv(mtl::vec::dense_vector<double>&& v) {
    return VectorView<double>(std::move(v));
}

void require_positive(std::size_t n, const char* who) {
    if (n == 0) throw std::invalid_argument(std::string(who) + ": n must be >= 1");
}

}  // namespace

// ===========================================================================
void register_generators(nb::module_& m) {
    nb::module_ g = m.def_submodule(
        "generators", "Named test matrices with known pathologies, and "
                      "NumPy-style range vectors");

    // ----- square, size only -------------------------------------------------
    g.def("clement", [](std::size_t n) {
        require_positive(n, "clement");
        nogil guard; return wrap(gen::clement<double>(n));
    }, "n"_a,
       "Tridiagonal with eigenvalues exactly -(n-1), -(n-3), ..., n-1 — a "
       "spectrum you can check against without a reference implementation");

    g.def("frank", [](std::size_t n) {
        require_positive(n, "frank");
        nogil guard; return wrap(gen::frank<double>(n));
    }, "n"_a, "Upper Hessenberg with ill-conditioned small eigenvalues");

    g.def("pascal", [](std::size_t n) {
        require_positive(n, "pascal");
        nogil guard; return wrap(gen::pascal<double>(n));
    }, "n"_a, "Symmetric positive definite, binomial entries, grows ill-conditioned fast");

    g.def("wilkinson", [](std::size_t n) {
        if (n % 2 == 0)
            throw std::invalid_argument("wilkinson: n must be odd");
        require_positive(n, "wilkinson");
        nogil guard; return wrap(gen::wilkinson<double>(n));
    }, "n"_a, "Symmetric tridiagonal with pairs of nearly-equal eigenvalues (odd n)");

    g.def("rosser", []() {
        nogil guard; return wrap(gen::rosser<double>());
    }, "The 8x8 Rosser matrix: a double eigenvalue, a zero eigenvalue, and two "
       "eigenvalues that differ only in the last bits");

    g.def("magic", [](std::size_t n) {
        require_positive(n, "magic");
        nogil guard; return wrap(gen::magic<double>(n));
    }, "n"_a,
       "Magic square of order n. Singly-even orders (n = 4k+2) are not "
       "supported upstream and raise.");

    g.def("hilbert", [](std::size_t n) {
        require_positive(n, "hilbert");
        return materialize(gen::hilbert<double>(n));
    }, "n"_a,
       "H(i,j) = 1/(i+j+1) — the canonical ill-conditioned matrix; the "
       "condition number grows roughly like e^(3.5n)");

    g.def("lehmer", [](std::size_t n) {
        require_positive(n, "lehmer");
        return materialize(gen::lehmer<double>(n));
    }, "n"_a, "Symmetric positive definite, min(i,j)/max(i,j); inverse is tridiagonal");

    g.def("lotkin", [](std::size_t n) {
        require_positive(n, "lotkin");
        return materialize(gen::lotkin<double>(n));
    }, "n"_a, "Hilbert with the first row replaced by ones — nonsymmetric and ill-conditioned");

    g.def("minij", [](std::size_t n) {
        require_positive(n, "minij");
        return materialize(gen::minij<double>(n));
    }, "n"_a, "A(i,j) = min(i,j)+1, symmetric positive definite");

    g.def("ones", [](std::size_t rows, std::size_t cols) {
        require_positive(rows, "ones");
        return materialize(gen::ones<double>(rows, cols ? cols : rows));
    }, "m"_a, "n"_a = 0, "All-ones matrix; n defaults to m");

    // ----- parameterized -----------------------------------------------------
    g.def("forsythe", [](std::size_t n, double alpha) {
        require_positive(n, "forsythe");
        nogil guard; return wrap(gen::forsythe<double>(n, alpha));
    }, "n"_a, "alpha"_a = 1e-10,
       "Perturbed Jordan block: a companion matrix whose eigenvalues are the "
       "n-th roots of alpha, so a tiny alpha makes them cluster savagely");

    g.def("kahan", [](std::size_t n, double theta) {
        require_positive(n, "kahan");
        nogil guard; return wrap(gen::kahan<double>(n, theta));
    }, "n"_a, "theta"_a = 1.2,
       "Upper triangular; column-pivoted QR fails to reveal its rank");

    g.def("moler", [](std::size_t n, double alpha) {
        require_positive(n, "moler");
        nogil guard; return wrap(gen::moler<double>(n, alpha));
    }, "n"_a, "alpha"_a = -1.0,
       "Symmetric positive definite with one small eigenvalue");

    g.def("companion", [](const std::vector<double>& coeffs) {
        if (coeffs.empty())
            throw std::invalid_argument("companion: need at least one coefficient");
        nogil guard; return wrap(gen::companion<double>(coeffs));
    }, "coeffs"_a, "Companion matrix of a monic polynomial; its eigenvalues are the roots");

    g.def("vandermonde", [](const std::vector<double>& nodes) {
        if (nodes.empty())
            throw std::invalid_argument("vandermonde: need at least one node");
        nogil guard; return wrap(gen::vandermonde<double>(nodes));
    }, "nodes"_a, "Vandermonde matrix on the given nodes — ill-conditioned for clustered nodes");

    // ----- random with prescribed structure ----------------------------------
    g.def("randorth", [](std::size_t n) {
        require_positive(n, "randorth");
        nogil guard; return wrap(gen::randorth<double>(n));
    }, "n"_a, "Random orthogonal matrix (Q from a QR of a random matrix)");

    g.def("randspd", [](std::size_t n, const std::vector<double>& eigenvalues) {
        if (eigenvalues.size() != n)
            throw std::invalid_argument("randspd: need exactly n eigenvalues");
        for (double e : eigenvalues)
            if (e <= 0.0)
                throw std::invalid_argument("randspd: all eigenvalues must be positive");
        nogil guard; return wrap(gen::randspd<double>(n, eigenvalues));
    }, "n"_a, "eigenvalues"_a,
       "Symmetric positive definite with exactly the prescribed spectrum — the "
       "way to build a matrix of a chosen condition number");

    g.def("randsym", [](std::size_t n, const std::vector<double>& eigenvalues) {
        if (eigenvalues.size() != n)
            throw std::invalid_argument("randsym: need exactly n eigenvalues");
        nogil guard; return wrap(gen::randsym<double>(n, eigenvalues));
    }, "n"_a, "eigenvalues"_a,
       "Symmetric with the prescribed spectrum; negative eigenvalues are allowed, "
       "so this is how to build a controlled indefinite matrix");

    g.def("randsvd", [](std::size_t rows, std::size_t cols, double kappa, int mode) {
        require_positive(rows, "randsvd");
        require_positive(cols, "randsvd");
        if (!(kappa >= 1.0))
            throw std::invalid_argument("randsvd: kappa must be >= 1");
        nogil guard; return wrap(gen::randsvd<double>(rows, cols, kappa, mode));
    }, "m"_a, "n"_a, "kappa"_a, "mode"_a = 3,
       "Random matrix with condition number exactly kappa; mode selects how the "
       "singular values are distributed between 1 and 1/kappa");

    // ----- sparse ------------------------------------------------------------
    // These return CSR, not dense — they are the structured sparse systems the
    // iterative and direct solvers are usually benchmarked on.
    g.def("laplacian_1d", [](std::size_t n) {
        require_positive(n, "laplacian_1d");
        nogil guard; return gen::laplacian_1d<double>(n);
    }, "n"_a, "1-D Laplacian, tridiagonal [-1 2 -1], as a sparse matrix");

    g.def("laplacian_2d", [](std::size_t nx, std::size_t ny) {
        require_positive(nx, "laplacian_2d");
        require_positive(ny, "laplacian_2d");
        nogil guard; return gen::laplacian_2d<double>(nx, ny);
    }, "nx"_a, "ny"_a, "2-D 5-point Laplacian on an nx-by-ny grid, as a sparse matrix");

    g.def("poisson2d", [](std::size_t nx, std::size_t ny) {
        require_positive(nx, "poisson2d");
        require_positive(ny, "poisson2d");
        nogil guard; return gen::poisson2d_dirichlet<double>(nx, ny);
    }, "nx"_a, "ny"_a, "2-D Poisson with Dirichlet boundaries, as a sparse matrix");

    // ----- the named test suite ----------------------------------------------
    // names() and kappa() are pure data. by_name() is deliberately absent: it
    // reads .mtx files out of the MTL5 source tree, which a wheel does not
    // carry. Load those with mtl5.sparse.from_scipy(scipy.io.mmread(...)).
    g.def("testsuite_names", []() { return mtl::testsuite::names(); },
          "Names in MTL5's published test-matrix catalog");

    g.def("testsuite_kappa", [](const std::string& name) {
        return mtl::testsuite::kappa(name);
    }, "name"_a,
       "Published condition number for a catalog matrix — a reference value to "
       "check a computed condition number against");

    // ----- range vectors -----------------------------------------------------
    m.def("arange", [](std::int64_t start, std::int64_t stop, std::int64_t step) {
        if (step == 0) throw std::invalid_argument("arange: step must be nonzero");
        nogil guard; return wrapv(gen::arange<double>(start, stop, step));
    }, "start"_a, "stop"_a, "step"_a = 1, "Integer-stepped range, NumPy semantics");

    m.def("linspace", [](double start, double stop, std::size_t steps, bool endpoint) {
        nogil guard; return wrapv(gen::linspace<double>(start, stop, steps, endpoint));
    }, "start"_a, "stop"_a, "steps"_a, "endpoint"_a = true,
       "Evenly spaced values, NumPy semantics");

    m.def("logspace", [](double start, double stop, std::size_t steps, bool endpoint,
                         double base) {
        nogil guard; return wrapv(gen::logspace<double>(start, stop, steps, endpoint, base));
    }, "start"_a, "stop"_a, "steps"_a, "endpoint"_a = true, "base"_a = 10.0,
       "Evenly spaced on a log scale between base**start and base**stop");

    m.def("geomspace", [](double start, double stop, std::size_t steps, bool endpoint) {
        if (start == 0.0 || stop == 0.0)
            throw std::invalid_argument("geomspace: endpoints must be nonzero");
        nogil guard; return wrapv(gen::geomspace<double>(start, stop, steps, endpoint));
    }, "start"_a, "stop"_a, "steps"_a, "endpoint"_a = true,
       "Geometric progression between the endpoints themselves — note this is a "
       "true geometric progression, not logspace's exponents");
}
