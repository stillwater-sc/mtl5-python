#pragma once
// Sparse iterative refinement, bound once for every factorization kind.
//
// mtl::sparse::iterative_refine drives any object exposing solve(dx, r), so the
// binding is generic over a wrapper type that provides factor(). That covers
// the ILU(0)/IC(0) preconditioners and the sparse direct factorizations alike.
//
// The interesting case is a factor whose Value is NARROWER than the residual
// precision: lu_numeric::solve casts the RHS down to its own Value, solves, and
// writes back, so a float32 (or lower) factorization refined against a float64
// residual is the mixed-precision workflow with no extra machinery.

#include "mtl5_types.hpp"

#include <mtl/sparse/iterative_refine.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

/// Refine x for A x = b through `M`, returning (x, info).
template <typename Wrap>
std::pair<nb::ndarray<nb::numpy, double, nb::ndim<1>>, nb::dict>
sparse_refine_through(const mtl::mat::compressed2D<double>& A, const Wrap& M,
                      nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np,
                      int max_iter, double rel_tol, bool scaled, int patience) {
    const std::size_t n = A.num_rows();
    if (A.num_cols() != n)
        throw std::invalid_argument("iterative_refine: A must be square");
    if (b_np.shape(0) != n)
        throw std::invalid_argument("iterative_refine: len(b) must match A");
    if (M.n != n)
        throw std::invalid_argument(
            "iterative_refine: factorization size " + std::to_string(M.n) +
            " does not match A (" + std::to_string(n) + ")");
    if (patience < 1)
        throw std::invalid_argument("iterative_refine: patience must be >= 1");
    const double* bp = b_np.data();

    mtl::sparse::refine_options opt;
    opt.max_iter = max_iter;
    opt.rel_tol  = rel_tol;
    opt.scaled   = scaled;
    opt.patience = patience;

    double* buf = new double[n];
    mtl::sparse::refine_result res;
    try {
        nogil guard;
        mtl::vec::dense_vector<double> b(n), x(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) b[i] = bp[i];
        res = mtl::sparse::iterative_refine(A, M.factor(), b, x, opt);
        for (std::size_t i = 0; i < n; ++i) buf[i] = x[i];
    } catch (...) {
        delete[] buf;   // no capsule owns it yet
        throw;
    }

    std::size_t shape[1] = { n };
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    auto out = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
    nb::dict d;
    d["iters"]        = res.iters;
    d["rel_residual"] = res.rel_residual;
    d["converged"]    = res.converged;
    return std::make_pair(out, d);
}

inline const char* const kRefineDoc =
    "Refine a solution of A x = b through a factorization M exposing solve(). "
    "A and b are float64; M may be a narrower precision, which is the "
    "mixed-precision workflow. Returns (x, info).";
