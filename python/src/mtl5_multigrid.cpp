// mtl5-python -- smoothers, grid transfer, and multigrid.
//
// Three groups, and one shape decision in each.
//
// **Smoothers** are seven class templates upstream (jacobi, gauss_seidel and
// sor, each of the latter two with backward and symmetric variants). They share
// one interface -- `x = M(x, b)`, a relaxation sweep in place -- so they are
// bound as a single `smooth(A, x, b, kind=...)` rather than seven classes. A
// smoother has no state worth keeping between calls: it is rebuilt from A each
// time, which is what the upstream constructors do anyway.
//
// **Grid transfer** is bound as-is, plus a `galerkin(R, A, P)` that upstream
// does not provide. Building the coarse operator is what multigrid's own
// constructor asks for, and the obvious spelling -- `R * A * P` -- goes through
// operator* on two compressed2D, which returns a *dense* matrix. At fine-grid
// size that is exactly the intermediate you cannot afford, so `galerkin` forms
// the triple product staying sparse throughout.
//
// **multigrid** takes level matrices, restrictors, prolongators, a smoother
// factory and a coarse solver. Handing all five to Python would put the burden
// of building a consistent hierarchy on the caller, so `multigrid_1d(A, ...)`
// builds it: standard 1-D coarsening, Galerkin coarse operators, the chosen
// smoother at every level, and a smoother-based coarse solve. The raw
// constructor stays unbound until there is a caller who needs a hierarchy this
// does not produce.
//
// Note mg/multigrid.hpp is not self-contained -- it uses `A * x` without
// including mat/operators.hpp (stillwater-sc/mtl5#401), so that include comes
// first below and must stay there until the fix lands.

#include "mtl5_types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mtl/mat/inserter.hpp>
#include <mtl/mat/operators.hpp>   // must precede multigrid.hpp -- mtl5#401
#include <mtl/itl/mg/multigrid.hpp>
#include <mtl/itl/mg/prolongation.hpp>
#include <mtl/itl/mg/restriction.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/smoother/jacobi.hpp>
#include <mtl/itl/smoother/sor.hpp>
#include <mtl/operation/mult.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

namespace sm = mtl::itl::smoother;
namespace mg = mtl::itl::mg;

using SMat = mtl::mat::compressed2D<double>;
using Vec  = mtl::vec::dense_vector<double>;

const char* const kSmootherHelp =
    "valid smoothers: 'jacobi', 'gauss_seidel', 'backward_gauss_seidel', "
    "'symmetric_gauss_seidel', 'sor', 'backward_sor', 'symmetric_sor'";

/// Run `sweeps` relaxation sweeps of the named smoother, in place on x.
void apply_smoother(const std::string& kind, const SMat& A, Vec& x, const Vec& b,
                    int sweeps, double omega) {
    auto run = [&](auto&& s) { for (int i = 0; i < sweeps; ++i) s(x, b); };
    if (kind == "jacobi")                 { sm::jacobi<SMat> s(A);                   run(s); }
    else if (kind == "gauss_seidel")      { sm::gauss_seidel<SMat> s(A);             run(s); }
    else if (kind == "backward_gauss_seidel") { sm::backward_gauss_seidel<SMat> s(A); run(s); }
    else if (kind == "symmetric_gauss_seidel"){ sm::symmetric_gauss_seidel<SMat> s(A); run(s); }
    else if (kind == "sor")               { sm::sor<SMat> s(A, omega);               run(s); }
    else if (kind == "backward_sor")      { sm::backward_sor<SMat> s(A, omega);      run(s); }
    else if (kind == "symmetric_sor")     { sm::symmetric_sor<SMat> s(A, omega);     run(s); }
    else throw std::invalid_argument("unknown smoother '" + kind + "'; " + kSmootherHelp);
}

/// Sparse triple product R * A * P, formed without a dense intermediate.
SMat galerkin(const SMat& R, const SMat& A, const SMat& P) {
    if (R.num_cols() != A.num_rows() || A.num_cols() != P.num_rows())
        throw std::invalid_argument(
            "galerkin: shapes must chain as R(c x f) A(f x f) P(f x c)");
    const std::size_t nc = R.num_rows();
    // (RA)(i, :) = sum_k R(i,k) * A(k, :), accumulated in a dense row buffer
    // that is nc-wide only after the second product, so nothing fine-sized is
    // materialised in full.
    const auto& rs = R.ref_major(); const auto& ri = R.ref_minor(); const auto& rv = R.ref_data();
    const auto& as = A.ref_major(); const auto& ai = A.ref_minor(); const auto& av = A.ref_data();
    const auto& ps = P.ref_major(); const auto& pi = P.ref_minor(); const auto& pv = P.ref_data();

    SMat out(nc, nc);
    {
        mtl::mat::inserter<SMat> ins(out);
        std::vector<double> row(nc, 0.0);
        std::vector<std::size_t> touched;
        for (std::size_t i = 0; i < nc; ++i) {
            touched.clear();
            for (std::size_t kk = rs[i]; kk < rs[i + 1]; ++kk) {
                const std::size_t k = ri[kk];
                const double rik = rv[kk];
                for (std::size_t mm = as[k]; mm < as[k + 1]; ++mm) {
                    const std::size_t mcol = ai[mm];
                    const double a = rik * av[mm];
                    // ...then straight through P, so the fine index never
                    // reaches the buffer.
                    for (std::size_t pp = ps[mcol]; pp < ps[mcol + 1]; ++pp) {
                        const std::size_t j = pi[pp];
                        if (row[j] == 0.0) touched.push_back(j);
                        row[j] += a * pv[pp];
                    }
                }
            }
            for (std::size_t j : touched)
                if (row[j] != 0.0) { ins[i][j] << row[j]; row[j] = 0.0; }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Multigrid, with the hierarchy built for the caller
// ---------------------------------------------------------------------------
struct Multigrid {
    std::vector<SMat> levels, restrictors, prolongators;
    std::string smoother;
    double omega;
    int nu_pre, nu_post;
    std::size_t n;

    /// Rebuilt per call: mg::multigrid holds references into the vectors above
    /// and captures the factory lambdas, so keeping one alive across calls
    /// would mean pinning all of that. Construction is cheap next to a cycle.
    template <typename F>
    void with_solver(F&& f) {
        const std::string kind = smoother;
        const double w = omega;
        auto factory = [kind, w](const SMat& M) {
            return [&M, kind, w](Vec& x, const Vec& b) {
                apply_smoother(kind, M, x, b, 1, w);
            };
        };
        const SMat& coarsest = levels.back();
        auto coarse = [&coarsest, kind, w](Vec& x, const Vec& b) {
            apply_smoother(kind, coarsest, x, b, 200, w);
        };
        mg::multigrid<double> solver(levels, restrictors, prolongators,
                                     factory, coarse, nu_pre, nu_post);
        f(solver);
    }
};

Multigrid build_1d(const SMat& A, int n_levels, const std::string& smoother,
                   double omega, int nu_pre, int nu_post) {
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument("multigrid_1d: A must be square");
    if (n_levels < 2)
        throw nb::value_error("multigrid_1d: n_levels must be at least 2");
    // Validate the smoother name up front rather than on the first cycle.
    { Vec probe_x(1, 0.0), probe_b(1, 0.0); SMat one(1, 1);
      { mtl::mat::inserter<SMat> ins(one); ins[0][0] << 1.0; }
      apply_smoother(smoother, one, probe_x, probe_b, 1, omega); }

    Multigrid M;
    M.smoother = smoother; M.omega = omega;
    M.nu_pre = nu_pre; M.nu_post = nu_post; M.n = A.num_rows();
    M.levels.push_back(A);
    for (int l = 1; l < n_levels; ++l) {
        const SMat& fine = M.levels.back();
        if (fine.num_rows() < 4) break;         // no useful coarsening left
        SMat R = mg::make_restriction_1d(fine.num_rows());
        SMat P = mg::make_prolongation_1d(R.num_rows());
        M.levels.push_back(galerkin(R, fine, P));
        M.restrictors.push_back(std::move(R));
        M.prolongators.push_back(std::move(P));
    }
    if (M.levels.size() < 2)
        throw nb::value_error(
            "multigrid_1d: the matrix is too small to coarsen; needs at least "
            "4 rows");
    return M;
}

}  // namespace

// ===========================================================================
void register_multigrid(nb::module_& m) {
    nb::module_ g = m.def_submodule(
        "mg", "Smoothers, grid transfer operators and multigrid");

    g.def("smooth", [](const SMat& A, const VectorView<double>& x,
                       const VectorView<double>& b, const std::string& kind,
                       int sweeps, double omega) {
        if (A.num_rows() != A.num_cols())
            throw std::invalid_argument("smooth: A must be square");
        if (x.vec.size() != A.num_rows() || b.vec.size() != A.num_rows())
            throw std::invalid_argument("smooth: x and b must match A");
        if (sweeps < 1) throw nb::value_error("smooth: sweeps must be >= 1");
        Vec out(A.num_rows()), rhs(A.num_rows());
        for (std::size_t i = 0; i < A.num_rows(); ++i) {
            out[i] = x.vec[i];
            rhs[i] = b.vec[i];
        }
        {
            nogil guard;
            apply_smoother(kind, A, out, rhs, sweeps, omega);
        }
        return VectorView<double>(std::move(out));
    }, "A"_a, "x"_a, "b"_a, "kind"_a = "gauss_seidel", "sweeps"_a = 1,
       "omega"_a = 1.0,
       "Run relaxation sweeps of `kind` on A x = b, starting from x. Returns "
       "the updated x; the input is not modified. `omega` applies to the SOR "
       "variants and is ignored by the others.");

    g.def("smoothers", []() {
        return std::vector<std::string>{
            "jacobi", "gauss_seidel", "backward_gauss_seidel",
            "symmetric_gauss_seidel", "sor", "backward_sor", "symmetric_sor"};
    }, "The smoothers accepted by smooth() and multigrid_1d()");

    g.def("make_restriction_1d", &mg::make_restriction_1d, "n_fine"_a,
          "Full-weighting restriction from n_fine to (n_fine - 1) / 2 points");
    g.def("make_prolongation_1d", &mg::make_prolongation_1d, "n_coarse"_a,
          "Linear interpolation from n_coarse to 2 * n_coarse + 1 points");

    g.def("restrict", [](const SMat& R, const VectorView<double>& v) {
        if (R.num_cols() != v.vec.size())
            throw std::invalid_argument("restrict: R.num_cols must equal len(v)");
        Vec in(v.vec.size());
        for (std::size_t i = 0; i < v.vec.size(); ++i) in[i] = v.vec[i];
        nogil guard;
        return VectorView<double>(mg::restrict(R, in));
    }, "R"_a, "v"_a, "Restrict a fine-grid vector to the coarse grid");

    g.def("prolongate", [](const SMat& P, const VectorView<double>& v) {
        if (P.num_cols() != v.vec.size())
            throw std::invalid_argument("prolongate: P.num_cols must equal len(v)");
        Vec in(v.vec.size());
        for (std::size_t i = 0; i < v.vec.size(); ++i) in[i] = v.vec[i];
        nogil guard;
        return VectorView<double>(mg::prolongate(P, in));
    }, "P"_a, "v"_a, "Interpolate a coarse-grid vector to the fine grid");

    g.def("galerkin", [](const SMat& R, const SMat& A, const SMat& P) {
        nogil guard;
        return galerkin(R, A, P);
    }, "R"_a, "A"_a, "P"_a,
       "Coarse-grid operator R A P, formed sparsely. Spelling it as R @ A @ P "
       "instead would go through a sparse-times-sparse product that returns a "
       "dense matrix, materialising a fine-sized intermediate.");

    // -- the hierarchy ------------------------------------------------------
    nb::class_<Multigrid>(g, "Multigrid",
        "A multigrid hierarchy built by multigrid_1d()")
        .def_prop_ro("n", [](const Multigrid& M) { return M.n; })
        .def_prop_ro("n_levels", [](const Multigrid& M) { return M.levels.size(); })
        .def_prop_ro("level_sizes", [](const Multigrid& M) {
            std::vector<std::size_t> out;
            for (const auto& L : M.levels) out.push_back(L.num_rows());
            return out;
        }, "Rows at each level, finest first")
        .def_prop_ro("smoother", [](const Multigrid& M) { return M.smoother; })
        .def("vcycle", [](Multigrid& M, const VectorView<double>& x,
                          const VectorView<double>& b, int cycles) {
            if (x.vec.size() != M.n || b.vec.size() != M.n)
                throw std::invalid_argument("vcycle: x and b must match the hierarchy");
            if (cycles < 1) throw nb::value_error("vcycle: cycles must be >= 1");
            Vec xx(M.n), bb(M.n);
            for (std::size_t i = 0; i < M.n; ++i) { xx[i] = x.vec[i]; bb[i] = b.vec[i]; }
            {
                nogil guard;
                M.with_solver([&](auto& s) {
                    for (int c = 0; c < cycles; ++c) s.vcycle(xx, bb);
                });
            }
            return VectorView<double>(std::move(xx));
        }, "x"_a, "b"_a, "cycles"_a = 1, "Run V-cycles, returning the updated x")
        .def("wcycle", [](Multigrid& M, const VectorView<double>& x,
                          const VectorView<double>& b, int cycles) {
            if (x.vec.size() != M.n || b.vec.size() != M.n)
                throw std::invalid_argument("wcycle: x and b must match the hierarchy");
            if (cycles < 1) throw nb::value_error("wcycle: cycles must be >= 1");
            Vec xx(M.n), bb(M.n);
            for (std::size_t i = 0; i < M.n; ++i) { xx[i] = x.vec[i]; bb[i] = b.vec[i]; }
            {
                nogil guard;
                M.with_solver([&](auto& s) {
                    for (int c = 0; c < cycles; ++c) s.wcycle(xx, bb);
                });
            }
            return VectorView<double>(std::move(xx));
        }, "x"_a, "b"_a, "cycles"_a = 1,
           "Run W-cycles -- two coarse-grid corrections per level instead of "
           "one, which costs more per cycle and is more robust")
        .def("__repr__", [](const Multigrid& M) {
            std::string s = "mtl5.mg.Multigrid(n=" + std::to_string(M.n) +
                            ", levels=" + std::to_string(M.levels.size()) +
                            ", smoother='" + M.smoother + "')";
            return s;
        });

    g.def("multigrid_1d", &build_1d, "A"_a, "n_levels"_a = 3,
          "smoother"_a = "gauss_seidel", "omega"_a = 1.0,
          "nu_pre"_a = 2, "nu_post"_a = 2,
          "Build a geometric multigrid hierarchy for a 1-D-structured matrix.\n\n"
          "Coarsens with standard 1-D full-weighting restriction and linear "
          "interpolation, forming each coarse operator as the Galerkin product "
          "R A P. Coarsening stops early if a level would fall below 4 rows, so "
          "n_levels is an upper bound -- check `level_sizes`.");
}
