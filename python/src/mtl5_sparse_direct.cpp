// mtl5-python -- sparse direct solvers (MTL5 #116/#153/#154/#182/#184).
//
// The point of this surface, versus scipy.sparse.linalg.splu, is the
// analyze / factor / refactor split:
//
//     splu(A)          analyze (ordering + symbolic) THEN factor (numeric)
//     lu.refactor(A2)  numeric only -- reuses the ordering, the symbolic
//                      structure and the pivot sequence
//
// Refactorization is the transient-circuit workflow: the sparsity pattern is
// fixed across timesteps and only the values change, so the analysis is done
// once. Upstream measures ~2.2x (sparse_lu) and 1.9-3.2x (supernodal) over a
// full factor.
//
// The second thing SciPy cannot do: the factor's precision is independent of
// the residual precision. A float32 factorization refined against a float64
// residual (mtl5.mixed.iterative_refine) is a genuine mixed-precision direct
// solve.

#include "mtl5_types.hpp"
#include "mtl5_sparse_refine.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

namespace fact = mtl::sparse::factorization;
namespace ord  = mtl::sparse::ordering;

const char* const kOrderingHelp =
    "valid orderings: 'amd' (symmetric minimum degree on A+A^T), "
    "'colamd' (column minimum degree on A^T*A; for unsymmetric matrices), "
    "'rcm' (Reverse Cuthill-McKee, bandwidth reducing), "
    "'natural' (identity, no fill reduction)";

/// Fill-reducing ordering as a standalone permutation.
template <typename T>
std::vector<std::size_t> permutation_for(const mtl::mat::compressed2D<T>& A,
                                         const std::string& name) {
    if (name == "amd")     return ord::amd{}(A);
    if (name == "colamd")  return ord::colamd{}(A);
    if (name == "rcm")     return ord::rcm{}(A);
    if (name == "natural") {
        std::vector<std::size_t> p(A.num_rows());
        for (std::size_t i = 0; i < p.size(); ++i) p[i] = i;
        return p;
    }
    throw std::invalid_argument("unknown ordering '" + name + "'; " + kOrderingHelp);
}

/// Symbolic analysis under a named ordering.
template <typename T>
fact::lu_symbolic symbolic_for(const mtl::mat::compressed2D<T>& A,
                               const std::string& name) {
    if (name == "amd")     return fact::sparse_lu_symbolic(A, ord::amd{});
    if (name == "colamd")  return fact::sparse_lu_symbolic(A, ord::colamd{});
    if (name == "rcm")     return fact::sparse_lu_symbolic(A, ord::rcm{});
    if (name == "natural") return fact::sparse_lu_symbolic(A);
    throw std::invalid_argument("unknown ordering '" + name + "'; " + kOrderingHelp);
}

// ===========================================================================
// SparseLU<T> — Gilbert-Peierls left-looking LU with threshold partial pivoting
// ===========================================================================
template <typename T>
struct SparseLU {
    fact::lu_numeric<T> num;
    std::string ordering;
    std::size_t n;

    const fact::lu_numeric<T>& factor() const { return num; }
};

// ===========================================================================
// KLU<T> — block triangular form + per-block LU (the circuit-simulation path)
// ===========================================================================
template <typename T>
struct KLU {
    fact::klu_numeric<T> num;
    std::size_t n;

    const fact::klu_numeric<T>& factor() const { return num; }
};

/// Shared solve body: run the factorization's triangular solves on a copy of b.
template <typename Fac, typename T>
VectorView<T> solve_with(const Fac& fac, std::size_t n,
                         const double* bp, std::size_t blen) {
    if (blen != n)
        throw std::invalid_argument(
            "solve: RHS length " + std::to_string(blen) +
            " does not match factor size " + std::to_string(n));
    mtl::vec::dense_vector<T> x(n);
    {
        nogil guard;
        mtl::vec::dense_vector<T> b(n);
        for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<T>(bp[i]);
        fac.solve(x, b);
    }
    return VectorView<T>(std::move(x));
}

template <typename T>
void register_sparse_lu(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Wrap = SparseLU<T>;
    const std::string name = std::string("SparseLU_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const SMat& A, const std::string& ordering,
                            double threshold, double pivot_perturb) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("SparseLU: matrix must be square");
            // Validate the ordering name before doing any work, so a typo is a
            // clean error rather than a wasted factorization.
            fact::lu_symbolic sym = symbolic_for(A, ordering);
            fact::lu_numeric<T> num;
            {
                nogil guard;
                num = fact::sparse_lu_numeric(A, sym, static_cast<T>(threshold),
                                              static_cast<T>(pivot_perturb));
            }
            new (self) Wrap{std::move(num), ordering, A.num_rows()};
        }, "A"_a, "ordering"_a = "colamd", "threshold"_a = 1.0, "pivot_perturb"_a = 0.0,
           "Analyze and factor a square CSR matrix (Gilbert-Peierls LU with "
           "threshold partial pivoting)")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("ordering", [](const Wrap& s) { return s.ordering; })
        .def_prop_ro("nnz", [](const Wrap& s) {
            return s.num.factorL().nnz() + s.num.factorU().nnz();
        }, "Nonzeros in L plus U — the fill the ordering produced")
        .def_prop_ro("num_perturbed", [](const Wrap& s) { return s.num.num_perturbed; },
                     "Pivots replaced by perturbation; 0 means a clean factor")
        .def("solve", [](const Wrap& s,
                         nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                     nb::device::cpu> b) {
            return solve_with<fact::lu_numeric<T>, T>(s.num, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b using the stored factors")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            // Numeric-only: reuses the ordering, symbolic structure and pivot
            // sequence. Assign only after it succeeds, so a failed refactor
            // leaves the previous factor intact and usable.
            auto fresh = fact::sparse_lu_refactor(A, s.num);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute the numeric factors for a matrix with the SAME sparsity "
           "pattern, reusing the ordering, symbolic analysis and pivot sequence")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", ordering='" + s.ordering + "', nnz=" +
                   std::to_string(s.num.factorL().nnz() + s.num.factorU().nnz()) + ")";
        });
}

template <typename T>
void register_klu(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Wrap = KLU<T>;
    const std::string name = std::string("KLU_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const SMat& A, double threshold,
                            bool scale, double pivot_perturb) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("KLU: matrix must be square");
            fact::klu_numeric<T> num;
            {
                nogil guard;
                num = fact::native_klu_factor(A, static_cast<T>(threshold), scale,
                                              static_cast<T>(pivot_perturb));
            }
            new (self) Wrap{std::move(num), A.num_rows()};
        }, "A"_a, "threshold"_a = 1.0, "scale"_a = true, "pivot_perturb"_a = 0.0,
           "Factor a square CSR matrix with native KLU: Dulmage-Mendelsohn block "
           "triangular form, AMD per block, then a left-looking LU of each "
           "diagonal block")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("nblocks", [](const Wrap& s) { return s.num.nblocks(); },
                     "Diagonal blocks found by the block triangular form; more "
                     "blocks means more reducible structure to exploit")
        .def_prop_ro("num_perturbed", [](const Wrap& s) { return s.num.num_perturbed; },
                     "Pivots replaced by perturbation across all blocks")
        .def("solve", [](const Wrap& s,
                         nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                     nb::device::cpu> b) {
            return solve_with<fact::klu_numeric<T>, T>(s.num, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b by block back-substitution")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            auto fresh = fact::native_klu_refactor(A, s.num);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute the numeric factors for a matrix with the SAME sparsity "
           "pattern, reusing the BTF permutation, per-block ordering and pivot "
           "sequence — the SPICE transient path")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", nblocks=" + std::to_string(s.num.nblocks()) + ")";
        });
}

/// Orderings as standalone permutations, for inspection and for reuse with
/// scipy (`A[perm][:, perm]`).
template <typename T>
void register_orderings(nb::module_& m) {
    m.def("_ordering", [](const mtl::mat::compressed2D<T>& A, const std::string& name) {
        std::vector<std::size_t> perm;
        {
            nogil guard;
            perm = permutation_for(A, name);
        }
        const std::size_t n = perm.size();
        int64_t* buf = new int64_t[n];
        for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<int64_t>(perm[i]);
        std::size_t shape[1] = { n };
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
        return nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(buf, 1, shape, owner);
    }, "A"_a, "name"_a);
}

}  // namespace

// ===========================================================================
void register_sparse_direct(nb::module_& m) {
    register_sparse_lu<float>(m);
    register_sparse_lu<double>(m);
    register_klu<float>(m);
    register_klu<double>(m);
    register_orderings<float>(m);
    register_orderings<double>(m);

    m.def("orderings", []() {
        return std::vector<std::string>{"amd", "colamd", "rcm", "natural"};
    }, "Fill-reducing orderings accepted by the sparse factorizations");

    // Make the direct factorizations usable as the refinement factorization.
    // The narrow-factor overloads (float32 factor, float64 residual) are the
    // mixed-precision direct solve; that pairing is the reason both exist.
    nb::module_ mx = m.attr("mixed");
    mx.def("iterative_refine", &sparse_refine_through<SparseLU<double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3, kRefineDoc);
    mx.def("iterative_refine", &sparse_refine_through<SparseLU<float>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<KLU<double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<KLU<float>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
}
