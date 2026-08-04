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

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <mtl/math/accumulator_traits.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

#include <cstddef>
#include <optional>
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
// ===========================================================================
// Accumulator policy for the numeric factorizations
//
// sparse_lu_numeric, native_klu_factor, supernodal_lu_numeric and
// supernodal_ldlt_numeric each take an Accumulator template parameter that
// types their dense numeric workspace -- `std::vector<Accumulator> x(n)` in the
// left-looking solve -- and round once, via accumulator_traits::value(), when a
// value is written into L or U. So a float32 factor can accumulate its updates
// in float64: the factor stays narrow (every one of these returns a type
// parameterized on Value alone, not on Accumulator), while the arithmetic that
// produced it is wider. That is the capability a fixed-precision library like
// SuiteSparse structurally cannot offer.
//
// Because the result type does not depend on the accumulator, one wrapper class
// per element type is still enough; only the construction call dispatches.
//
// The quire is deliberately absent: SparseMatrix is float32/float64 only, and
// Universal defines a quire only for its own number systems. parse_acc already
// says so when asked.
template <typename T>
AccKind parse_sparse_acc(const std::optional<std::string>& spec) {
    const AccKind kind = parse_acc(spec, type_suffix<T>(), /*quire_ok=*/false);
    if constexpr (std::is_same_v<T, double>) {
        if (kind == AccKind::F32 || kind == AccKind::FMA32)
            throw std::invalid_argument(
                "accumulator='f32'/'fma32' is narrower than the float64 element "
                "type, so it would lose precision rather than gain it. Use "
                "'f64', 'fma64', or None for the element precision.");
    }
    return kind;
}

/// Invoke `f.template operator()<Acc>()` for the selected accumulator. Every
/// branch yields the same type, since the factor type depends only on T.
template <typename T, typename F>
decltype(auto) with_acc(AccKind kind, F&& f) {
    using fma32 = mtl::math::fma_accumulator<float>;
    using fma64 = mtl::math::fma_accumulator<double>;
    if (kind == AccKind::Default) return f.template operator()<T>();
    if (kind == AccKind::F64)     return f.template operator()<double>();
    if (kind == AccKind::FMA64)   return f.template operator()<fma64>();
    if constexpr (std::is_same_v<T, float>) {
        // For a float64 element these are rejected in parse_sparse_acc, so
        // guarding here keeps them from being instantiated at all.
        if (kind == AccKind::F32)   return f.template operator()<float>();
        if (kind == AccKind::FMA32) return f.template operator()<fma32>();
    }
    throw std::invalid_argument("unsupported accumulator for a sparse factorization");
}

/// Refactor reuses the stored pivot sequence through an entry point that takes
/// no Accumulator, so it cannot honour a non-default policy. Refusing is the
/// honest option: silently refactoring in element precision would quietly undo
/// what the caller asked for.
inline void require_default_acc_for_refactor(const std::string& acc, const char* what) {
    if (acc != "default")
        throw std::invalid_argument(
            std::string("refactor: this factorization was built with "
                        "accumulator='") + acc + "', and " + what +
            " reuses the stored pivot sequence through an entry point that takes "
            "no accumulator -- it would silently drop back to element precision. "
            "Construct a new factorization instead.");
}

template <typename T>
struct SparseLU {
    fact::lu_numeric<T> num;
    std::string ordering;
    std::size_t n;
    std::string accumulator;

    const fact::lu_numeric<T>& factor() const { return num; }
};

// ===========================================================================
// KLU<T> — block triangular form + per-block LU (the circuit-simulation path)
// ===========================================================================
template <typename T>
struct KLU {
    fact::klu_numeric<T> num;
    std::size_t n;
    std::string accumulator;

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
                            double threshold, double pivot_perturb,
                            std::optional<std::string> accumulator) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("SparseLU: matrix must be square");
            const AccKind acc = parse_sparse_acc<T>(accumulator);
            // Validate the ordering name before doing any work, so a typo is a
            // clean error rather than a wasted factorization.
            fact::lu_symbolic sym = symbolic_for(A, ordering);
            fact::lu_numeric<T> num;
            {
                nogil guard;
                num = with_acc<T>(acc, [&]<typename Acc>() {
                    return fact::sparse_lu_numeric<T, mtl::mat::parameters<>, Acc>(
                        A, sym, static_cast<T>(threshold),
                        static_cast<T>(pivot_perturb));
                });
            }
            new (self) Wrap{std::move(num), ordering, A.num_rows(),
                            accumulator.value_or("default")};
        }, "A"_a, "ordering"_a = "colamd", "threshold"_a = 1.0, "pivot_perturb"_a = 0.0,
           "accumulator"_a = nb::none(),
           "Analyze and factor a square CSR matrix (Gilbert-Peierls LU with "
           "threshold partial pivoting).\n\n"
           "`accumulator=` types the dense numeric workspace, so a float32 "
           "factor can accumulate its updates in float64 ('f64') or with an FMA "
           "('fma32'/'fma64'). The factor itself stays in the element type; only "
           "the arithmetic that built it widens.")
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
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def("solve", [](const Wrap& s,
                         nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                     nb::device::cpu> b) {
            return solve_with<fact::lu_numeric<T>, T>(s.num, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b using the stored factors")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            require_default_acc_for_refactor(s.accumulator, "sparse_lu_refactor");
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
                            bool scale, double pivot_perturb,
                            std::optional<std::string> accumulator) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("KLU: matrix must be square");
            const AccKind acc = parse_sparse_acc<T>(accumulator);
            fact::klu_numeric<T> num;
            {
                nogil guard;
                num = with_acc<T>(acc, [&]<typename Acc>() {
                    return fact::native_klu_factor<T, mtl::mat::parameters<>, Acc>(
                        A, static_cast<T>(threshold), scale,
                        static_cast<T>(pivot_perturb));
                });
            }
            new (self) Wrap{std::move(num), A.num_rows(),
                            accumulator.value_or("default")};
        }, "A"_a, "threshold"_a = 1.0, "scale"_a = true, "pivot_perturb"_a = 0.0,
           "accumulator"_a = nb::none(),
           "Factor a square CSR matrix with native KLU: Dulmage-Mendelsohn block "
           "triangular form, AMD per block, then a left-looking LU of each "
           "diagonal block.\n\n"
           "`accumulator=` types the dense numeric workspace of each block's LU "
           "-- see SparseLU.")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("nblocks", [](const Wrap& s) { return s.num.nblocks(); },
                     "Diagonal blocks found by the block triangular form; more "
                     "blocks means more reducible structure to exploit")
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def_prop_ro("nnz", [](const Wrap& s) {
            // Summed over the diagonal blocks: the off-diagonal blocks are not
            // factored, which is exactly where BTF saves work.
            std::size_t total = 0;
            for (const auto& blk : s.num.block_numeric)
                total += blk.factorL().nnz() + blk.factorU().nnz();
            return total;
        }, "Nonzeros in L plus U, summed over the diagonal blocks")
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
            require_default_acc_for_refactor(s.accumulator, "native_klu_refactor");
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

// ===========================================================================
// Cholesky / LDL^T / QR / supernodal
//
// None of these has a native refactor entry point the way sparse_lu and KLU do,
// but every one takes a precomputed symbolic analysis, so `.refactor(A)` reuses
// the ordering and the symbolic structure and recomputes only the numeric
// factor. For Cholesky and LDL^T that is the whole analysis — neither pivots,
// so there is no pivot sequence left to reuse.
// ===========================================================================

/// Symbolic dispatch for the analyses that take an Ordering functor.
#define MTL5PY_SYMBOLIC_DISPATCH(fn)                                          \
    if (name == "amd")     return fact::fn(A, ord::amd{});                    \
    if (name == "colamd")  return fact::fn(A, ord::colamd{});                 \
    if (name == "rcm")     return fact::fn(A, ord::rcm{});                    \
    if (name == "natural") return fact::fn(A);                                \
    throw std::invalid_argument("unknown ordering '" + name + "'; " + kOrderingHelp)

template <typename T>
fact::cholesky_symbolic chol_symbolic(const mtl::mat::compressed2D<T>& A,
                                      const std::string& name) {
    MTL5PY_SYMBOLIC_DISPATCH(sparse_cholesky_symbolic);
}

template <typename T>
fact::ldlt_symbolic ldlt_symbolic_for(const mtl::mat::compressed2D<T>& A,
                                      const std::string& name) {
    MTL5PY_SYMBOLIC_DISPATCH(sparse_ldlt_symbolic);
}

template <typename T>
fact::qr_symbolic qr_symbolic_for(const mtl::mat::compressed2D<T>& A,
                                  const std::string& name) {
    MTL5PY_SYMBOLIC_DISPATCH(sparse_qr_symbolic);
}

template <typename T>
fact::supernodal_lu_symbolic snlu_symbolic(const mtl::mat::compressed2D<T>& A,
                                           const std::string& name) {
    // No natural-ordering overload upstream; the identity permutation through
    // the ordered path is equivalent.
    if (name == "amd")     return fact::supernodal_lu_symbolic_analyze(A, ord::amd{});
    if (name == "colamd")  return fact::supernodal_lu_symbolic_analyze(A, ord::colamd{});
    if (name == "rcm")     return fact::supernodal_lu_symbolic_analyze(A, ord::rcm{});
    throw std::invalid_argument(
        "supernodal LU ordering must be 'amd', 'colamd' or 'rcm' (got '" + name + "')");
}

template <typename T>
fact::supernodal_symbolic snldlt_symbolic(const mtl::mat::compressed2D<T>& A,
                                          const std::string& name) {
    MTL5PY_SYMBOLIC_DISPATCH(supernodal_ldlt_symbolic);
}

#undef MTL5PY_SYMBOLIC_DISPATCH

template <typename T>
struct SparseCholesky {
    fact::cholesky_symbolic sym;
    fact::cholesky_numeric<T> num;
    std::string ordering;
    std::size_t n;
    const fact::cholesky_numeric<T>& factor() const { return num; }
};

template <typename T>
struct SparseLDLT {
    fact::ldlt_symbolic sym;
    fact::ldlt_numeric<T> num;
    std::string ordering;
    std::size_t n;
    const fact::ldlt_numeric<T>& factor() const { return num; }
};

template <typename T>
struct SparseQR {
    fact::qr_symbolic sym;
    fact::qr_numeric<T> num;
    std::string ordering;
    std::size_t nrows, ncols;
};

template <typename T>
struct SupernodalLU {
    fact::supernodal_lu_factor<T> num;
    std::string ordering;
    double threshold;
    std::size_t max_super;
    bool scale;
    double pivot_perturb;
    std::size_t n;
    std::string accumulator;
    const fact::supernodal_lu_factor<T>& factor() const { return num; }
};

template <typename T>
struct SupernodalLDLT {
    fact::supernodal_symbolic sym;
    fact::supernodal_ldlt_factor<T> num;
    std::string ordering;
    std::size_t n;
    std::string accumulator;
    const fact::supernodal_ldlt_factor<T>& factor() const { return num; }
};

template <typename T>
void register_cholesky(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Wrap = SparseCholesky<T>;
    const std::string name = std::string("SparseCholesky_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const SMat& A, const std::string& ordering) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("SparseCholesky: matrix must be square");
            auto sym = chol_symbolic(A, ordering);
            fact::cholesky_numeric<T> num;
            {
                nogil guard;
                num = fact::sparse_cholesky_numeric(A, sym);
            }
            new (self) Wrap{std::move(sym), std::move(num), ordering, A.num_rows()};
        }, "A"_a, "ordering"_a = "amd",
           "Cholesky factorization A = L L^T of a symmetric positive definite "
           "matrix. Raises RuntimeError if A is not positive definite.")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("ordering", [](const Wrap& s) { return s.ordering; })
        .def_prop_ro("nnz", [](const Wrap& s) { return s.num.factor().nnz(); },
                     "Nonzeros in the lower factor L")
        .def("solve", [](const Wrap& s, nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::cholesky_numeric<T>, T>(s.num, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b via the Cholesky factor")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            auto fresh = fact::sparse_cholesky_numeric(A, s.sym);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute L for a matrix with the SAME sparsity pattern, reusing the "
           "ordering and symbolic analysis (Cholesky does not pivot, so that is "
           "the entire analysis)")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", ordering='" + s.ordering + "', nnz=" +
                   std::to_string(s.num.factor().nnz()) + ")";
        });
}

template <typename T>
void register_ldlt(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Wrap = SparseLDLT<T>;
    const std::string name = std::string("SparseLDLT_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const SMat& A, const std::string& ordering) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("SparseLDLT: matrix must be square");
            auto sym = ldlt_symbolic_for(A, ordering);
            fact::ldlt_numeric<T> num;
            {
                nogil guard;
                num = fact::sparse_ldlt_numeric(A, sym);
            }
            new (self) Wrap{std::move(sym), std::move(num), ordering, A.num_rows()};
        }, "A"_a, "ordering"_a = "amd",
           "LDL^T factorization of a symmetric matrix. Unlike Cholesky it does "
           "not require positive definiteness, but it does not pivot either, so "
           "it raises RuntimeError on a zero pivot.")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("ordering", [](const Wrap& s) { return s.ordering; })
        .def_prop_ro("nnz", [](const Wrap& s) { return s.num.factorL().nnz(); },
                     "Nonzeros in the unit lower factor L")
        .def("diagonal", [](const Wrap& s) {
            const auto& d = s.num.diagonal();
            double* buf = new double[d.size()];
            for (std::size_t i = 0; i < d.size(); ++i) buf[i] = static_cast<double>(d[i]);
            std::size_t shape[1] = { d.size() };
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
        }, "The diagonal D. Its signs give the inertia: a negative entry means A "
           "is indefinite.")
        .def("solve", [](const Wrap& s, nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::ldlt_numeric<T>, T>(s.num, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b via the LDL^T factor")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            auto fresh = fact::sparse_ldlt_numeric(A, s.sym);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute L and D for a matrix with the SAME sparsity pattern, "
           "reusing the ordering and symbolic analysis")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", ordering='" + s.ordering + "', nnz=" +
                   std::to_string(s.num.factorL().nnz()) + ")";
        });
}

template <typename T>
void register_qr(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Wrap = SparseQR<T>;
    const std::string name = std::string("SparseQR_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const SMat& A, const std::string& ordering) {
            if (A.num_rows() < A.num_cols())
                throw std::invalid_argument(
                    "SparseQR: needs num_rows >= num_cols (overdetermined or square)");
            auto sym = qr_symbolic_for(A, ordering);
            fact::qr_numeric<T> num;
            {
                nogil guard;
                num = fact::sparse_qr_numeric(A, sym);
            }
            new (self) Wrap{std::move(sym), std::move(num), ordering,
                            A.num_rows(), A.num_cols()};
        }, "A"_a, "ordering"_a = "colamd",
           "Householder QR of a tall-or-square sparse matrix, for linear least "
           "squares. Accepts rectangular A, unlike the other factorizations.")
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.nrows, s.ncols);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("ordering", [](const Wrap& s) { return s.ordering; })
        .def_prop_ro("nnz", [](const Wrap& s) { return s.num.R.nnz(); },
                     "Nonzeros in the upper factor R")
        .def("solve", [](const Wrap& s, nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            if (b.shape(0) != s.nrows)
                throw std::invalid_argument(
                    "solve: RHS length " + std::to_string(b.shape(0)) +
                    " does not match the row count " + std::to_string(s.nrows));
            const double* bp = b.data();
            mtl::vec::dense_vector<T> x(s.ncols);
            {
                nogil guard;
                mtl::vec::dense_vector<T> rhs(s.nrows);
                for (std::size_t i = 0; i < s.nrows; ++i) rhs[i] = static_cast<T>(bp[i]);
                s.num.solve(x, rhs);
            }
            return VectorView<T>(std::move(x));
        }, "b"_a, "Least-squares solution of min ||A x - b||_2")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.nrows || A.num_cols() != s.ncols)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            auto fresh = fact::sparse_qr_numeric(A, s.sym);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute R and the Householder vectors for a matrix with the SAME "
           "sparsity pattern, reusing the ordering and symbolic analysis")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5.sparse." + name + "(shape=(" + std::to_string(s.nrows) +
                   ", " + std::to_string(s.ncols) + "), ordering='" + s.ordering + "')";
        });
}

template <typename T>
void register_supernodal_lu(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Wrap = SupernodalLU<T>;
    const std::string name = std::string("SupernodalLU_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const SMat& A, const std::string& ordering,
                            double threshold, std::size_t max_super, bool scale,
                            double pivot_perturb,
                            std::optional<std::string> accumulator) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("SupernodalLU: matrix must be square");
            const AccKind acc = parse_sparse_acc<T>(accumulator);
            auto sym = snlu_symbolic(A, ordering);
            fact::supernodal_lu_factor<T> num;
            {
                nogil guard;
                num = with_acc<T>(acc, [&]<typename Acc>() {
                    return fact::supernodal_lu_numeric<T, mtl::mat::parameters<>, Acc>(
                        A, sym, static_cast<T>(threshold), max_super, scale,
                        static_cast<T>(pivot_perturb));
                });
            }
            new (self) Wrap{std::move(num), ordering, threshold, max_super, scale,
                            pivot_perturb, A.num_rows(),
                            accumulator.value_or("default")};
        }, "A"_a, "ordering"_a = "colamd", "threshold"_a = 1.0, "max_super"_a = 64,
           "scale"_a = false, "pivot_perturb"_a = 0.0, "accumulator"_a = nb::none(),
           "Supernodal LU: columns are grouped into supernodes and applied as "
           "dense block updates, with threshold partial pivoting. `scale=True` "
           "row-equilibrates first, which matters most in low precision.")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("ordering", [](const Wrap& s) { return s.ordering; })
        .def_prop_ro("nsuper", [](const Wrap& s) { return s.num.nsuper(); },
                     "Supernodes formed; fewer, larger supernodes mean more of "
                     "the work runs as dense block updates")
        .def_prop_ro("nnz", [](const Wrap& s) {
            return s.num.factorL().nnz() + s.num.factorU().nnz();
        }, "Nonzeros in L plus U")
        .def_prop_ro("num_perturbed", [](const Wrap& s) { return s.num.num_perturbed; })
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def("solve", [](const Wrap& s, nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::supernodal_lu_factor<T>, T>(
                s.num, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b using the stored factors")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            require_default_acc_for_refactor(s.accumulator,
                                             "supernodal_lu_refactor");
            auto fresh = fact::supernodal_lu_refactor(A, s.num);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute the numeric factors for a matrix with the SAME sparsity "
           "pattern, reusing the ordering, supernode partition and pivot sequence")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", nsuper=" + std::to_string(s.num.nsuper()) + ")";
        });
}

template <typename T>
void register_supernodal_ldlt(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Wrap = SupernodalLDLT<T>;
    const std::string name = std::string("SupernodalLDLT_") + type_suffix<T>();

    nb::class_<Wrap>(m, name.c_str())
        .def("__init__", [](Wrap* self, const SMat& A, const std::string& ordering,
                            std::optional<std::string> accumulator) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("SupernodalLDLT: matrix must be square");
            const AccKind acc = parse_sparse_acc<T>(accumulator);
            auto sym = snldlt_symbolic(A, ordering);
            fact::supernodal_ldlt_factor<T> num;
            {
                nogil guard;
                num = with_acc<T>(acc, [&]<typename Acc>() {
                    return fact::supernodal_ldlt_numeric<T, mtl::mat::parameters<>, Acc>(A, sym);
                });
            }
            new (self) Wrap{std::move(sym), std::move(num), ordering, A.num_rows(),
                            accumulator.value_or("default")};
        }, "A"_a, "ordering"_a = "amd", "accumulator"_a = nb::none(),
           "Supernodal LDL^T of a symmetric matrix, applying each supernode as a "
           "dense block update")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("ordering", [](const Wrap& s) { return s.ordering; })
        .def_prop_ro("nnz", [](const Wrap& s) { return s.num.factorL().nnz(); })
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def("solve", [](const Wrap& s, nb::ndarray<double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::supernodal_ldlt_factor<T>, T>(
                s.num, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b via the supernodal LDL^T factor")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            // Unlike the LU refactors, this one re-runs the full numeric
            // factorization, so the accumulator policy carries through.
            auto fresh = with_acc<T>(parse_sparse_acc<T>(
                s.accumulator == "default" ? std::optional<std::string>{}
                                           : std::optional<std::string>{s.accumulator}),
                [&]<typename Acc>() {
                    return fact::supernodal_ldlt_numeric<T, mtl::mat::parameters<>, Acc>(A, s.sym);
                });
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute the numeric factor for a matrix with the SAME sparsity "
           "pattern, reusing the ordering and supernodal symbolic analysis")
        .def("__repr__", [name](const Wrap& s) {
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) + ")";
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
    register_cholesky<float>(m);
    register_cholesky<double>(m);
    register_ldlt<float>(m);
    register_ldlt<double>(m);
    register_qr<float>(m);
    register_qr<double>(m);
    register_supernodal_lu<float>(m);
    register_supernodal_lu<double>(m);
    register_supernodal_ldlt<float>(m);
    register_supernodal_ldlt<double>(m);
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
    // QR is intentionally absent: it solves a least-squares problem, not the
    // square system iterative_refine corrects.
    mx.def("iterative_refine", &sparse_refine_through<SparseCholesky<double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<SparseCholesky<float>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<SparseLDLT<double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<SparseLDLT<float>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<SupernodalLU<double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<SupernodalLU<float>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<SupernodalLDLT<double>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
    mx.def("iterative_refine", &sparse_refine_through<SupernodalLDLT<float>>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
}
