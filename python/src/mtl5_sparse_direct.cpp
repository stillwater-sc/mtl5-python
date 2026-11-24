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
#include <mtl/sparse/ordering/nested_dissection.hpp>
#include <mtl/sparse/ordering/rcm.hpp>

#include <cstddef>
#include <mutex>
#include <optional>
#include <shared_mutex>
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
    "'nd' (nested dissection on A+A^T; large 2-D and 3-D meshes), "
    "'natural' (identity, no fill reduction)";

/// Fill-reducing ordering as a standalone permutation.
template <typename T>
std::vector<std::size_t> permutation_for(const mtl::mat::compressed2D<T>& A,
                                         const std::string& name) {
    if (name == "amd")     return ord::amd{}(A);
    if (name == "colamd")  return ord::colamd{}(A);
    if (name == "rcm")     return ord::rcm{}(A);
    if (name == "nd")      return ord::nested_dissection{}(A);
    if (name == "natural") {
        std::vector<std::size_t> p(A.num_rows());
        for (std::size_t i = 0; i < p.size(); ++i) p[i] = i;
        return p;
    }
    throw std::invalid_argument("unknown ordering '" + name + "'; " + kOrderingHelp);
}

// ---------------------------------------------------------------------------
// Exposing the factors
//
// Every factorization below keeps its factors in MTL5's internal CSC layout.
// They are handed to Python as copies in the CRS SparseMatrix_* type the rest
// of the module speaks, so the result works with mtl5.io.spy, to_scipy and
// matvec like any other sparse matrix. The copy is O(nnz) on every access;
// that is deliberate, since a view would tie a Python object's lifetime to a
// factor that refactor() overwrites in place.
//
// Permutations follow MTL5's convention, p[new] = old, so a permutation
// indexes the ORIGINAL matrix: for an LU, A[row_perm][:, col_perm] == L @ U.
// That is the inverse of scipy's SuperLU.perm_r / perm_c, which is why the
// scipy names are not reused here.
// ---------------------------------------------------------------------------

/// A permutation as a NumPy int64 array.
inline nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>
perm_array(const std::vector<std::size_t>& p) {
    const std::size_t n = p.size();
    int64_t* buf = new int64_t[n];
    for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<int64_t>(p[i]);
    std::size_t shape[1] = { n };
    nb::capsule owner(buf, [](void* q) noexcept { delete[] static_cast<int64_t*>(q); });
    return nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(buf, 1, shape, owner);
}

/// A vector of factor values as a NumPy float64 array.
template <typename T>
nb::ndarray<nb::numpy, double, nb::ndim<1>> f64_array(const std::vector<T>& v) {
    const std::size_t n = v.size();
    double* buf = new double[n];
    for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<double>(v[i]);
    std::size_t shape[1] = { n };
    nb::capsule owner(buf, [](void* q) noexcept { delete[] static_cast<double*>(q); });
    return nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
}

/// Copy a CSC factor into a CRS SparseMatrix.
///
/// The LDL^T factors store L as unit lower triangular WITHOUT its diagonal --
/// the 1s are implicit in the solves. `add_unit_diagonal` writes them back, so
/// every L this module returns has its diagonal explicit and reconstructs with
/// plain products (L @ D @ L.T), the same way the LU and Cholesky L do. Each
/// column's stored rows are strictly below the diagonal and sorted, so the 1
/// goes first.
template <typename T, typename S>
mtl::mat::compressed2D<T> factor_to_crs(const mtl::sparse::util::csc_matrix<T, S>& M,
                                        bool add_unit_diagonal = false) {
    if (!add_unit_diagonal) return mtl::sparse::util::csc_to_crs(M);
    mtl::sparse::util::csc_matrix<T, S> W;
    W.nrows = M.nrows;
    W.ncols = M.ncols;
    W.col_ptr.resize(M.ncols + 1);
    W.row_ind.reserve(M.nnz() + M.ncols);
    W.values.reserve(M.nnz() + M.ncols);
    W.col_ptr[0] = 0;
    for (S c = 0; c < M.ncols; ++c) {
        W.row_ind.push_back(c);
        W.values.push_back(T{1});
        for (S k = M.col_ptr[c]; k < M.col_ptr[c + 1]; ++k) {
            if (M.row_ind[k] <= c)
                throw std::logic_error(
                    "factor_to_crs: expected a strictly lower factor with an "
                    "implicit unit diagonal");
            W.row_ind.push_back(M.row_ind[k]);
            W.values.push_back(M.values[k]);
        }
        W.col_ptr[c + 1] = static_cast<S>(W.row_ind.size());
    }
    return mtl::sparse::util::csc_to_crs(W);
}

/// Symbolic analysis under a named ordering.
template <typename T>
fact::lu_symbolic symbolic_for(const mtl::mat::compressed2D<T>& A,
                               const std::string& name) {
    if (name == "amd")     return fact::sparse_lu_symbolic(A, ord::amd{});
    if (name == "colamd")  return fact::sparse_lu_symbolic(A, ord::colamd{});
    if (name == "rcm")     return fact::sparse_lu_symbolic(A, ord::rcm{});
    if (name == "nd")      return fact::sparse_lu_symbolic(A, ord::nested_dissection{});
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
// The quire is deliberately absent. Universal 5.8.0 does define quire<float>
// and quire<double>, but the factorizations type a dense workspace of n
// accumulators, and at 4226 bits each a quire<double> workspace costs roughly
// 530 bytes per row. That needs its own design, so it is refused by name here
// rather than offered at that cost.
template <typename T>
AccKind parse_sparse_acc(const std::optional<std::string>& spec) {
    if (spec && *spec == "quire")
        throw std::invalid_argument(
            "accumulator='quire' is not available for the sparse "
            "factorizations: their numeric workspace holds one accumulator per "
            "row, and a quire<double> is 4226 bits wide. Use 'f64' or 'fma64' "
            "for a wider accumulator.");
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

/// The canonical name for a parsed accumulator. Storing this rather than the
/// caller's spelling keeps the aliases consistent: 'none' and 'default' are the
/// same policy, as are 'fma' and 'fma64', and refactor's default-only check
/// compares against one spelling.
inline const char* acc_name(AccKind kind) {
    switch (kind) {
        case AccKind::Default: return "default";
        case AccKind::F32:     return "f32";
        case AccKind::F64:     return "f64";
        case AccKind::FMA32:   return "fma32";
        case AccKind::FMA64:   return "fma64";
        case AccKind::Quire:   return "quire";
    }
    return "default";
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

/// The diagnostics every pivoting factorization's repr ends with. A nonzero
/// num_perturbed means the factor is of a perturbed matrix, and the
/// accumulator is otherwise invisible; both belong where people look first.
inline std::string repr_diagnostics(std::size_t num_perturbed, const std::string& acc) {
    return ", num_perturbed=" + std::to_string(num_perturbed) + ", accumulator='" + acc + "'";
}

// ---------------------------------------------------------------------------
// FactorLock: every wrapper below carries a std::shared_mutex `mtx` guarding
// `num`.
//
// refactor() replaces or rewrites `num` with the GIL released, so the GIL alone
// cannot order it against anything: solve() and iterative_refine read `num`
// with the GIL released too, and the L/U/perm getters read it with the GIL
// held. Without the lock a getter could copy a factor halfway through a
// refactor -- a torn mix of old and new values, or a read of storage the move
// assignment just freed. The rule:
//
//   - refactor() takes `mtx` exclusive;
//   - every read of `num` (getters, solve, iterative_refine) takes it shared.
//
// Lock order is always GIL-first: the paths that release the GIL release it
// BEFORE taking `mtx` and retake it only after dropping `mtx` (the lock is
// declared after the nogil guard, so it is destroyed first). A getter blocks on
// `mtx` while holding the GIL, which is safe because no holder of `mtx` ever
// waits for the GIL.
// ---------------------------------------------------------------------------
template <typename T>
struct SparseLU {
    fact::lu_numeric<T> num;
    std::string ordering;
    std::size_t n;
    std::string accumulator;

    const fact::lu_numeric<T>& factor() const { return num; }
    mutable std::shared_mutex mtx;  // guards num; see FactorLock
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
    mutable std::shared_mutex mtx;  // guards num; see FactorLock
};

/// Shared solve body: run the factorization's triangular solves on a copy of b.
template <typename Fac, typename T>
VectorView<T> solve_with(const Fac& fac, std::shared_mutex& mtx, std::size_t n,
                         const double* bp, std::size_t blen) {
    if (blen != n)
        throw std::invalid_argument(
            "solve: RHS length " + std::to_string(blen) +
            " does not match factor size " + std::to_string(n));
    mtl::vec::dense_vector<T> x(n);
    {
        nogil guard;
        std::shared_lock lock(mtx);   // after the GIL is released: see FactorLock
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
                    return fact::sparse_lu_numeric<Acc>(
                        A, sym, static_cast<T>(threshold),
                        static_cast<T>(pivot_perturb));
                });
            }
            new (self) Wrap{std::move(num), ordering, A.num_rows(), acc_name(acc)};
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
            std::shared_lock lock(s.mtx);
            return s.num.factorL().nnz() + s.num.factorU().nnz();
        }, "Nonzeros in L plus U — the fill the ordering produced")
        .def_prop_ro("num_perturbed", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.num_perturbed;
        }, "Pivots replaced by perturbation; 0 means a clean factor")
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def_prop_ro("L", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.factorL());
        }, "Unit lower triangular factor, diagonal stored, as a copy. "
           "A[row_perm][:, col_perm] == L @ U. Exact up to rounding only "
           "when num_perturbed == 0: a perturbed pivot (pivot_perturb > 0) "
           "makes L @ U reconstruct the perturbed matrix, not A.")
        .def_prop_ro("U", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.factorU());
        }, "Upper triangular factor, as a copy. "
           "A[row_perm][:, col_perm] == L @ U. Exact up to rounding only "
           "when num_perturbed == 0: a perturbed pivot (pivot_perturb > 0) "
           "makes L @ U reconstruct the perturbed matrix, not A.")
        .def_prop_ro("row_perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.row_perm);
        },
                     nb::rv_policy::move,
                     "Row permutation from threshold pivoting, p[new] = old")
        .def_prop_ro("col_perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.symbolic.col_perm);
        }, nb::rv_policy::move,
           "Column permutation from the fill-reducing ordering, q[new] = old")
        .def("solve", [](const Wrap& s,
                         nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                                     nb::device::cpu> b) {
            return solve_with<fact::lu_numeric<T>, T>(s.num, s.mtx, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b using the stored factors")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            require_default_acc_for_refactor(s.accumulator, "sparse_lu_refactor");
            nogil guard;
            std::unique_lock lock(s.mtx);  // exclusive: replaces s.num
            // Numeric-only: reuses the ordering, symbolic structure and pivot
            // sequence. The IN-PLACE form (MTL5 #310) is the one to call: the
            // value-returning sparse_lu_refactor is now a wrapper that copies
            // s.num first, and the copy is the cost -- duplicating the two
            // nnz-length solve schedules costs about what rebuilding them
            // costs, so the copy-then-move we used to do here paid the O(nnz)
            // price the in-place path exists to avoid (26.3 -> 20.3 ms per
            // refactor upstream, on a 390K-nonzero factor).
            //
            // The "assign only after it succeeds" guarantee the copy gave us is
            // not lost: sparse_lu_refactor_in_place computes into scratch and
            // installs only after the last column succeeds, so a zero pivot
            // throws with s.num untouched and still solvable with its previous
            // values. That is a documented strong guarantee, not an accident.
            fact::sparse_lu_refactor_in_place(A, s.num);
        }, "A"_a,
           "Recompute the numeric factors for a matrix with the SAME sparsity "
           "pattern, reusing the ordering, symbolic analysis and pivot sequence")
        .def("__repr__", [name](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", ordering='" + s.ordering + "', nnz=" +
                   std::to_string(s.num.factorL().nnz() + s.num.factorU().nnz()) +
                   repr_diagnostics(s.num.num_perturbed, s.accumulator) + ")";
        });
}

// KLU has no single pair of triangular factors. Its factorization is a block
// upper triangular form B = P*(R*A)*Q with an independent LU of each diagonal
// block; the off-diagonal blocks are never factored, only used in the block
// back-substitution. Folding that into one global L and U would need terms
// like L_i^{-1} B_ij, which fill in, and would misrepresent where BTF saves
// its work. So the structure is exposed as it is: the BTF permutations, the
// block boundaries, the row scaling, B itself, and each block's own factors.
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
                    return fact::native_klu_factor<Acc>(
                        A, static_cast<T>(threshold), scale,
                        static_cast<T>(pivot_perturb));
                });
            }
            new (self) Wrap{std::move(num), A.num_rows(), acc_name(acc)};
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
        .def_prop_ro("nblocks", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.nblocks();
        }, "Diagonal blocks found by the block triangular form; more "
           "blocks means more reducible structure to exploit")
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def_prop_ro("nnz", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            // Summed over the diagonal blocks: the off-diagonal blocks are not
            // factored, which is exactly where BTF saves work.
            std::size_t total = 0;
            for (const auto& blk : s.num.block_numeric)
                total += blk.factorL().nnz() + blk.factorU().nnz();
            return total;
        }, "Nonzeros in L plus U, summed over the diagonal blocks")
        .def_prop_ro("num_perturbed", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.num_perturbed;
        }, "Pivots replaced by perturbation across all blocks")
        .def_prop_ro("row_perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.btf.row_perm);
        }, nb::rv_policy::move,
           "Row permutation of the block triangular form, p[new] = old. "
           "B == (diag(row_scale) @ A)[row_perm][:, col_perm].")
        .def_prop_ro("col_perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.btf.col_perm);
        }, nb::rv_policy::move,
           "Column permutation of the block triangular form, q[new] = old")
        .def_prop_ro("block_starts", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.btf.blocks);
        }, nb::rv_policy::move,
           "Block boundaries, length nblocks + 1: block i spans rows and "
           "columns block_starts[i]:block_starts[i+1] of B")
        .def_prop_ro("row_scale", [](const Wrap& s)
                         -> std::optional<nb::ndarray<nb::numpy, double, nb::ndim<1>>> {
            std::shared_lock lock(s.mtx);
            if (s.num.row_scale.empty()) return std::nullopt;
            return f64_array(s.num.row_scale);
        }, nb::rv_policy::move,
           "Row equilibration factors indexed by ORIGINAL row, or None when "
           "the matrix was factored unscaled (scale=False)")
        .def_prop_ro("B", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return SMat(s.num.B);
        }, "The permuted, scaled matrix P*(R*A)*Q, as a copy. It is block "
           "upper triangular: only its diagonal blocks are factored, and the "
           "blocks above them are used as they are in the block "
           "back-substitution.")
        .def("block", [](const Wrap& s, std::ptrdiff_t i) {
            std::shared_lock lock(s.mtx);
            const auto nb_ = static_cast<std::ptrdiff_t>(s.num.nblocks());
            if (i < 0) i += nb_;
            if (i < 0 || i >= nb_)
                throw nb::index_error("block index out of range");
            const auto& f = s.num.block_numeric[static_cast<std::size_t>(i)];
            nb::dict d;
            d["start"] = s.num.btf.blocks[static_cast<std::size_t>(i)];
            d["stop"] = s.num.btf.blocks[static_cast<std::size_t>(i) + 1];
            d["L"] = factor_to_crs(f.factorL());
            d["U"] = factor_to_crs(f.factorU());
            d["row_perm"] = perm_array(f.row_perm);
            d["col_perm"] = perm_array(f.symbolic.col_perm);
            return d;
        }, "i"_a,
           "The factors of diagonal block i, as copies: a dict with 'start' "
           "and 'stop' (its span in B), and 'L', 'U', 'row_perm' and "
           "'col_perm' in the block's own indices. With Bi = "
           "B[start:stop, start:stop], Bi[row_perm][:, col_perm] == L @ U, "
           "exactly up to rounding when num_perturbed == 0. Negative i counts "
           "from the end.")
        .def("solve", [](const Wrap& s,
                         nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                                     nb::device::cpu> b) {
            return solve_with<fact::klu_numeric<T>, T>(s.num, s.mtx, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b by block back-substitution")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            std::unique_lock lock(s.mtx);  // exclusive: replaces s.num
            require_default_acc_for_refactor(s.accumulator, "native_klu_refactor");
            auto fresh = fact::native_klu_refactor(A, s.num);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute the numeric factors for a matrix with the SAME sparsity "
           "pattern, reusing the BTF permutation, per-block ordering and pivot "
           "sequence — the SPICE transient path")
        .def("__repr__", [name](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            std::size_t nnz = 0;
            for (const auto& blk : s.num.block_numeric)
                nnz += blk.factorL().nnz() + blk.factorU().nnz();
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", nblocks=" + std::to_string(s.num.nblocks()) +
                   ", nnz=" + std::to_string(nnz) +
                   repr_diagnostics(s.num.num_perturbed, s.accumulator) + ")";
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
    if (name == "nd")      return fact::fn(A, ord::nested_dissection{});      \
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
    if (name == "nd")      return fact::supernodal_lu_symbolic_analyze(A, ord::nested_dissection{});
    throw std::invalid_argument(
        "supernodal LU ordering must be 'amd', 'colamd', 'rcm' or 'nd' (got '" + name + "')");
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
    mutable std::shared_mutex mtx;  // guards num; see FactorLock
};

template <typename T>
struct SparseLDLT {
    fact::ldlt_symbolic sym;
    fact::ldlt_numeric<T> num;
    std::string ordering;
    std::size_t n;
    const fact::ldlt_numeric<T>& factor() const { return num; }
    mutable std::shared_mutex mtx;  // guards num; see FactorLock
};

template <typename T>
struct SparseQR {
    fact::qr_symbolic sym;
    fact::qr_numeric<T> num;
    std::string ordering;
    std::size_t nrows, ncols;
    mutable std::shared_mutex mtx;  // guards num; see FactorLock
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
    mutable std::shared_mutex mtx;  // guards num; see FactorLock
};

template <typename T>
struct SupernodalLDLT {
    fact::supernodal_symbolic sym;
    fact::supernodal_ldlt_factor<T> num;
    std::string ordering;
    std::size_t n;
    std::string accumulator;
    const fact::supernodal_ldlt_factor<T>& factor() const { return num; }
    mutable std::shared_mutex mtx;  // guards num; see FactorLock
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
        .def_prop_ro("nnz", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.factor().nnz();
        }, "Nonzeros in the lower factor L")
        .def_prop_ro("L", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.factor());
        }, "Lower triangular Cholesky factor, as a copy. "
           "A[perm][:, perm] == L @ L.T.")
        .def_prop_ro("perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.symbolic.perm);
        },
                     nb::rv_policy::move,
                     "Symmetric fill-reducing permutation, p[new] = old")
        .def("solve", [](const Wrap& s, nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::cholesky_numeric<T>, T>(s.num, s.mtx, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b via the Cholesky factor")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            std::unique_lock lock(s.mtx);  // exclusive: replaces s.num
            auto fresh = fact::sparse_cholesky_numeric(A, s.sym);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute L for a matrix with the SAME sparsity pattern, reusing the "
           "ordering and symbolic analysis (Cholesky does not pivot, so that is "
           "the entire analysis)")
        .def("__repr__", [name](const Wrap& s) {
            std::shared_lock lock(s.mtx);
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
        .def_prop_ro("nnz", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.factorL().nnz();
        }, "Nonzeros in the unit lower factor L, excluding its implicit "
           "unit diagonal (so L.nnz == nnz + n)")
        .def_prop_ro("L", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.factorL(), /*add_unit_diagonal=*/true);
        }, "Unit lower triangular factor, as a copy, with its unit diagonal made "
           "explicit. A[perm][:, perm] == L @ diag(diagonal()) @ L.T.")
        .def_prop_ro("perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.symbolic.perm);
        },
                     nb::rv_policy::move,
                     "Symmetric fill-reducing permutation, p[new] = old")
        .def("diagonal", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return f64_array(s.num.diagonal());
        }, "The diagonal D. Its signs give the inertia: a negative entry means A "
             "is indefinite.")
        .def("solve", [](const Wrap& s, nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::ldlt_numeric<T>, T>(s.num, s.mtx, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b via the LDL^T factor")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            std::unique_lock lock(s.mtx);  // exclusive: replaces s.num
            auto fresh = fact::sparse_ldlt_numeric(A, s.sym);
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute L and D for a matrix with the SAME sparsity pattern, "
           "reusing the ordering and symbolic analysis")
        .def("__repr__", [name](const Wrap& s) {
            std::shared_lock lock(s.mtx);
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
        .def_prop_ro("nnz", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.R.nnz();
        }, "Nonzeros in the upper factor R")
        .def_prop_ro("R", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.R);
        }, "Upper triangular factor R (ncols x ncols), as a copy. "
           "A[:, col_perm] == Q @ R; Q is held implicitly as Householder "
           "vectors and is not exposed, but R.T @ R == "
           "A[:, col_perm].T @ A[:, col_perm].")
        .def_prop_ro("col_perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.symbolic.col_perm);
        }, nb::rv_policy::move,
           "Column permutation from the fill-reducing ordering, q[new] = old")
        .def("solve", [](const Wrap& s, nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            if (b.shape(0) != s.nrows)
                throw std::invalid_argument(
                    "solve: RHS length " + std::to_string(b.shape(0)) +
                    " does not match the row count " + std::to_string(s.nrows));
            const double* bp = b.data();
            mtl::vec::dense_vector<T> x(s.ncols);
            {
                nogil guard;
                std::shared_lock lock(s.mtx);
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
            std::unique_lock lock(s.mtx);  // exclusive: replaces s.num
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
                    return fact::supernodal_lu_numeric<Acc>(
                        A, sym, static_cast<T>(threshold), max_super, scale,
                        static_cast<T>(pivot_perturb));
                });
            }
            new (self) Wrap{std::move(num), ordering, threshold, max_super, scale,
                            pivot_perturb, A.num_rows(), acc_name(acc)};
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
        .def_prop_ro("nsuper", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.nsuper();
        }, "Supernodes formed; fewer, larger supernodes mean more of "
           "the work runs as dense block updates")
        .def_prop_ro("nnz", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.factorL().nnz() + s.num.factorU().nnz();
        }, "Nonzeros in L plus U")
        .def_prop_ro("L", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.factorL());
        }, "Unit lower triangular factor, diagonal stored, as a copy. "
           "With scale=True, (diag(row_scale) @ A)[row_perm][:, col_perm] "
           "== L @ U. With scale=False, row_scale is None and the "
           "identity is A[row_perm][:, col_perm] == L @ U. Either holds up "
           "to rounding only when num_perturbed == 0: a perturbed pivot makes "
           "L @ U reconstruct the perturbed matrix.")
        .def_prop_ro("U", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.factorU());
        }, "Upper triangular factor, as a copy. See L for the identity.")
        .def_prop_ro("row_perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.row_perm);
        },
                     nb::rv_policy::move,
                     "Row permutation from threshold pivoting, p[new] = old")
        .def_prop_ro("col_perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.symbolic.col_perm);
        }, nb::rv_policy::move,
           "Column permutation, q[new] = old: the fill-reducing ordering "
           "followed by the column elimination tree postorder")
        .def_prop_ro("row_scale", [](const Wrap& s)
                         -> std::optional<nb::ndarray<nb::numpy, double, nb::ndim<1>>> {
            std::shared_lock lock(s.mtx);
            if (s.num.row_scale.empty()) return std::nullopt;
            return f64_array(s.num.row_scale);
        }, nb::rv_policy::move,
           "Row equilibration factors indexed by ORIGINAL row. None when the "
           "matrix was factored unscaled (scale=False): reconstruct from A "
           "itself then, with no row scaling")
        .def_prop_ro("num_perturbed", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.num_perturbed;
        })
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def("solve", [](const Wrap& s, nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::supernodal_lu_factor<T>, T>(
                s.num, s.mtx, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b using the stored factors")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            std::unique_lock lock(s.mtx);  // exclusive: replaces s.num
            require_default_acc_for_refactor(s.accumulator,
                                             "supernodal_lu_refactor");
            // In-place for the same reason as sparse_lu above (MTL5 #310):
            // the value-returning form copies s.num, and that copy is the cost.
            // Strong guarantee preserved -- a zero pivot leaves s.num intact.
            fact::supernodal_lu_refactor_in_place(A, s.num);
        }, "A"_a,
           "Recompute the numeric factors for a matrix with the SAME sparsity "
           "pattern, reusing the ordering, supernode partition and pivot sequence")
        .def("__repr__", [name](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", ordering='" + s.ordering + "', nnz=" +
                   std::to_string(s.num.factorL().nnz() + s.num.factorU().nnz()) +
                   ", nsuper=" + std::to_string(s.num.nsuper()) +
                   repr_diagnostics(s.num.num_perturbed, s.accumulator) + ")";
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
                    return fact::supernodal_ldlt_numeric<Acc>(A, sym);
                });
            }
            new (self) Wrap{std::move(sym), std::move(num), ordering, A.num_rows(),
                            acc_name(acc)};
        }, "A"_a, "ordering"_a = "amd", "accumulator"_a = nb::none(),
           "Supernodal LDL^T of a symmetric matrix, applying each supernode as a "
           "dense block update")
        .def_prop_ro("n", [](const Wrap& s) { return s.n; })
        .def_prop_ro("shape", [](const Wrap& s) {
            return std::pair<std::size_t, std::size_t>(s.n, s.n);
        })
        .def_prop_ro("dtype", [](const Wrap&) { return type_suffix<T>(); })
        .def_prop_ro("ordering", [](const Wrap& s) { return s.ordering; })
        .def_prop_ro("nnz", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return s.num.factorL().nnz();
        }, "Nonzeros in the unit lower factor L, excluding its implicit "
           "unit diagonal (so L.nnz == nnz + n)")
        .def_prop_ro("accumulator", [](const Wrap& s) { return s.accumulator; },
                     "The accumulator policy this factor was built with")
        .def_prop_ro("L", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return factor_to_crs(s.num.factorL(), /*add_unit_diagonal=*/true);
        }, "Unit lower triangular factor, as a copy, with its unit diagonal made "
           "explicit. A[perm][:, perm] == L @ diag(diagonal()) @ L.T.")
        .def_prop_ro("perm", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return perm_array(s.num.symbolic.sperm);
        },
                     nb::rv_policy::move,
                     "Symmetric permutation, p[new] = old: the fill-reducing "
                     "ordering followed by the elimination tree postorder")
        .def("diagonal", [](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return f64_array(s.num.diagonal());
        }, "The diagonal D. Its signs give the inertia: a negative entry means A "
             "is indefinite.")
        .def("solve", [](const Wrap& s, nb::ndarray<const double, nb::ndim<1>, nb::c_contig,
                                                    nb::device::cpu> b) {
            return solve_with<fact::supernodal_ldlt_factor<T>, T>(
                s.num, s.mtx, s.n, b.data(), b.shape(0));
        }, "b"_a, "Solve A x = b via the supernodal LDL^T factor")
        .def("refactor", [](Wrap& s, const SMat& A) {
            if (A.num_rows() != s.n || A.num_cols() != s.n)
                throw std::invalid_argument(
                    "refactor: matrix dimensions do not match the factorization");
            nogil guard;
            std::unique_lock lock(s.mtx);  // exclusive: replaces s.num
            // Unlike the LU refactors, this one re-runs the full numeric
            // factorization, so the accumulator policy carries through.
            auto fresh = with_acc<T>(parse_sparse_acc<T>(
                s.accumulator == "default" ? std::optional<std::string>{}
                                           : std::optional<std::string>{s.accumulator}),
                [&]<typename Acc>() {
                    return fact::supernodal_ldlt_numeric<Acc>(A, s.sym);
                });
            s.num = std::move(fresh);
        }, "A"_a,
           "Recompute the numeric factor for a matrix with the SAME sparsity "
           "pattern, reusing the ordering and supernodal symbolic analysis")
        .def("__repr__", [name](const Wrap& s) {
            std::shared_lock lock(s.mtx);
            return "mtl5.sparse." + name + "(n=" + std::to_string(s.n) +
                   ", ordering='" + s.ordering + "', nnz=" +
                   std::to_string(s.num.factorL().nnz()) +
                   ", accumulator='" + s.accumulator + "')";
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
        return perm_array(perm);
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
        return std::vector<std::string>{"amd", "colamd", "rcm", "nd", "natural"};
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
