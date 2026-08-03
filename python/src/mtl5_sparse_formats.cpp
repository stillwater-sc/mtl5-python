// mtl5-python -- alternative sparse storage formats: COO and ELL.
//
// CSR (`compressed2D`) is bound in mtl5_module.cpp. This TU adds the two other
// formats MTL5 actually implements.
//
//   COO  `coordinate2D`  a triplet list. The format you build a matrix in when
//                        you do not yet know the pattern, and the one that
//                        accumulates duplicates: entries at the same (r, c) sum
//                        rather than overwrite, matching scipy.sparse.coo_matrix.
//
//   ELL  `ell_matrix`    a fixed-width row-major layout, nrows x max_width, with
//                        padded slots. Regular enough to vectorise over, which
//                        is what makes it the format of interest for accelerator
//                        dispatch -- but it costs nrows*max_width storage
//                        regardless of the actual nnz, so it is only a good idea
//                        when the row widths are near-uniform. `padding_ratio`
//                        reports exactly that, so the decision is measurable
//                        rather than a guess.
//
// **CSC is deliberately absent.** `compressed2D` takes an orientation parameter
// (`tag::col_major`), but it is inert: the inserter, `operator()` and `mult` all
// treat the storage as row-major regardless. Measured on a 2x3 matrix, a
// col_major `compressed2D` builds a `major` array of length 3 (nrows + 1) where
// CSC needs 4 (ncols + 1), and feeding it genuine scipy CSC arrays yields the
// transpose. Binding it would hand callers a matrix that silently disagrees with
// scipy about which way it is oriented. Filed upstream; until it is implemented,
// `mtl5.sparse.from_scipy` converts CSC input to CSR at the boundary, which is
// honest about the cost.
//
// What CSC was actually wanted for -- a native transpose-SpMV so that a
// LinearOperator's rmatvec does not round-trip through scipy -- is provided
// instead by `SparseMatrix.rmatvec`, which uses MTL5's `trans` view. See
// mtl5_module.cpp.

#include "mtl5_types.hpp"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/ell_matrix.hpp>
#include <mtl/operation/mult.hpp>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

using namespace nb::literals;

namespace {

/// Hand a heap buffer to NumPy with a capsule that owns it.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> adopt_1d(T* p, std::size_t n) {
    std::size_t shape[1] = { n };
    nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(p, 1, shape, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> adopt_2d(T* p, std::size_t r, std::size_t c) {
    std::size_t shape[2] = { r, c };
    nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<T*>(q); });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(p, 2, shape, owner);
}

// ---------------------------------------------------------------------------
// COO -- coordinate2D
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_floating_point_v<T>
void register_coo(nb::module_& m) {
    using COO       = mtl::mat::coordinate2D<T>;
    using size_type = typename COO::size_type;
    using VV        = VectorView<T>;
    const std::string name = std::string("SparseMatrixCOO_") + type_suffix<T>();

    nb::class_<COO>(m, name.c_str())
        .def("__init__", [](COO* self, std::size_t nrows, std::size_t ncols,
                            nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> row,
                            nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> col,
                            nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> data) {
            const std::size_t nnz = data.shape(0);
            if (row.shape(0) != nnz || col.shape(0) != nnz)
                throw std::invalid_argument(
                    "row, col and data must all have the same length");

            // Bounds-check before inserting: coordinate2D asserts, which is a
            // no-op in a release build, so an out-of-range index would be
            // written and only surface later as a wrong answer.
            const int64_t* rp = row.data();
            const int64_t* cp = col.data();
            for (std::size_t k = 0; k < nnz; ++k) {
                if (rp[k] < 0 || static_cast<std::size_t>(rp[k]) >= nrows)
                    throw nb::index_error(
                        ("row index " + std::to_string(rp[k]) + " out of range for " +
                         std::to_string(nrows) + " rows").c_str());
                if (cp[k] < 0 || static_cast<std::size_t>(cp[k]) >= ncols)
                    throw nb::index_error(
                        ("column index " + std::to_string(cp[k]) + " out of range for " +
                         std::to_string(ncols) + " columns").c_str());
            }

            new (self) COO(nrows, ncols);
            self->reserve(nnz);
            const T* dp = data.data();
            for (std::size_t k = 0; k < nnz; ++k)
                self->insert(static_cast<size_type>(rp[k]),
                             static_cast<size_type>(cp[k]), dp[k]);
        }, "nrows"_a, "ncols"_a, "row"_a, "col"_a, "data"_a,
           "Construct a COO matrix from scipy-style (row, col, data) triplets")
        .def_prop_ro("num_rows", &COO::num_rows)
        .def_prop_ro("num_cols", &COO::num_cols)
        .def_prop_ro("nnz", &COO::nnz,
                     "Number of stored triplets -- duplicates are counted "
                     "separately until tocsr() sums them")
        .def_prop_ro("dtype", [](const COO&) { return type_suffix<T>(); })
        .def_prop_ro("shape", [](const COO& A) {
            return std::pair<std::size_t, std::size_t>(A.num_rows(), A.num_cols());
        })
        .def_prop_ro("is_sorted", &COO::is_sorted,
                     "Whether the triplets are in row-major order")
        .def("sort", [](COO& A) { nogil guard; A.sort(); },
             "Sort the triplets into row-major order, in place")
        .def("insert", [](COO& A, std::size_t r, std::size_t c, T v) {
            if (r >= A.num_rows() || c >= A.num_cols())
                throw nb::index_error();
            A.insert(static_cast<size_type>(r), static_cast<size_type>(c), v);
        }, "row"_a, "col"_a, "value"_a,
           "Append a triplet. A repeated (row, col) accumulates rather than "
           "overwrites, matching scipy.sparse.coo_matrix.")
        .def("__getitem__", [](const COO& A, std::pair<std::size_t, std::size_t> idx) -> T {
            if (idx.first >= A.num_rows() || idx.second >= A.num_cols())
                throw nb::index_error();
            // O(nnz) -- COO has no index. Fine for a spot check, wrong for a
            // loop; convert to CSR first if you are reading many elements.
            return A(idx.first, idx.second);
        }, "Read one element. O(nnz) -- COO is unindexed, so use tocsr() if you "
           "are reading more than a few.")
        .def("to_coo_arrays", [](const COO& A) {
            const std::size_t nnz = A.nnz();
            int64_t* rp = new int64_t[nnz];
            int64_t* cp = new int64_t[nnz];
            T*       dp = new T[nnz];
            std::size_t k = 0;
            for (const auto& [r, c, v] : A.ref_entries()) {
                rp[k] = static_cast<int64_t>(r);
                cp[k] = static_cast<int64_t>(c);
                dp[k] = v;
                ++k;
            }
            return std::make_tuple(adopt_1d(rp, nnz), adopt_1d(cp, nnz), adopt_1d(dp, nnz));
        }, "Return (row, col, data) as NumPy arrays -- scipy COO layout")
        .def("tocsr", [](const COO& A) {
            nogil guard;
            return A.compress();
        }, "Convert to CSR, summing any duplicate (row, col) entries")
        .def("matvec", [](const COO& A, const VV& x) {
            if (A.num_cols() != x.vec.size())
                throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
            mtl::vec::dense_vector<T> y(A.num_rows());
            {
                nogil guard;
                mtl::mult(A, x.vec, y);
            }
            return VV(std::move(y));
        }, "x"_a, "Sparse matrix-vector product: y = A @ x")
        .def("matvec", [](const COO& A,
                          nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> x_np) {
            if (A.num_cols() != x_np.shape(0))
                throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
            const std::size_t n = x_np.shape(0);
            const T* xp = x_np.data();
            mtl::vec::dense_vector<T> y(A.num_rows());
            {
                nogil guard;
                mtl::vec::dense_vector<T> x(n);
                for (std::size_t i = 0; i < n; ++i) x[i] = xp[i];
                mtl::mult(A, x, y);
            }
            return VV(std::move(y));
        }, "x"_a)
        .def("__repr__", [](const COO& A) {
            std::ostringstream os;
            os << "mtl5.SparseMatrixCOO_" << type_suffix<T>()
               << "(shape=(" << A.num_rows() << ", " << A.num_cols() << ")"
               << ", nnz=" << A.nnz()
               << (A.is_sorted() ? ", sorted" : "") << ")";
            return os.str();
        });
}

// ---------------------------------------------------------------------------
// ELL -- ell_matrix
//
// ell_matrix has no element setter: the (nrows, ncols, max_width) constructor
// produces an all-padding matrix with no way to fill it. So the CSR constructor
// is the only usable entry point, and that is the only one bound.
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_floating_point_v<T>
void register_ell(nb::module_& m) {
    using ELL       = mtl::mat::ell_matrix<T>;
    using CSR       = mtl::mat::compressed2D<T>;
    using size_type = typename ELL::size_type;
    using VV        = VectorView<T>;
    const std::string name = std::string("SparseMatrixELL_") + type_suffix<T>();

    // Count the occupied slots; ELL stores nrows * max_width of them and pads
    // the rest, so this is not derivable from the shape.
    auto count_nnz = [](const ELL& A) {
        std::size_t n = 0;
        for (auto idx : A.ref_indices())
            if (idx != ELL::invalid) ++n;
        return n;
    };

    nb::class_<ELL>(m, name.c_str())
        .def(nb::init<const CSR&>(), "A"_a,
             "Build from a CSR matrix. max_width is taken from the widest row.")
        .def_prop_ro("num_rows", &ELL::num_rows)
        .def_prop_ro("num_cols", &ELL::num_cols)
        .def_prop_ro("max_width", &ELL::max_width,
                     "Slots stored per row -- the widest row's nnz")
        .def_prop_ro("dtype", [](const ELL&) { return type_suffix<T>(); })
        .def_prop_ro("shape", [](const ELL& A) {
            return std::pair<std::size_t, std::size_t>(A.num_rows(), A.num_cols());
        })
        .def_prop_ro("nnz", [count_nnz](const ELL& A) {
            nogil guard;
            return count_nnz(A);
        }, "Occupied slots, excluding padding")
        .def_prop_ro("stored", [](const ELL& A) {
            return A.num_rows() * A.max_width();
        }, "Total slots held, padding included -- what ELL actually costs")
        .def_prop_ro("padding_ratio", [count_nnz](const ELL& A) -> double {
            const std::size_t stored = A.num_rows() * A.max_width();
            if (stored == 0) return 0.0;
            std::size_t nnz;
            {
                nogil guard;
                nnz = count_nnz(A);
            }
            return 1.0 - static_cast<double>(nnz) / static_cast<double>(stored);
        }, "Fraction of stored slots that are padding. Near 0 means the rows "
           "are uniform and ELL is a good fit; near 1 means one long row is "
           "forcing everyone else to carry empty slots, and CSR is better.")
        .def("__getitem__", [](const ELL& A, std::pair<std::size_t, std::size_t> idx) -> T {
            if (idx.first >= A.num_rows() || idx.second >= A.num_cols())
                throw nb::index_error();
            return A(idx.first, idx.second);
        }, "Read one element. O(max_width) -- a scan of the row's slots.")
        .def("to_ell_arrays", [](const ELL& A) {
            // Two nrows x max_width arrays. Padding is -1 in indices, matching
            // the sentinel convention a caller would expect from NumPy, rather
            // than ell_matrix's size_type(-1) which would come through as a
            // huge positive number.
            const std::size_t r = A.num_rows(), w = A.max_width();
            const std::size_t total = r * w;
            int64_t* ip = new int64_t[total];
            T*       dp = new T[total];
            const auto& src_i = A.ref_indices();
            const auto& src_d = A.ref_data();
            for (std::size_t k = 0; k < total; ++k) {
                ip[k] = (src_i[k] == ELL::invalid) ? int64_t(-1)
                                                   : static_cast<int64_t>(src_i[k]);
                dp[k] = src_d[k];
            }
            return std::make_tuple(adopt_2d(ip, r, w), adopt_2d(dp, r, w));
        }, "Return (indices, data), each nrows x max_width. Padded slots are -1 "
           "in indices and zero in data.")
        .def("matvec", [](const ELL& A, const VV& x) {
            if (A.num_cols() != x.vec.size())
                throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
            mtl::vec::dense_vector<T> y(A.num_rows());
            {
                nogil guard;
                mtl::mult(A, x.vec, y);
            }
            return VV(std::move(y));
        }, "x"_a, "Sparse matrix-vector product: y = A @ x")
        .def("matvec", [](const ELL& A,
                          nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> x_np) {
            if (A.num_cols() != x_np.shape(0))
                throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
            const std::size_t n = x_np.shape(0);
            const T* xp = x_np.data();
            mtl::vec::dense_vector<T> y(A.num_rows());
            {
                nogil guard;
                mtl::vec::dense_vector<T> x(n);
                for (std::size_t i = 0; i < n; ++i) x[i] = xp[i];
                mtl::mult(A, x, y);
            }
            return VV(std::move(y));
        }, "x"_a)
        .def("__repr__", [count_nnz](const ELL& A) {
            std::ostringstream os;
            os << "mtl5.SparseMatrixELL_" << type_suffix<T>()
               << "(shape=(" << A.num_rows() << ", " << A.num_cols() << ")"
               << ", max_width=" << A.max_width()
               << ", nnz=" << count_nnz(A) << ")";
            return os.str();
        });

    (void)sizeof(size_type);
}

}  // namespace

// ===========================================================================
void register_sparse_formats(nb::module_& m) {
    register_coo<float>(m);
    register_coo<double>(m);
    register_ell<float>(m);
    register_ell<double>(m);
}
