#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/lu.hpp>

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <sstream>

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Helper: NumPy array (float64, 1-D, C-contiguous) → MTL5 dense_vector view
// ---------------------------------------------------------------------------
static mtl::vec::dense_vector<double>
numpy_to_vector(nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
    return mtl::vec::dense_vector<double>(a.shape(0), a.data());
}

// ---------------------------------------------------------------------------
// Helper: NumPy array (float64, 2-D, C-contiguous) → MTL5 dense2D view
// ---------------------------------------------------------------------------
static mtl::mat::dense2D<double>
numpy_to_matrix(nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
    return mtl::mat::dense2D<double>(a.shape(0), a.shape(1),
                                     const_cast<double*>(a.data()));
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
NB_MODULE(_core, m) {
    m.doc() = "MTL5 Python bindings — nanobind core module";

    // -- version info -------------------------------------------------------
    m.attr("__version__") = "0.1.0";

    // -- vector wrapper -----------------------------------------------------
    m.def("vector", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        // Create an owning MTL5 vector (copy from NumPy)
        auto v = mtl::vec::dense_vector<double>(a.shape(0));
        const double* src = a.data();
        for (std::size_t i = 0; i < a.shape(0); ++i)
            v[i] = src[i];
        return v;
    }, "a"_a, "Create an MTL5 dense_vector<double> from a 1-D NumPy array");

    // -- matrix wrapper -----------------------------------------------------
    m.def("matrix", [](nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        // Create an owning MTL5 matrix (copy from NumPy)
        auto M = mtl::mat::dense2D<double>(a.shape(0), a.shape(1));
        const double* src = a.data();
        for (std::size_t r = 0; r < a.shape(0); ++r)
            for (std::size_t c = 0; c < a.shape(1); ++c)
                M(r, c) = src[r * a.shape(1) + c];
        return M;
    }, "a"_a, "Create an MTL5 dense2D<double> from a 2-D NumPy array");

    // -- dense_vector<double> bindings --------------------------------------
    nb::class_<mtl::vec::dense_vector<double>>(m, "DenseVector")
        .def("__len__", &mtl::vec::dense_vector<double>::size)
        .def("__getitem__", [](const mtl::vec::dense_vector<double>& v,
                               std::size_t i) -> double {
            if (i >= v.size())
                throw nb::index_error();
            return v[i];
        })
        .def("__setitem__", [](mtl::vec::dense_vector<double>& v,
                               std::size_t i, double val) {
            if (i >= v.size())
                throw nb::index_error();
            v[i] = val;
        })
        .def("to_numpy", [](const mtl::vec::dense_vector<double>& v) {
            std::size_t shape[1] = { v.size() };
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(
                const_cast<double*>(v.data()), 1, shape, nb::handle());
        }, nb::rv_policy::copy,
           "Return a NumPy array copy of this vector")
        .def("__repr__", [](const mtl::vec::dense_vector<double>& v) {
            std::ostringstream os;
            os << "mtl5.DenseVector([";
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (i) os << ", ";
                if (i >= 6 && v.size() > 8) { os << "..."; break; }
                os << v[i];
            }
            os << "], size=" << v.size() << ")";
            return os.str();
        });

    // -- dense2D<double> bindings -------------------------------------------
    nb::class_<mtl::mat::dense2D<double>>(m, "DenseMatrix")
        .def_prop_ro("num_rows", &mtl::mat::dense2D<double>::num_rows)
        .def_prop_ro("num_cols", &mtl::mat::dense2D<double>::num_cols)
        .def_prop_ro("shape", [](const mtl::mat::dense2D<double>& M) {
            return std::pair<std::size_t, std::size_t>(M.num_rows(), M.num_cols());
        })
        .def("__getitem__", [](const mtl::mat::dense2D<double>& M,
                               std::pair<std::size_t, std::size_t> idx) -> double {
            if (idx.first >= M.num_rows() || idx.second >= M.num_cols())
                throw nb::index_error();
            return M(idx.first, idx.second);
        })
        .def("__setitem__", [](mtl::mat::dense2D<double>& M,
                               std::pair<std::size_t, std::size_t> idx, double val) {
            if (idx.first >= M.num_rows() || idx.second >= M.num_cols())
                throw nb::index_error();
            M(idx.first, idx.second) = val;
        })
        .def("to_numpy", [](const mtl::mat::dense2D<double>& M) {
            std::size_t shape[2] = { M.num_rows(), M.num_cols() };
            // Copy data in row-major order
            double* buf = new double[M.num_rows() * M.num_cols()];
            for (std::size_t r = 0; r < M.num_rows(); ++r)
                for (std::size_t c = 0; c < M.num_cols(); ++c)
                    buf[r * M.num_cols() + c] = M(r, c);
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(buf, 2, shape, owner);
        }, "Return a NumPy array copy of this matrix")
        .def("__repr__", [](const mtl::mat::dense2D<double>& M) {
            std::ostringstream os;
            os << "mtl5.DenseMatrix(shape=(" << M.num_rows() << ", " << M.num_cols() << "))";
            return os.str();
        });

    // -- norm ---------------------------------------------------------------
    m.def("norm", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                     int ord) -> double {
        auto v = numpy_to_vector(a);
        switch (ord) {
            case 1:  return mtl::one_norm(v);
            case 2:  return mtl::two_norm(v);
            default:
                throw std::invalid_argument("ord must be 1 or 2 (infinity_norm via ord=-1)");
            case -1: return mtl::infinity_norm(v);
        }
    }, "x"_a, "ord"_a = 2,
       "Compute vector norm (ord=1: L1, ord=2: L2, ord=-1: Linf)");

    // -- dot product --------------------------------------------------------
    m.def("dot", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                    nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b) -> double {
        if (a.shape(0) != b.shape(0))
            throw std::invalid_argument("vectors must have the same length");
        auto va = numpy_to_vector(a);
        auto vb = numpy_to_vector(b);
        return mtl::dot(va, vb);
    }, "a"_a, "b"_a, "Compute dot product of two vectors");

    // -- solve (dense LU) ---------------------------------------------------
    m.def("solve", [](nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np,
                      nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
        std::size_t n = A_np.shape(0);
        if (A_np.shape(1) != n)
            throw std::invalid_argument("A must be square");
        if (b_np.shape(0) != n)
            throw std::invalid_argument("b length must match A dimensions");

        // Copy A (LU overwrites it)
        auto A = mtl::mat::dense2D<double>(n, n);
        const double* src_A = A_np.data();
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                A(r, c) = src_A[r * n + c];

        // Copy b
        auto b = mtl::vec::dense_vector<double>(n);
        const double* src_b = b_np.data();
        for (std::size_t i = 0; i < n; ++i)
            b[i] = src_b[i];

        // LU factorize
        std::vector<std::size_t> pivot;
        int info = mtl::lu_factor(A, pivot);
        if (info != 0) {
            std::ostringstream msg;
            msg << "Singular matrix (pivot " << info << " is zero)";
            throw std::runtime_error(msg.str());
        }

        // Forward/back substitution: lu_solve(LU, pivot, x, b)
        auto x = mtl::vec::dense_vector<double>(n);
        mtl::lu_solve(A, pivot, x, b);

        // Return as NumPy array
        std::size_t shape[1] = { n };
        double* buf = new double[n];
        for (std::size_t i = 0; i < n; ++i)
            buf[i] = x[i];
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
    }, "A"_a, "b"_a,
       "Solve Ax = b via LU factorization. Returns x as a NumPy array.");
}
