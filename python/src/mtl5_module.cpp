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
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// ===========================================================================
// Type registration templates
//
// Each register_*<T>() function stamps out a complete set of bindings for
// one scalar type.  Adding a new precision is a single call at the bottom
// of NB_MODULE.
// ===========================================================================

// ---------------------------------------------------------------------------
// NumPy dtype code for a C++ scalar type
// ---------------------------------------------------------------------------
template <typename T> struct numpy_dtype;
template <> struct numpy_dtype<float>    { static constexpr auto value = nb::dtype<float>();    };
template <> struct numpy_dtype<double>   { static constexpr auto value = nb::dtype<double>();   };
template <> struct numpy_dtype<int32_t>  { static constexpr auto value = nb::dtype<int32_t>();  };
template <> struct numpy_dtype<int64_t>  { static constexpr auto value = nb::dtype<int64_t>();  };

// ---------------------------------------------------------------------------
// Human-readable suffix for Python class names:  "f32", "f64", "i32", "i64"
// ---------------------------------------------------------------------------
template <typename T> constexpr const char* type_suffix();
template <> constexpr const char* type_suffix<float>()   { return "f32"; }
template <> constexpr const char* type_suffix<double>()  { return "f64"; }
template <> constexpr const char* type_suffix<int32_t>() { return "i32"; }
template <> constexpr const char* type_suffix<int64_t>() { return "i64"; }

// ---------------------------------------------------------------------------
// register_dense_vector<T>  →  DenseVector_f32, DenseVector_f64, …
// ---------------------------------------------------------------------------
template <typename T>
void register_dense_vector(nb::module_& m) {
    using Vec = mtl::vec::dense_vector<T>;
    std::string name = std::string("DenseVector_") + type_suffix<T>();

    nb::class_<Vec>(m, name.c_str())
        .def("__len__", &Vec::size)
        .def("__getitem__", [](const Vec& v, std::size_t i) -> T {
            if (i >= v.size()) throw nb::index_error();
            return v[i];
        })
        .def("__setitem__", [](Vec& v, std::size_t i, T val) {
            if (i >= v.size()) throw nb::index_error();
            v[i] = val;
        })
        .def_prop_ro("dtype", [](const Vec&) { return type_suffix<T>(); })
        .def("to_numpy", [](const Vec& v) {
            std::size_t shape[1] = { v.size() };
            T* buf = new T[v.size()];
            for (std::size_t i = 0; i < v.size(); ++i)
                buf[i] = v[i];
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
            return nb::ndarray<nb::numpy, T, nb::ndim<1>>(buf, 1, shape, owner);
        }, "Return a NumPy array copy of this vector")
        .def("__repr__", [](const Vec& v) {
            std::ostringstream os;
            os << "mtl5.DenseVector_" << type_suffix<T>() << "([";
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (i) os << ", ";
                if (i >= 6 && v.size() > 8) { os << "..."; break; }
                os << v[i];
            }
            os << "], size=" << v.size() << ")";
            return os.str();
        });
}

// ---------------------------------------------------------------------------
// register_dense_matrix<T>  →  DenseMatrix_f32, DenseMatrix_f64, …
// ---------------------------------------------------------------------------
template <typename T>
void register_dense_matrix(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    std::string name = std::string("DenseMatrix_") + type_suffix<T>();

    nb::class_<Mat>(m, name.c_str())
        .def_prop_ro("num_rows", &Mat::num_rows)
        .def_prop_ro("num_cols", &Mat::num_cols)
        .def_prop_ro("shape", [](const Mat& M) {
            return std::pair<std::size_t, std::size_t>(M.num_rows(), M.num_cols());
        })
        .def_prop_ro("dtype", [](const Mat&) { return type_suffix<T>(); })
        .def("__getitem__", [](const Mat& M,
                               std::pair<std::size_t, std::size_t> idx) -> T {
            if (idx.first >= M.num_rows() || idx.second >= M.num_cols())
                throw nb::index_error();
            return M(idx.first, idx.second);
        })
        .def("__setitem__", [](Mat& M,
                               std::pair<std::size_t, std::size_t> idx, T val) {
            if (idx.first >= M.num_rows() || idx.second >= M.num_cols())
                throw nb::index_error();
            M(idx.first, idx.second) = val;
        })
        .def("to_numpy", [](const Mat& M) {
            std::size_t shape[2] = { M.num_rows(), M.num_cols() };
            T* buf = new T[M.num_rows() * M.num_cols()];
            for (std::size_t r = 0; r < M.num_rows(); ++r)
                for (std::size_t c = 0; c < M.num_cols(); ++c)
                    buf[r * M.num_cols() + c] = M(r, c);
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
            return nb::ndarray<nb::numpy, T, nb::ndim<2>>(buf, 2, shape, owner);
        }, "Return a NumPy array copy of this matrix")
        .def("__repr__", [](const Mat& M) {
            std::ostringstream os;
            os << "mtl5.DenseMatrix_" << type_suffix<T>()
               << "(shape=(" << M.num_rows() << ", " << M.num_cols() << "))";
            return os.str();
        });
}

// ---------------------------------------------------------------------------
// register_vector_overload<T>  — add a typed overload to mtl5.vector()
// ---------------------------------------------------------------------------
template <typename T>
void register_vector_overload(nb::module_& m) {
    m.def("vector", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        auto v = mtl::vec::dense_vector<T>(a.shape(0));
        const T* src = a.data();
        for (std::size_t i = 0; i < a.shape(0); ++i)
            v[i] = src[i];
        return v;
    }, "a"_a);
}

// ---------------------------------------------------------------------------
// register_matrix_overload<T>  — add a typed overload to mtl5.matrix()
// ---------------------------------------------------------------------------
template <typename T>
void register_matrix_overload(nb::module_& m) {
    m.def("matrix", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        auto M = mtl::mat::dense2D<T>(a.shape(0), a.shape(1));
        const T* src = a.data();
        for (std::size_t r = 0; r < a.shape(0); ++r)
            for (std::size_t c = 0; c < a.shape(1); ++c)
                M(r, c) = src[r * a.shape(1) + c];
        return M;
    }, "a"_a);
}

// ---------------------------------------------------------------------------
// register_norm_overload<T>  — add a typed overload to mtl5.norm()
// ---------------------------------------------------------------------------
template <typename T>
void register_norm_overload(nb::module_& m) {
    m.def("norm", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                     int ord) -> double {
        auto v = mtl::vec::dense_vector<T>(a.shape(0), const_cast<T*>(a.data()));
        switch (ord) {
            case 1:  return static_cast<double>(mtl::one_norm(v));
            case 2:  return static_cast<double>(mtl::two_norm(v));
            case -1: return static_cast<double>(mtl::infinity_norm(v));
            default:
                throw std::invalid_argument("ord must be 1, 2, or -1 (infinity)");
        }
    }, "x"_a, "ord"_a = 2);
}

// ---------------------------------------------------------------------------
// register_dot_overload<T>  — add a typed overload to mtl5.dot()
// ---------------------------------------------------------------------------
template <typename T>
void register_dot_overload(nb::module_& m) {
    m.def("dot", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                    nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b) -> double {
        if (a.shape(0) != b.shape(0))
            throw std::invalid_argument("vectors must have the same length");
        auto va = mtl::vec::dense_vector<T>(a.shape(0), const_cast<T*>(a.data()));
        auto vb = mtl::vec::dense_vector<T>(b.shape(0), const_cast<T*>(b.data()));
        return static_cast<double>(mtl::dot(va, vb));
    }, "a"_a, "b"_a);
}

// ---------------------------------------------------------------------------
// register_solve_overload<T>  — add a typed overload to mtl5.solve()
//
// LU factorization is only meaningful for floating-point types, so this
// is only instantiated for float/double (and future Universal types).
// ---------------------------------------------------------------------------
template <typename T>
void register_solve_overload(nb::module_& m) {
    m.def("solve", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np,
                      nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
        std::size_t n = A_np.shape(0);
        if (A_np.shape(1) != n)
            throw std::invalid_argument("A must be square");
        if (b_np.shape(0) != n)
            throw std::invalid_argument("b length must match A dimensions");

        // Copy A (LU overwrites it)
        auto A = mtl::mat::dense2D<T>(n, n);
        const T* src_A = A_np.data();
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                A(r, c) = src_A[r * n + c];

        // Copy b
        auto b = mtl::vec::dense_vector<T>(n);
        const T* src_b = b_np.data();
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

        // Forward/back substitution
        auto x = mtl::vec::dense_vector<T>(n);
        mtl::lu_solve(A, pivot, x, b);

        // Return as NumPy array
        std::size_t shape[1] = { n };
        T* buf = new T[n];
        for (std::size_t i = 0; i < n; ++i)
            buf[i] = x[i];
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
        return nb::ndarray<nb::numpy, T, nb::ndim<1>>(buf, 1, shape, owner);
    }, "A"_a, "b"_a);
}

// ===========================================================================
// Convenience: register all bindings for one scalar type in a single call
// ===========================================================================
template <typename T>
void register_all(nb::module_& m) {
    register_dense_vector<T>(m);
    register_dense_matrix<T>(m);
    register_vector_overload<T>(m);
    register_matrix_overload<T>(m);
    register_norm_overload<T>(m);
    register_dot_overload<T>(m);
}

// Floating-point types also get solve()
template <typename T>
void register_all_with_solve(nb::module_& m) {
    register_all<T>(m);
    register_solve_overload<T>(m);
}

// ===========================================================================
// Module definition
// ===========================================================================
NB_MODULE(_core, m) {
    m.doc() = "MTL5 Python bindings — nanobind core module";

    m.attr("__version__") = "0.1.0";

    // ----- Register types ---------------------------------------------------
    // Each line stamps out: DenseVector_*, DenseMatrix_*, and typed overloads
    // for vector(), matrix(), norm(), dot(), and (for floats) solve().
    //
    // To add a new precision, add one line here:
    //   register_all_with_solve<half>(m);          // fp16
    //   register_all_with_solve<posit<16,2>>(m);   // posit16
    //   register_all<int8_t>(m);                   // integer-only (no solve)

    register_all_with_solve<float>(m);     // f32
    register_all_with_solve<double>(m);    // f64

    register_all<int32_t>(m);              // i32 — no solve (integer)
    register_all<int64_t>(m);              // i64 — no solve (integer)
}
