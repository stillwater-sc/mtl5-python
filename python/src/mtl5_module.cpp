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

// Universal number types
#include <universal/number/cfloat/cfloat.hpp>

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// Universal type aliases
using fp8  = sw::universal::fp8;    // cfloat<8, 2, uint8_t, true, false, false>
using fp16 = sw::universal::fp16;   // cfloat<16, 5, uint16_t, true, false, false>

// ===========================================================================
// Type registration templates
//
// Each register_*<T>() function stamps out a complete set of bindings for
// one scalar type.  Adding a new precision is a single call at the bottom
// of NB_MODULE.
// ===========================================================================

// ---------------------------------------------------------------------------
// Human-readable suffix for Python class names
// ---------------------------------------------------------------------------
template <typename T> constexpr const char* type_suffix();
template <> constexpr const char* type_suffix<float>()   { return "f32"; }
template <> constexpr const char* type_suffix<double>()  { return "f64"; }
template <> constexpr const char* type_suffix<int32_t>() { return "i32"; }
template <> constexpr const char* type_suffix<int64_t>() { return "i64"; }
template <> constexpr const char* type_suffix<fp8>()     { return "fp8"; }
template <> constexpr const char* type_suffix<fp16>()    { return "fp16"; }

// ---------------------------------------------------------------------------
// register_dense_vector<T>  →  DenseVector_f32, DenseVector_fp8, …
// ---------------------------------------------------------------------------
template <typename T>
void register_dense_vector(nb::module_& m) {
    using Vec = mtl::vec::dense_vector<T>;
    std::string name = std::string("DenseVector_") + type_suffix<T>();

    nb::class_<Vec>(m, name.c_str())
        .def("__len__", &Vec::size)
        .def("__getitem__", [](const Vec& v, std::size_t i) -> double {
            if (i >= v.size()) throw nb::index_error();
            return static_cast<double>(v[i]);
        })
        .def("__setitem__", [](Vec& v, std::size_t i, double val) {
            if (i >= v.size()) throw nb::index_error();
            v[i] = static_cast<T>(val);
        })
        .def_prop_ro("dtype", [](const Vec&) { return type_suffix<T>(); })
        .def("to_numpy", [](const Vec& v) {
            // Universal types convert to float64 for NumPy export
            std::size_t shape[1] = { v.size() };
            double* buf = new double[v.size()];
            for (std::size_t i = 0; i < v.size(); ++i)
                buf[i] = static_cast<double>(v[i]);
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            return nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
        }, "Return a NumPy float64 array (values converted from native precision)")
        .def("to_list", [](const Vec& v) {
            std::vector<double> result(v.size());
            for (std::size_t i = 0; i < v.size(); ++i)
                result[i] = static_cast<double>(v[i]);
            return result;
        }, "Return values as a Python list of floats")
        .def("__repr__", [](const Vec& v) {
            std::ostringstream os;
            os << "mtl5.DenseVector_" << type_suffix<T>() << "([";
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (i) os << ", ";
                if (i >= 6 && v.size() > 8) { os << "..."; break; }
                os << static_cast<double>(v[i]);
            }
            os << "], size=" << v.size() << ")";
            return os.str();
        });
}

// Specialization for native NumPy types — to_numpy returns same dtype
template <typename T>
    requires std::is_arithmetic_v<T>
void register_dense_vector_native(nb::module_& m) {
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
// register_dense_matrix<T>  →  DenseMatrix_f32, DenseMatrix_fp8, …
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
                               std::pair<std::size_t, std::size_t> idx) -> double {
            if (idx.first >= M.num_rows() || idx.second >= M.num_cols())
                throw nb::index_error();
            return static_cast<double>(M(idx.first, idx.second));
        })
        .def("__setitem__", [](Mat& M,
                               std::pair<std::size_t, std::size_t> idx, double val) {
            if (idx.first >= M.num_rows() || idx.second >= M.num_cols())
                throw nb::index_error();
            M(idx.first, idx.second) = static_cast<T>(val);
        })
        .def("to_numpy", [](const Mat& M) {
            std::size_t shape[2] = { M.num_rows(), M.num_cols() };
            double* buf = new double[M.num_rows() * M.num_cols()];
            for (std::size_t r = 0; r < M.num_rows(); ++r)
                for (std::size_t c = 0; c < M.num_cols(); ++c)
                    buf[r * M.num_cols() + c] = static_cast<double>(M(r, c));
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            return nb::ndarray<nb::numpy, double, nb::ndim<2>>(buf, 2, shape, owner);
        }, "Return a NumPy float64 array (values converted from native precision)")
        .def("__repr__", [](const Mat& M) {
            std::ostringstream os;
            os << "mtl5.DenseMatrix_" << type_suffix<T>()
               << "(shape=(" << M.num_rows() << ", " << M.num_cols() << "))";
            return os.str();
        });
}

// Specialization for native NumPy types
template <typename T>
    requires std::is_arithmetic_v<T>
void register_dense_matrix_native(nb::module_& m) {
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
// Factory overloads for native NumPy types (dtype-dispatched via nb::ndarray)
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_vector_overload(nb::module_& m) {
    m.def("vector", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        auto v = mtl::vec::dense_vector<T>(a.shape(0));
        const T* src = a.data();
        for (std::size_t i = 0; i < a.shape(0); ++i)
            v[i] = src[i];
        return v;
    }, "a"_a);
}

template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_matrix_overload(nb::module_& m) {
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
// Factory functions for Universal types (accept float64 array, convert)
// ---------------------------------------------------------------------------
template <typename T>
void register_universal_vector_factory(nb::module_& m, const char* func_name) {
    m.def(func_name, [](nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        auto v = mtl::vec::dense_vector<T>(a.shape(0));
        const double* src = a.data();
        for (std::size_t i = 0; i < a.shape(0); ++i)
            v[i] = static_cast<T>(src[i]);
        return v;
    }, "a"_a,
       (std::string("Create a DenseVector_") + type_suffix<T>() +
        " from a NumPy array (values converted to " + type_suffix<T>() + ")").c_str());
}

template <typename T>
void register_universal_matrix_factory(nb::module_& m, const char* func_name) {
    m.def(func_name, [](nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        auto M = mtl::mat::dense2D<T>(a.shape(0), a.shape(1));
        const double* src = a.data();
        for (std::size_t r = 0; r < a.shape(0); ++r)
            for (std::size_t c = 0; c < a.shape(1); ++c)
                M(r, c) = static_cast<T>(src[r * a.shape(1) + c]);
        return M;
    }, "a"_a,
       (std::string("Create a DenseMatrix_") + type_suffix<T>() +
        " from a NumPy array (values converted to " + type_suffix<T>() + ")").c_str());
}

// ---------------------------------------------------------------------------
// Norm and dot overloads for native types (dtype-dispatched)
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_norm_overload(nb::module_& m) {
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

template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_dot_overload(nb::module_& m) {
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
// Norm and dot for Universal types (operate on DenseVector<T> directly)
// ---------------------------------------------------------------------------
template <typename T>
void register_universal_norm(nb::module_& m) {
    using Vec = mtl::vec::dense_vector<T>;
    m.def("norm", [](const Vec& v, int ord) -> double {
        switch (ord) {
            case 1:  return static_cast<double>(mtl::one_norm(v));
            case 2:  return static_cast<double>(mtl::two_norm(v));
            case -1: return static_cast<double>(mtl::infinity_norm(v));
            default:
                throw std::invalid_argument("ord must be 1, 2, or -1 (infinity)");
        }
    }, "x"_a, "ord"_a = 2,
       (std::string("Compute norm of a ") + type_suffix<T>() + " vector").c_str());
}

template <typename T>
void register_universal_dot(nb::module_& m) {
    using Vec = mtl::vec::dense_vector<T>;
    m.def("dot", [](const Vec& a, const Vec& b) -> double {
        if (a.size() != b.size())
            throw std::invalid_argument("vectors must have the same length");
        return static_cast<double>(mtl::dot(a, b));
    }, "a"_a, "b"_a,
       (std::string("Compute dot product of two ") + type_suffix<T>() + " vectors").c_str());
}

// ---------------------------------------------------------------------------
// Solve overload for native floating-point types
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_floating_point_v<T>
void register_native_solve_overload(nb::module_& m) {
    m.def("solve", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np,
                      nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
        std::size_t n = A_np.shape(0);
        if (A_np.shape(1) != n)
            throw std::invalid_argument("A must be square");
        if (b_np.shape(0) != n)
            throw std::invalid_argument("b length must match A dimensions");

        auto A = mtl::mat::dense2D<T>(n, n);
        const T* src_A = A_np.data();
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                A(r, c) = src_A[r * n + c];

        auto b = mtl::vec::dense_vector<T>(n);
        const T* src_b = b_np.data();
        for (std::size_t i = 0; i < n; ++i)
            b[i] = src_b[i];

        std::vector<std::size_t> pivot;
        int info = mtl::lu_factor(A, pivot);
        if (info != 0) {
            std::ostringstream msg;
            msg << "Singular matrix (pivot " << info << " is zero)";
            throw std::runtime_error(msg.str());
        }

        auto x = mtl::vec::dense_vector<T>(n);
        mtl::lu_solve(A, pivot, x, b);

        std::size_t shape[1] = { n };
        T* buf = new T[n];
        for (std::size_t i = 0; i < n; ++i)
            buf[i] = x[i];
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
        return nb::ndarray<nb::numpy, T, nb::ndim<1>>(buf, 1, shape, owner);
    }, "A"_a, "b"_a);
}

// ---------------------------------------------------------------------------
// Solve for Universal types (accept DenseMatrix + DenseVector, return DenseVector)
// ---------------------------------------------------------------------------
template <typename T>
void register_universal_solve(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    using Vec = mtl::vec::dense_vector<T>;
    m.def("solve", [](const Mat& A_in, const Vec& b_in) {
        std::size_t n = A_in.num_rows();
        if (A_in.num_cols() != n)
            throw std::invalid_argument("A must be square");
        if (b_in.size() != n)
            throw std::invalid_argument("b length must match A dimensions");

        // Copy A (LU overwrites it)
        auto A = mtl::mat::dense2D<T>(n, n);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                A(r, c) = A_in(r, c);

        auto b = mtl::vec::dense_vector<T>(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = b_in[i];

        std::vector<std::size_t> pivot;
        int info = mtl::lu_factor(A, pivot);
        if (info != 0) {
            std::ostringstream msg;
            msg << "Singular matrix (pivot " << info << " is zero)";
            throw std::runtime_error(msg.str());
        }

        auto x = mtl::vec::dense_vector<T>(n);
        mtl::lu_solve(A, pivot, x, b);
        return x;
    }, "A"_a, "b"_a,
       (std::string("Solve Ax=b in ") + type_suffix<T>() + " precision").c_str());
}

// ===========================================================================
// Convenience: register all bindings for one type
// ===========================================================================

// Native C++ arithmetic types (float, double, int32_t, int64_t)
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native(nb::module_& m) {
    register_dense_vector_native<T>(m);
    register_dense_matrix_native<T>(m);
    register_native_vector_overload<T>(m);
    register_native_matrix_overload<T>(m);
    register_native_norm_overload<T>(m);
    register_native_dot_overload<T>(m);
}

template <typename T>
    requires std::is_floating_point_v<T>
void register_native_with_solve(nb::module_& m) {
    register_native<T>(m);
    register_native_solve_overload<T>(m);
}

// Universal number types (fp8, fp16, posit, cfloat, etc.)
template <typename T>
void register_universal(nb::module_& m, const char* vector_factory, const char* matrix_factory) {
    register_dense_vector<T>(m);
    register_dense_matrix<T>(m);
    register_universal_vector_factory<T>(m, vector_factory);
    register_universal_matrix_factory<T>(m, matrix_factory);
    register_universal_norm<T>(m);
    register_universal_dot<T>(m);
    register_universal_solve<T>(m);
}

// ===========================================================================
// Module definition
// ===========================================================================
NB_MODULE(_core, m) {
    m.doc() = "MTL5 Python bindings — nanobind core module";

    m.attr("__version__") = "0.1.0";

    // ----- Native C++ types (dtype-dispatched via nb::ndarray) ---------------
    register_native_with_solve<float>(m);     // f32
    register_native_with_solve<double>(m);    // f64
    register_native<int32_t>(m);              // i32
    register_native<int64_t>(m);              // i64

    // ----- Universal number types --------------------------------------------
    // These use named factory functions since they have no NumPy dtype.
    // Usage: mtl5.vector_fp8(np_array), mtl5.vector_fp16(np_array)
    //
    // To add a new Universal type, add one line here:
    //   register_universal<posit<16,2>>(m, "vector_posit16", "matrix_posit16");

    register_universal<fp8>(m, "vector_fp8", "matrix_fp8");
    register_universal<fp16>(m, "vector_fp16", "matrix_fp16");
}
