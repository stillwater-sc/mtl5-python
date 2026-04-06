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
using fp8  = sw::universal::fp8;
using fp16 = sw::universal::fp16;

// ===========================================================================
// Human-readable suffix for Python class names
// ===========================================================================
template <typename T> constexpr const char* type_suffix();
template <> constexpr const char* type_suffix<float>()   { return "f32"; }
template <> constexpr const char* type_suffix<double>()  { return "f64"; }
template <> constexpr const char* type_suffix<int32_t>() { return "i32"; }
template <> constexpr const char* type_suffix<int64_t>() { return "i64"; }
template <> constexpr const char* type_suffix<fp8>()     { return "fp8"; }
template <> constexpr const char* type_suffix<fp16>()    { return "fp16"; }

// ===========================================================================
// VectorView<T> — zero-copy wrapper around dense_vector<T>
//
// Holds a nb::object reference to the Python source array to prevent GC.
// The dense_vector uses MTL5's non-owning constructor (borrows memory).
// When _source is empty, the vector owns its memory (from vector_copy or solve).
// ===========================================================================
template <typename T>
struct VectorView {
    mtl::vec::dense_vector<T> vec;
    nb::object source;  // prevents GC of source array; empty if owning
    std::string device_name = "cpu";

    // Non-owning view of external data
    VectorView(std::size_t n, T* data, nb::object src)
        : vec(n, data), source(std::move(src)) {}

    // Owning vector (copy or result of computation)
    explicit VectorView(mtl::vec::dense_vector<T>&& v)
        : vec(std::move(v)) {}

    bool is_view() const { return source.is_valid(); }
};

// ===========================================================================
// MatrixView<T> — zero-copy wrapper around dense2D<T>
// ===========================================================================
template <typename T>
struct MatrixView {
    mtl::mat::dense2D<T> mat;
    nb::object source;
    std::string device_name = "cpu";

    // Non-owning view of external data
    MatrixView(std::size_t rows, std::size_t cols, T* data, nb::object src)
        : mat(rows, cols, data), source(std::move(src)) {}

    // Owning matrix (copy or result of computation)
    explicit MatrixView(mtl::mat::dense2D<T>&& m)
        : mat(std::move(m)) {}

    bool is_view() const { return source.is_valid(); }
};

// ===========================================================================
// Registration for native types (float, double, int32_t, int64_t)
// ===========================================================================
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_vector(nb::module_& m) {
    using VV = VectorView<T>;
    std::string name = std::string("DenseVector_") + type_suffix<T>();

    nb::class_<VV>(m, name.c_str())
        .def("__len__", [](const VV& vv) { return vv.vec.size(); })
        .def("__getitem__", [](const VV& vv, std::size_t i) -> T {
            if (i >= vv.vec.size()) throw nb::index_error();
            return vv.vec[i];
        })
        .def("__setitem__", [](VV& vv, std::size_t i, T val) {
            if (i >= vv.vec.size()) throw nb::index_error();
            vv.vec[i] = val;
        })
        .def_prop_ro("dtype", [](const VV&) { return type_suffix<T>(); })
        .def_prop_ro("device", [](const VV& vv) { return vv.device_name; })
        .def_prop_ro("is_view", [](const VV& vv) { return vv.is_view(); })
        .def("to_numpy", [](VV& vv) {
            // Zero-copy: return a NumPy array that shares memory with this vector
            std::size_t shape[1] = { vv.vec.size() };
            // The capsule prevents this VectorView from being GC'd while
            // the NumPy array exists, keeping the memory alive
            return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
                vv.vec.data(), 1, shape, nb::cast(vv));
        }, "Return a zero-copy NumPy array view of this vector")
        .def("copy", [](const VV& vv) {
            auto owned = mtl::vec::dense_vector<T>(vv.vec.size());
            for (std::size_t i = 0; i < vv.vec.size(); ++i)
                owned[i] = vv.vec[i];
            return VV(std::move(owned));
        }, "Return an owning copy of this vector")
        .def("to", [](const VV& vv, const std::string& device) {
            if (device != "cpu")
                throw std::runtime_error(
                    "Device '" + device + "' not available. "
                    "Currently supported: 'cpu'. KPU support coming soon.");
            // CPU → CPU: return a copy (future: device transfer)
            auto owned = mtl::vec::dense_vector<T>(vv.vec.size());
            for (std::size_t i = 0; i < vv.vec.size(); ++i)
                owned[i] = vv.vec[i];
            return VV(std::move(owned));
        }, "device"_a, "Transfer vector to device (currently: 'cpu' only)")
        .def("__repr__", [](const VV& vv) {
            std::ostringstream os;
            os << "mtl5.DenseVector_" << type_suffix<T>() << "([";
            for (std::size_t i = 0; i < vv.vec.size(); ++i) {
                if (i) os << ", ";
                if (i >= 6 && vv.vec.size() > 8) { os << "..."; break; }
                os << vv.vec[i];
            }
            os << "], size=" << vv.vec.size()
               << ", device='" << vv.device_name << "'"
               << (vv.is_view() ? ", view" : "") << ")";
            return os.str();
        });
}

template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_matrix(nb::module_& m) {
    using MV = MatrixView<T>;
    std::string name = std::string("DenseMatrix_") + type_suffix<T>();

    nb::class_<MV>(m, name.c_str())
        .def_prop_ro("num_rows", [](const MV& mv) { return mv.mat.num_rows(); })
        .def_prop_ro("num_cols", [](const MV& mv) { return mv.mat.num_cols(); })
        .def_prop_ro("shape", [](const MV& mv) {
            return std::pair<std::size_t, std::size_t>(mv.mat.num_rows(), mv.mat.num_cols());
        })
        .def_prop_ro("dtype", [](const MV&) { return type_suffix<T>(); })
        .def_prop_ro("device", [](const MV& mv) { return mv.device_name; })
        .def_prop_ro("is_view", [](const MV& mv) { return mv.is_view(); })
        .def("__getitem__", [](const MV& mv,
                               std::pair<std::size_t, std::size_t> idx) -> T {
            if (idx.first >= mv.mat.num_rows() || idx.second >= mv.mat.num_cols())
                throw nb::index_error();
            return mv.mat(idx.first, idx.second);
        })
        .def("__setitem__", [](MV& mv,
                               std::pair<std::size_t, std::size_t> idx, T val) {
            if (idx.first >= mv.mat.num_rows() || idx.second >= mv.mat.num_cols())
                throw nb::index_error();
            mv.mat(idx.first, idx.second) = val;
        })
        .def("to_numpy", [](MV& mv) {
            // Zero-copy: return a NumPy array that shares memory
            std::size_t shape[2] = { mv.mat.num_rows(), mv.mat.num_cols() };
            return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
                mv.mat.data(), 2, shape, nb::cast(mv));
        }, "Return a zero-copy NumPy array view of this matrix")
        .def("copy", [](const MV& mv) {
            auto owned = mtl::mat::dense2D<T>(mv.mat.num_rows(), mv.mat.num_cols());
            for (std::size_t r = 0; r < mv.mat.num_rows(); ++r)
                for (std::size_t c = 0; c < mv.mat.num_cols(); ++c)
                    owned(r, c) = mv.mat(r, c);
            return MV(std::move(owned));
        }, "Return an owning copy of this matrix")
        .def("to", [](const MV& mv, const std::string& device) {
            if (device != "cpu")
                throw std::runtime_error(
                    "Device '" + device + "' not available. "
                    "Currently supported: 'cpu'. KPU support coming soon.");
            auto owned = mtl::mat::dense2D<T>(mv.mat.num_rows(), mv.mat.num_cols());
            for (std::size_t r = 0; r < mv.mat.num_rows(); ++r)
                for (std::size_t c = 0; c < mv.mat.num_cols(); ++c)
                    owned(r, c) = mv.mat(r, c);
            return MV(std::move(owned));
        }, "device"_a, "Transfer matrix to device (currently: 'cpu' only)")
        .def("__repr__", [](const MV& mv) {
            std::ostringstream os;
            os << "mtl5.DenseMatrix_" << type_suffix<T>()
               << "(shape=(" << mv.mat.num_rows() << ", " << mv.mat.num_cols() << ")"
               << ", device='" << mv.device_name << "'"
               << (mv.is_view() ? ", view" : "") << ")";
            return os.str();
        });
}

// ---------------------------------------------------------------------------
// Zero-copy vector() factory — creates a view borrowing NumPy memory
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_vector_factory(nb::module_& m) {
    // Zero-copy: borrow memory from the NumPy array
    m.def("vector", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        return VectorView<T>(a.shape(0), a.data(), nb::cast(a));
    }, "a"_a, "Create a zero-copy MTL5 vector view of a 1-D NumPy array");

    // Explicit copy variant
    m.def("vector_copy", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        auto v = mtl::vec::dense_vector<T>(a.shape(0));
        const T* src = a.data();
        for (std::size_t i = 0; i < a.shape(0); ++i)
            v[i] = src[i];
        return VectorView<T>(std::move(v));
    }, "a"_a, "Create an owning MTL5 vector (copies data from NumPy array)");
}

// ---------------------------------------------------------------------------
// Zero-copy matrix() factory — creates a view borrowing NumPy memory
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_matrix_factory(nb::module_& m) {
    // Zero-copy: borrow memory from the NumPy array
    m.def("matrix", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        return MatrixView<T>(a.shape(0), a.shape(1), a.data(), nb::cast(a));
    }, "a"_a, "Create a zero-copy MTL5 matrix view of a 2-D NumPy array");

    // Explicit copy variant
    m.def("matrix_copy", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        auto M = mtl::mat::dense2D<T>(a.shape(0), a.shape(1));
        const T* src = a.data();
        for (std::size_t r = 0; r < a.shape(0); ++r)
            for (std::size_t c = 0; c < a.shape(1); ++c)
                M(r, c) = src[r * a.shape(1) + c];
        return MatrixView<T>(std::move(M));
    }, "a"_a, "Create an owning MTL5 matrix (copies data from NumPy array)");
}

// ---------------------------------------------------------------------------
// Norm/dot overloads for native VectorView<T>
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_norm_overload(nb::module_& m) {
    // Overload on ndarray (direct NumPy pass-through)
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

    // Overload on VectorView<T>
    m.def("norm", [](const VectorView<T>& vv, int ord) -> double {
        switch (ord) {
            case 1:  return static_cast<double>(mtl::one_norm(vv.vec));
            case 2:  return static_cast<double>(mtl::two_norm(vv.vec));
            case -1: return static_cast<double>(mtl::infinity_norm(vv.vec));
            default:
                throw std::invalid_argument("ord must be 1, 2, or -1 (infinity)");
        }
    }, "x"_a, "ord"_a = 2);
}

template <typename T>
    requires std::is_arithmetic_v<T>
void register_native_dot_overload(nb::module_& m) {
    // ndarray overload
    m.def("dot", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                    nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b) -> double {
        if (a.shape(0) != b.shape(0))
            throw std::invalid_argument("vectors must have the same length");
        auto va = mtl::vec::dense_vector<T>(a.shape(0), const_cast<T*>(a.data()));
        auto vb = mtl::vec::dense_vector<T>(b.shape(0), const_cast<T*>(b.data()));
        return static_cast<double>(mtl::dot(va, vb));
    }, "a"_a, "b"_a);

    // VectorView overload
    m.def("dot", [](const VectorView<T>& a, const VectorView<T>& b) -> double {
        if (a.vec.size() != b.vec.size())
            throw std::invalid_argument("vectors must have the same length");
        return static_cast<double>(mtl::dot(a.vec, b.vec));
    }, "a"_a, "b"_a);
}

// ---------------------------------------------------------------------------
// Solve for native types — accepts both ndarray and VectorView/MatrixView
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_floating_point_v<T>
void register_native_solve(nb::module_& m) {
    // ndarray overload (existing behavior)
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
        if (info != 0)
            throw std::runtime_error("Singular matrix (pivot " + std::to_string(info) + " is zero)");

        auto x = mtl::vec::dense_vector<T>(n);
        mtl::lu_solve(A, pivot, x, b);

        std::size_t shape[1] = { n };
        T* buf = new T[n];
        for (std::size_t i = 0; i < n; ++i)
            buf[i] = x[i];
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
        return nb::ndarray<nb::numpy, T, nb::ndim<1>>(buf, 1, shape, owner);
    }, "A"_a, "b"_a);

    // VectorView/MatrixView overload
    m.def("solve", [](const MatrixView<T>& A_mv, const VectorView<T>& b_vv) {
        std::size_t n = A_mv.mat.num_rows();
        if (A_mv.mat.num_cols() != n)
            throw std::invalid_argument("A must be square");
        if (b_vv.vec.size() != n)
            throw std::invalid_argument("b length must match A dimensions");

        auto A = mtl::mat::dense2D<T>(n, n);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                A(r, c) = A_mv.mat(r, c);

        auto b = mtl::vec::dense_vector<T>(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = b_vv.vec[i];

        std::vector<std::size_t> pivot;
        int info = mtl::lu_factor(A, pivot);
        if (info != 0)
            throw std::runtime_error("Singular matrix (pivot " + std::to_string(info) + " is zero)");

        auto x = mtl::vec::dense_vector<T>(n);
        mtl::lu_solve(A, pivot, x, b);
        return VectorView<T>(std::move(x));
    }, "A"_a, "b"_a);
}

// ===========================================================================
// Registration for Universal types (fp8, fp16, posit, etc.)
// These always copy (no NumPy dtype), but get device stubs
// ===========================================================================
template <typename T>
void register_universal_vector(nb::module_& m) {
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
        .def_prop_ro("device", [](const Vec&) { return "cpu"; })
        .def("to_numpy", [](const Vec& v) {
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

template <typename T>
void register_universal_matrix(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    std::string name = std::string("DenseMatrix_") + type_suffix<T>();

    nb::class_<Mat>(m, name.c_str())
        .def_prop_ro("num_rows", &Mat::num_rows)
        .def_prop_ro("num_cols", &Mat::num_cols)
        .def_prop_ro("shape", [](const Mat& M) {
            return std::pair<std::size_t, std::size_t>(M.num_rows(), M.num_cols());
        })
        .def_prop_ro("dtype", [](const Mat&) { return type_suffix<T>(); })
        .def_prop_ro("device", [](const Mat&) { return "cpu"; })
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

template <typename T>
void register_universal_vector_factory(nb::module_& m, const char* func_name) {
    m.def(func_name, [](nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        auto v = mtl::vec::dense_vector<T>(a.shape(0));
        const double* src = a.data();
        for (std::size_t i = 0; i < a.shape(0); ++i)
            v[i] = static_cast<T>(src[i]);
        return v;
    }, "a"_a);
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
    }, "a"_a);
}

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
    }, "x"_a, "ord"_a = 2);
}

template <typename T>
void register_universal_dot(nb::module_& m) {
    using Vec = mtl::vec::dense_vector<T>;
    m.def("dot", [](const Vec& a, const Vec& b) -> double {
        if (a.size() != b.size())
            throw std::invalid_argument("vectors must have the same length");
        return static_cast<double>(mtl::dot(a, b));
    }, "a"_a, "b"_a);
}

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

        auto A = mtl::mat::dense2D<T>(n, n);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                A(r, c) = A_in(r, c);

        auto b = mtl::vec::dense_vector<T>(n);
        for (std::size_t i = 0; i < n; ++i)
            b[i] = b_in[i];

        std::vector<std::size_t> pivot;
        int info = mtl::lu_factor(A, pivot);
        if (info != 0)
            throw std::runtime_error("Singular matrix (pivot " + std::to_string(info) + " is zero)");

        auto x = mtl::vec::dense_vector<T>(n);
        mtl::lu_solve(A, pivot, x, b);
        return x;
    }, "A"_a, "b"_a);
}

// ===========================================================================
// Convenience: register all bindings for one type
// ===========================================================================
template <typename T>
    requires std::is_arithmetic_v<T>
void register_native(nb::module_& m) {
    register_native_vector<T>(m);
    register_native_matrix<T>(m);
    register_native_vector_factory<T>(m);
    register_native_matrix_factory<T>(m);
    register_native_norm_overload<T>(m);
    register_native_dot_overload<T>(m);
}

template <typename T>
    requires std::is_floating_point_v<T>
void register_native_with_solve(nb::module_& m) {
    register_native<T>(m);
    register_native_solve<T>(m);
}

template <typename T>
void register_universal(nb::module_& m, const char* vec_factory, const char* mat_factory) {
    register_universal_vector<T>(m);
    register_universal_matrix<T>(m);
    register_universal_vector_factory<T>(m, vec_factory);
    register_universal_matrix_factory<T>(m, mat_factory);
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

    // ----- Device management -------------------------------------------------
    m.def("devices", []() {
        return std::vector<std::string>{"cpu"};
        // Future: enumerate KPU devices, BLAS backends
    }, "List available execution devices");

    // ----- Native C++ types (zero-copy via nb::ndarray) ----------------------
    register_native_with_solve<float>(m);     // f32
    register_native_with_solve<double>(m);    // f64
    register_native<int32_t>(m);              // i32
    register_native<int64_t>(m);              // i64

    // ----- Universal number types (copy-converting from float64) -------------
    register_universal<fp8>(m, "vector_fp8", "matrix_fp8");
    register_universal<fp16>(m, "vector_fp16", "matrix_fp16");
}
