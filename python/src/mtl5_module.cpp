#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/mat/compressed2D.hpp>

// Iterative solvers and preconditioners
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>

// Universal number types
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>
#include <universal/number/fixpnt/fixpnt.hpp>
#include <universal/number/lns/lns.hpp>

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// Universal type aliases
using fp8     = sw::universal::fp8;
using fp16    = sw::universal::fp16;

// Posit types — tapered precision floats with two exponent bits
using posit8  = sw::universal::posit<8, 2>;
using posit16 = sw::universal::posit<16, 2>;
using posit32 = sw::universal::posit<32, 2>;
using posit64 = sw::universal::posit<64, 2>;

// Fixed-point types — saturating arithmetic for bounded precision
// fixpnt8<8,4>: range [-8, 8) with 4 fractional bits (resolution 1/16)
// fixpnt16<16,8>: range [-128, 128) with 8 fractional bits (resolution 1/256)
using fixpnt8  = sw::universal::fixpnt<8, 4, sw::universal::Saturate>;
using fixpnt16 = sw::universal::fixpnt<16, 8, sw::universal::Saturate>;

// Logarithmic number system — multiplications become additions
using lns16 = sw::universal::lns<16, 8>;
using lns32 = sw::universal::lns<32, 16>;

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
template <> constexpr const char* type_suffix<posit8>()  { return "posit8"; }
template <> constexpr const char* type_suffix<posit16>() { return "posit16"; }
template <> constexpr const char* type_suffix<posit32>() { return "posit32"; }
template <> constexpr const char* type_suffix<posit64>() { return "posit64"; }
template <> constexpr const char* type_suffix<fixpnt8>() { return "fixpnt8"; }
template <> constexpr const char* type_suffix<fixpnt16>(){ return "fixpnt16"; }
template <> constexpr const char* type_suffix<lns16>()   { return "lns16"; }
template <> constexpr const char* type_suffix<lns32>()   { return "lns32"; }

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
        .def("to_numpy", [](nb::handle self) {
            // Zero-copy: return a NumPy array that shares memory with this vector.
            // Use nb::handle to get the actual Python wrapper object as the
            // keep-alive owner — nb::cast(vv) on a C++ reference creates a
            // separate copy whose data() differs from the source, causing
            // use-after-free on chained calls like solve(b).to_numpy().
            VV& vv = nb::cast<VV&>(self);
            std::size_t shape[1] = { vv.vec.size() };
            return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
                vv.vec.data(), 1, shape, self);
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
        .def("to_numpy", [](nb::handle self) {
            // Use nb::handle for keep-alive — see VectorView::to_numpy comment.
            MV& mv = nb::cast<MV&>(self);
            std::size_t shape[2] = { mv.mat.num_rows(), mv.mat.num_cols() };
            return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
                mv.mat.data(), 2, shape, self);
        }, "Return a zero-copy NumPy array view of this matrix")
        .def_prop_ro("T", [](const MV& mv) {
            std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
            mtl::mat::dense2D<T> AT(c, r);
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t j = 0; j < c; ++j)
                    AT(j, i) = mv.mat(i, j);
            return MV(std::move(AT));
        })
        .def("__matmul__", [](const MV& A, const MV& B) {
            if (A.mat.num_cols() != B.mat.num_rows())
                throw std::invalid_argument("matmul: A.num_cols != B.num_rows");
            mtl::mat::dense2D<T> Ac(A.mat.num_rows(), A.mat.num_cols());
            for (std::size_t i = 0; i < A.mat.num_rows(); ++i)
                for (std::size_t j = 0; j < A.mat.num_cols(); ++j)
                    Ac(i, j) = A.mat(i, j);
            mtl::mat::dense2D<T> Bc(B.mat.num_rows(), B.mat.num_cols());
            for (std::size_t i = 0; i < B.mat.num_rows(); ++i)
                for (std::size_t j = 0; j < B.mat.num_cols(); ++j)
                    Bc(i, j) = B.mat(i, j);
            mtl::mat::dense2D<T> C(Ac.num_rows(), Bc.num_cols());
            mtl::mult(Ac, Bc, C);
            return MV(std::move(C));
        })
        .def("__matmul__", [](const MV& A, const VectorView<T>& x) {
            if (A.mat.num_cols() != x.vec.size())
                throw std::invalid_argument("matmul: A.num_cols != len(x)");
            mtl::mat::dense2D<T> Ac(A.mat.num_rows(), A.mat.num_cols());
            for (std::size_t i = 0; i < A.mat.num_rows(); ++i)
                for (std::size_t j = 0; j < A.mat.num_cols(); ++j)
                    Ac(i, j) = A.mat(i, j);
            mtl::vec::dense_vector<T> xc(x.vec.size());
            for (std::size_t i = 0; i < x.vec.size(); ++i) xc[i] = x.vec[i];
            mtl::vec::dense_vector<T> y(A.mat.num_rows());
            mtl::mult(Ac, xc, y);
            return VectorView<T>(std::move(y));
        })
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
// LUFactor<T> — wraps an LU factorization (LU matrix + pivot vector)
// Returned by mtl5.lu(A); supports .solve(b) for repeated solves.
// ===========================================================================
template <typename T>
struct LUFactor {
    mtl::mat::dense2D<T> LU;
    std::vector<std::size_t> pivot;
    std::size_t n;

    LUFactor(const mtl::mat::dense2D<T>& A)
        : LU(A.num_rows(), A.num_cols()), n(A.num_rows())
    {
        // Copy A into LU (lu_factor is in-place)
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                LU(r, c) = A(r, c);
        int info = mtl::lu_factor(LU, pivot);
        if (info != 0)
            throw std::runtime_error("lu: singular matrix (zero pivot at row " +
                                     std::to_string(info - 1) + ")");
    }
};

// ===========================================================================
// CholeskyFactor<T> — wraps a Cholesky factorization (lower triangular L)
// Returned by mtl5.cholesky(A); supports .solve(b) for repeated SPD solves.
// ===========================================================================
template <typename T>
struct CholeskyFactor {
    mtl::mat::dense2D<T> L;
    std::size_t n;

    CholeskyFactor(const mtl::mat::dense2D<T>& A)
        : L(A.num_rows(), A.num_cols()), n(A.num_rows())
    {
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                L(r, c) = A(r, c);
        int info = mtl::cholesky_factor(L);
        if (info != 0)
            throw std::runtime_error(
                "cholesky: matrix is not symmetric positive definite "
                "(failure at row " + std::to_string(info - 1) + ")");
    }
};

// ---------------------------------------------------------------------------
// Helpers to copy a MatrixView/ndarray into an owning dense2D
// ---------------------------------------------------------------------------
template <typename T>
mtl::mat::dense2D<T> copy_to_dense(const MatrixView<T>& mv) {
    std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
    mtl::mat::dense2D<T> out(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            out(i, j) = mv.mat(i, j);
    return out;
}

template <typename T>
mtl::mat::dense2D<T> copy_ndarray_to_dense(
    nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> a)
{
    std::size_t r = a.shape(0), c = a.shape(1);
    mtl::mat::dense2D<T> out(r, c);
    const T* src = a.data();
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            out(i, j) = src[i * c + j];
    return out;
}

// ---------------------------------------------------------------------------
// register_native_dense_ops<T> — matmul, transpose, det, inv, lu, cholesky
// ---------------------------------------------------------------------------
template <typename T>
    requires std::is_floating_point_v<T>
void register_native_dense_ops(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    using Vec = mtl::vec::dense_vector<T>;
    using MV  = MatrixView<T>;
    using VV  = VectorView<T>;

    // -- matmul (matrix × matrix) -----------------------------------------
    m.def("matmul", [](const MV& A_mv, const MV& B_mv) {
        if (A_mv.mat.num_cols() != B_mv.mat.num_rows())
            throw std::invalid_argument(
                "matmul: A.num_cols must equal B.num_rows");
        Mat A = copy_to_dense(A_mv);
        Mat B = copy_to_dense(B_mv);
        Mat C(A.num_rows(), B.num_cols());
        mtl::mult(A, B, C);
        return MV(std::move(C));
    }, "A"_a, "B"_a, "Matrix-matrix multiplication: C = A @ B");

    // matmul accepting ndarray inputs
    m.def("matmul",
          [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np,
             nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> B_np) {
        if (A_np.shape(1) != B_np.shape(0))
            throw std::invalid_argument(
                "matmul: A.num_cols must equal B.num_rows");
        Mat A = copy_ndarray_to_dense<T>(A_np);
        Mat B = copy_ndarray_to_dense<T>(B_np);
        Mat C(A.num_rows(), B.num_cols());
        mtl::mult(A, B, C);
        return MV(std::move(C));
    }, "A"_a, "B"_a);

    // matrix-vector multiplication
    m.def("matvec", [](const MV& A_mv, const VV& x_vv) {
        if (A_mv.mat.num_cols() != x_vv.vec.size())
            throw std::invalid_argument(
                "matvec: A.num_cols must equal len(x)");
        Mat A = copy_to_dense(A_mv);
        Vec x(x_vv.vec.size());
        for (std::size_t i = 0; i < x_vv.vec.size(); ++i) x[i] = x_vv.vec[i];
        Vec y(A.num_rows());
        mtl::mult(A, x, y);
        return VV(std::move(y));
    }, "A"_a, "x"_a, "Matrix-vector multiplication: y = A @ x");

    // -- transpose (out-of-place copy for now) ----------------------------
    m.def("transpose", [](const MV& A_mv) {
        std::size_t r = A_mv.mat.num_rows(), c = A_mv.mat.num_cols();
        Mat AT(c, r);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < c; ++j)
                AT(j, i) = A_mv.mat(i, j);
        return MV(std::move(AT));
    }, "A"_a, "Return the transpose of A");

    // -- det (via LU factorization) ---------------------------------------
    auto compute_det = [](Mat&& LU) -> double {
        std::size_t n = LU.num_rows();
        std::vector<std::size_t> pivot;
        int info = mtl::lu_factor(LU, pivot);
        if (info != 0)
            return 0.0;
        double d = 1.0;
        std::size_t swaps = 0;
        for (std::size_t i = 0; i < n; ++i) {
            d *= static_cast<double>(LU(i, i));
            if (pivot[i] != i) ++swaps;
        }
        if (swaps % 2 == 1) d = -d;
        return d;
    };

    m.def("det", [compute_det](const MV& A_mv) -> double {
        if (A_mv.mat.num_cols() != A_mv.mat.num_rows())
            throw std::invalid_argument("det: A must be square");
        return compute_det(copy_to_dense(A_mv));
    }, "A"_a, "Compute determinant via LU factorization");

    m.def("det",
          [compute_det](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np)
          -> double {
        if (A_np.shape(0) != A_np.shape(1))
            throw std::invalid_argument("det: A must be square");
        return compute_det(copy_ndarray_to_dense<T>(A_np));
    }, "A"_a);

    // -- inv (matrix inverse) ---------------------------------------------
    m.def("inv", [](const MV& A_mv) {
        if (A_mv.mat.num_cols() != A_mv.mat.num_rows())
            throw std::invalid_argument("inv: A must be square");
        Mat A = copy_to_dense(A_mv);
        auto Ainv = mtl::inv(A);
        return MV(std::move(Ainv));
    }, "A"_a, "Compute matrix inverse via LU factorization");

    m.def("inv",
          [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np) {
        if (A_np.shape(0) != A_np.shape(1))
            throw std::invalid_argument("inv: A must be square");
        Mat A = copy_ndarray_to_dense<T>(A_np);
        auto Ainv = mtl::inv(A);
        return MV(std::move(Ainv));
    }, "A"_a);

    // -- LU factorization object ------------------------------------------
    using LUF = LUFactor<T>;
    std::string lu_name = std::string("LUFactor_") + type_suffix<T>();
    nb::class_<LUF>(m, lu_name.c_str())
        .def("solve", [](const LUF& self, const VV& b_vv) {
            if (b_vv.vec.size() != self.n)
                throw std::invalid_argument("LU.solve: dimension mismatch");
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_vv.vec[i];
            Vec x(self.n);
            mtl::lu_solve(self.LU, self.pivot, x, b);
            return VV(std::move(x));
        }, "b"_a, "Solve LUx = b for the previously factored A")
        .def("solve",
             [](const LUF& self,
                nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
            if (b_np.shape(0) != self.n)
                throw std::invalid_argument("LU.solve: dimension mismatch");
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_np.data()[i];
            Vec x(self.n);
            mtl::lu_solve(self.LU, self.pivot, x, b);
            return VV(std::move(x));
        }, "b"_a)
        .def_prop_ro("n", [](const LUF& self) { return self.n; })
        .def("__repr__", [](const LUF& self) {
            return std::string("mtl5.LUFactor_") + type_suffix<T>() +
                   "(n=" + std::to_string(self.n) + ")";
        });

    m.def("lu", [](const MV& A_mv) {
        if (A_mv.mat.num_cols() != A_mv.mat.num_rows())
            throw std::invalid_argument("lu: A must be square");
        return LUF(A_mv.mat);
    }, "A"_a, "Return LU factorization of A");

    m.def("lu",
          [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np) {
        if (A_np.shape(0) != A_np.shape(1))
            throw std::invalid_argument("lu: A must be square");
        Mat A = copy_ndarray_to_dense<T>(A_np);
        return LUF(A);
    }, "A"_a);

    // -- Cholesky factorization object ------------------------------------
    using CF = CholeskyFactor<T>;
    std::string ch_name = std::string("CholeskyFactor_") + type_suffix<T>();
    nb::class_<CF>(m, ch_name.c_str())
        .def("solve", [](const CF& self, const VV& b_vv) {
            if (b_vv.vec.size() != self.n)
                throw std::invalid_argument("cholesky.solve: dimension mismatch");
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_vv.vec[i];
            Vec x(self.n);
            mtl::cholesky_solve(self.L, x, b);
            return VV(std::move(x));
        }, "b"_a, "Solve A·x = b using the Cholesky factor")
        .def("solve",
             [](const CF& self,
                nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
            if (b_np.shape(0) != self.n)
                throw std::invalid_argument("cholesky.solve: dimension mismatch");
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_np.data()[i];
            Vec x(self.n);
            mtl::cholesky_solve(self.L, x, b);
            return VV(std::move(x));
        }, "b"_a)
        .def_prop_ro("n", [](const CF& self) { return self.n; })
        .def("__repr__", [](const CF& self) {
            return std::string("mtl5.CholeskyFactor_") + type_suffix<T>() +
                   "(n=" + std::to_string(self.n) + ")";
        });

    m.def("cholesky", [](const MV& A_mv) {
        if (A_mv.mat.num_cols() != A_mv.mat.num_rows())
            throw std::invalid_argument("cholesky: A must be square");
        return CF(A_mv.mat);
    }, "A"_a, "Return Cholesky factorization of an SPD matrix A");

    m.def("cholesky",
          [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np) {
        if (A_np.shape(0) != A_np.shape(1))
            throw std::invalid_argument("cholesky: A must be square");
        Mat A = copy_ndarray_to_dense<T>(A_np);
        return CF(A);
    }, "A"_a);
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
void register_universal_matvec(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    using Vec = mtl::vec::dense_vector<T>;
    m.def("matvec", [](const Mat& A, const Vec& x) {
        if (A.num_cols() != x.size())
            throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
        Vec y(A.num_rows());
        mtl::mult(A, x, y);
        return y;
    }, "A"_a, "x"_a,
       "Matrix-vector multiplication: y = A @ x");
}

template <typename T>
void register_universal_matmul(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    m.def("matmul", [](const Mat& A, const Mat& B) {
        if (A.num_cols() != B.num_rows())
            throw std::invalid_argument("matmul: A.num_cols must equal B.num_rows");
        Mat C(A.num_rows(), B.num_cols());
        mtl::mult(A, B, C);
        return C;
    }, "A"_a, "B"_a,
       "Matrix-matrix multiplication: C = A @ B");
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
    register_native_dense_ops<T>(m);
}

template <typename T>
void register_universal(nb::module_& m, const char* vec_factory, const char* mat_factory) {
    register_universal_vector<T>(m);
    register_universal_matrix<T>(m);
    register_universal_vector_factory<T>(m, vec_factory);
    register_universal_matrix_factory<T>(m, mat_factory);
    register_universal_norm<T>(m);
    register_universal_dot<T>(m);
    register_universal_matvec<T>(m);
    register_universal_matmul<T>(m);
    register_universal_solve<T>(m);
}

// ===========================================================================
// Sparse matrix bindings — compressed2D (CSR) for f32/f64
//
// Wraps mtl::mat::compressed2D<T> with constructors that accept the three
// CSR arrays (indptr, indices, data) from scipy.sparse.csr_matrix, and an
// extractor that returns them back. Provides matvec(x) for SpMV.
//
// MTL5's compressed2D uses size_type = std::size_t (uint64), so scipy
// matrices with int32 indices need to be converted on the boundary —
// scipy uses int32 indices by default for matrices below ~2 billion nnz.
// ===========================================================================
template <typename T>
    requires std::is_floating_point_v<T>
void register_sparse_matrix(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using size_type = typename SMat::size_type;
    using VV = VectorView<T>;
    std::string name = std::string("SparseMatrix_") + type_suffix<T>();

    nb::class_<SMat>(m, name.c_str())
        .def("__init__", [](SMat* self, std::size_t nrows, std::size_t ncols,
                            nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> indptr,
                            nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> indices,
                            nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> data) {
            // CSR layout: indptr.size() == nrows + 1, indices.size() == data.size() == nnz
            if (indptr.shape(0) != nrows + 1)
                throw std::invalid_argument("indptr length must be nrows + 1");
            if (indices.shape(0) != data.shape(0))
                throw std::invalid_argument("indices and data must have the same length");

            std::size_t nnz = data.shape(0);

            // Convert int64 indices/indptr to size_type (the existing constructor copies)
            std::vector<size_type> starts(nrows + 1);
            for (std::size_t i = 0; i <= nrows; ++i)
                starts[i] = static_cast<size_type>(indptr.data()[i]);
            std::vector<size_type> idx(nnz);
            for (std::size_t i = 0; i < nnz; ++i)
                idx[i] = static_cast<size_type>(indices.data()[i]);

            new (self) SMat(nrows, ncols, nnz, starts.data(), idx.data(), data.data());
        }, "nrows"_a, "ncols"_a, "indptr"_a, "indices"_a, "data"_a,
           "Construct a CSR sparse matrix from scipy-style arrays")
        .def_prop_ro("num_rows", &SMat::num_rows)
        .def_prop_ro("num_cols", &SMat::num_cols)
        .def_prop_ro("nnz", &SMat::nnz)
        .def_prop_ro("dtype", [](const SMat&) { return type_suffix<T>(); })
        .def_prop_ro("shape", [](const SMat& A) {
            return std::pair<std::size_t, std::size_t>(A.num_rows(), A.num_cols());
        })
        .def("to_csr_arrays", [](const SMat& A) {
            // Return (indptr, indices, data) as int64 + T NumPy arrays.
            // We allocate fresh buffers and let NumPy own them via capsule.
            std::size_t nrows = A.num_rows();
            std::size_t nnz = A.nnz();

            int64_t* ip = new int64_t[nrows + 1];
            for (std::size_t i = 0; i <= nrows; ++i)
                ip[i] = static_cast<int64_t>(A.ref_major()[i]);
            int64_t* ix = new int64_t[nnz];
            for (std::size_t i = 0; i < nnz; ++i)
                ix[i] = static_cast<int64_t>(A.ref_minor()[i]);
            T* dt = new T[nnz];
            for (std::size_t i = 0; i < nnz; ++i)
                dt[i] = A.ref_data()[i];

            std::size_t shape_ip[1] = { nrows + 1 };
            std::size_t shape_ix[1] = { nnz };
            std::size_t shape_dt[1] = { nnz };
            nb::capsule own_ip(ip, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
            nb::capsule own_ix(ix, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
            nb::capsule own_dt(dt, [](void* p) noexcept { delete[] static_cast<T*>(p); });

            auto a_ip = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(ip, 1, shape_ip, own_ip);
            auto a_ix = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(ix, 1, shape_ix, own_ix);
            auto a_dt = nb::ndarray<nb::numpy, T, nb::ndim<1>>(dt, 1, shape_dt, own_dt);
            return std::make_tuple(a_ip, a_ix, a_dt);
        }, "Return (indptr, indices, data) as NumPy arrays — scipy CSR layout")
        .def("matvec", [](const SMat& A, const VV& x) {
            if (A.num_cols() != x.vec.size())
                throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
            mtl::vec::dense_vector<T> y(A.num_rows());
            mtl::mult(A, x.vec, y);
            return VV(std::move(y));
        }, "x"_a, "Sparse matrix-vector product: y = A @ x")
        .def("matvec", [](const SMat& A,
                          nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> x_np) {
            if (A.num_cols() != x_np.shape(0))
                throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
            mtl::vec::dense_vector<T> x(x_np.shape(0));
            for (std::size_t i = 0; i < x_np.shape(0); ++i) x[i] = x_np.data()[i];
            mtl::vec::dense_vector<T> y(A.num_rows());
            mtl::mult(A, x, y);
            return VV(std::move(y));
        }, "x"_a)
        .def("__repr__", [](const SMat& A) {
            std::ostringstream os;
            os << "mtl5.SparseMatrix_" << type_suffix<T>()
               << "(shape=(" << A.num_rows() << ", " << A.num_cols() << ")"
               << ", nnz=" << A.nnz() << ")";
            return os.str();
        });
}

// ===========================================================================
// Iterative solvers and preconditioners
//
// Binds MTL5's CG, GMRES, BiCGSTAB along with ILU(0) and IC(0) preconditioners
// for compressed2D sparse matrices. All solvers return (x, info) following
// SciPy convention: info=0 on convergence, info=1 if max_iter exceeded.
//
// Optional callback receives the iteration number and current residual.
// Preconditioners expose .solve(b) for direct application and integrate
// with scipy via mtl5.sparse.as_linear_operator() on the Python side.
// ===========================================================================
template <typename T>
    requires std::is_floating_point_v<T>
void register_sparse_solvers(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Vec  = mtl::vec::dense_vector<T>;
    using VV   = VectorView<T>;
    using BasicIter = mtl::itl::basic_iteration<T>;

    // Helper: extract a fresh dense_vector copy from a VectorView or ndarray
    auto vec_from_view = [](const VV& v) {
        Vec out(v.vec.size());
        for (std::size_t i = 0; i < v.vec.size(); ++i) out[i] = v.vec[i];
        return out;
    };

    // ----- Conjugate Gradient -------------------------------------------
    m.def("_sparse_cg",
          [vec_from_view](const SMat& A, const VV& b_vv, T rtol, int maxiter) {
        if (A.num_rows() != A.num_cols())
            throw std::invalid_argument("cg: A must be square");
        if (A.num_rows() != b_vv.vec.size())
            throw std::invalid_argument("cg: A.num_rows must equal len(b)");

        Vec b = vec_from_view(b_vv);
        Vec x(A.num_rows(), T{0});  // initial guess: zero
        // Compute r0 = b - A*0 = b for the iteration controller
        BasicIter iter(b, maxiter, rtol);
        mtl::itl::pc::identity<SMat> M(A);
        int info = mtl::itl::cg(A, x, b, M, iter);

        return std::make_tuple(VV(std::move(x)), info, iter.iterations(),
                               static_cast<double>(iter.resid()));
    }, "A"_a, "b"_a, "rtol"_a, "maxiter"_a,
       "Internal CG kernel — use mtl5.sparse.cg() for the public API");

    // ----- GMRES --------------------------------------------------------
    m.def("_sparse_gmres",
          [vec_from_view](const SMat& A, const VV& b_vv,
                          T rtol, int maxiter, int restart) {
        if (A.num_rows() != A.num_cols())
            throw std::invalid_argument("gmres: A must be square");
        if (A.num_rows() != b_vv.vec.size())
            throw std::invalid_argument("gmres: A.num_rows must equal len(b)");

        Vec b = vec_from_view(b_vv);
        Vec x(A.num_rows(), T{0});
        BasicIter iter(b, maxiter, rtol);
        mtl::itl::pc::identity<SMat> M(A);
        int info = mtl::itl::gmres(A, x, b, M, iter, restart);

        return std::make_tuple(VV(std::move(x)), info, iter.iterations(),
                               static_cast<double>(iter.resid()));
    }, "A"_a, "b"_a, "rtol"_a, "maxiter"_a, "restart"_a,
       "Internal GMRES kernel — use mtl5.sparse.gmres() for the public API");

    // ----- BiCGSTAB -----------------------------------------------------
    m.def("_sparse_bicgstab",
          [vec_from_view](const SMat& A, const VV& b_vv, T rtol, int maxiter) {
        if (A.num_rows() != A.num_cols())
            throw std::invalid_argument("bicgstab: A must be square");
        if (A.num_rows() != b_vv.vec.size())
            throw std::invalid_argument("bicgstab: A.num_rows must equal len(b)");

        Vec b = vec_from_view(b_vv);
        Vec x(A.num_rows(), T{0});
        BasicIter iter(b, maxiter, rtol);
        mtl::itl::pc::identity<SMat> M(A);
        int info = mtl::itl::bicgstab(A, x, b, M, iter);

        return std::make_tuple(VV(std::move(x)), info, iter.iterations(),
                               static_cast<double>(iter.resid()));
    }, "A"_a, "b"_a, "rtol"_a, "maxiter"_a,
       "Internal BiCGSTAB kernel — use mtl5.sparse.bicgstab() for the public API");
}

// ===========================================================================
// Preconditioner bindings — wrap ILU(0) and IC(0) as Python objects with
// .solve(b) so they can be used both standalone and as scipy LinearOperators.
//
// Python-side wrappers store the factor dimension (n) alongside the
// underlying preconditioner so that mismatched RHS lengths fail with a clean
// Python ValueError before reaching the native implementation.
// ===========================================================================
template <typename PC, typename T>
struct PreconditionerWrapper {
    PC pc;
    std::size_t n;

    PreconditionerWrapper(const mtl::mat::compressed2D<T>& A)
        : pc(A), n(A.num_rows()) {}
};

template <typename T>
    requires std::is_floating_point_v<T>
void register_preconditioners(nb::module_& m) {
    using SMat = mtl::mat::compressed2D<T>;
    using Vec  = mtl::vec::dense_vector<T>;
    using VV   = VectorView<T>;

    // ----- ILU(0) -------------------------------------------------------
    using ILUWrap = PreconditionerWrapper<mtl::itl::pc::ilu_0<T>, T>;
    std::string ilu_name = std::string("ILU0_") + type_suffix<T>();
    nb::class_<ILUWrap>(m, ilu_name.c_str())
        .def("__init__", [](ILUWrap* self, const SMat& A) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("ILU0: matrix must be square");
            new (self) ILUWrap(A);
        }, "A"_a, "Construct ILU(0) preconditioner from a square CSR matrix")
        .def_prop_ro("n", [](const ILUWrap& self) { return self.n; })
        .def("solve", [](const ILUWrap& self, const VV& b_vv) {
            if (b_vv.vec.size() != self.n)
                throw std::invalid_argument(
                    "ILU0.solve: RHS length " + std::to_string(b_vv.vec.size()) +
                    " does not match factor size " + std::to_string(self.n));
            Vec x(self.n);
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_vv.vec[i];
            self.pc.solve(x, b);
            return VV(std::move(x));
        }, "b"_a, "Apply preconditioner: solve (LU)·x = b")
        .def("solve",
             [](const ILUWrap& self,
                nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
            if (b_np.shape(0) != self.n)
                throw std::invalid_argument(
                    "ILU0.solve: RHS length " + std::to_string(b_np.shape(0)) +
                    " does not match factor size " + std::to_string(self.n));
            Vec x(self.n);
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_np.data()[i];
            self.pc.solve(x, b);
            return VV(std::move(x));
        }, "b"_a)
        .def("__repr__", [ilu_name](const ILUWrap& self) {
            return std::string("mtl5.sparse.") + ilu_name +
                   "(n=" + std::to_string(self.n) + ")";
        });

    // ----- IC(0) --------------------------------------------------------
    using ICWrap = PreconditionerWrapper<mtl::itl::pc::ic_0<T>, T>;
    std::string ic_name = std::string("IC0_") + type_suffix<T>();
    nb::class_<ICWrap>(m, ic_name.c_str())
        .def("__init__", [](ICWrap* self, const SMat& A) {
            if (A.num_rows() != A.num_cols())
                throw std::invalid_argument("IC0: matrix must be square");
            new (self) ICWrap(A);
        }, "A"_a, "Construct IC(0) preconditioner from a square SPD CSR matrix")
        .def_prop_ro("n", [](const ICWrap& self) { return self.n; })
        .def("solve", [](const ICWrap& self, const VV& b_vv) {
            if (b_vv.vec.size() != self.n)
                throw std::invalid_argument(
                    "IC0.solve: RHS length " + std::to_string(b_vv.vec.size()) +
                    " does not match factor size " + std::to_string(self.n));
            Vec x(self.n);
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_vv.vec[i];
            self.pc.solve(x, b);
            return VV(std::move(x));
        }, "b"_a, "Apply preconditioner: solve (L·L^T)·x = b")
        .def("solve",
             [](const ICWrap& self,
                nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
            if (b_np.shape(0) != self.n)
                throw std::invalid_argument(
                    "IC0.solve: RHS length " + std::to_string(b_np.shape(0)) +
                    " does not match factor size " + std::to_string(self.n));
            Vec x(self.n);
            Vec b(self.n);
            for (std::size_t i = 0; i < self.n; ++i) b[i] = b_np.data()[i];
            self.pc.solve(x, b);
            return VV(std::move(x));
        }, "b"_a)
        .def("__repr__", [ic_name](const ICWrap& self) {
            return std::string("mtl5.sparse.") + ic_name +
                   "(n=" + std::to_string(self.n) + ")";
        });
}

// ===========================================================================
// Module definition
// ===========================================================================
NB_MODULE(_core, m) {
    m.doc() = "MTL5 Python bindings — nanobind core module";

    m.attr("__version__") = "0.1.0";

    // ----- Device & backend management ---------------------------------------
    m.def("devices", []() {
        return std::vector<std::string>{"cpu"};
        // Future: enumerate KPU devices
    }, "List available execution devices");

    m.def("backends", []() {
        // Reports the dispatch hierarchy from highest to lowest preference.
        // KPU is the principal target; CPU reference is the portable fallback.
        std::vector<std::string> b;
#ifdef MTL5_HAS_KPU
        b.push_back("kpu");
#endif
#ifdef MTL5_HAS_BLAS
        b.push_back("blas");
#endif
        b.push_back("reference");
        return b;
    }, "List available compute backends in dispatch order (KPU > BLAS > reference)");

    m.def("get_backend", []() {
#ifdef MTL5_HAS_KPU
        return std::string("kpu");
#elif defined(MTL5_HAS_BLAS)
        return std::string("blas");
#else
        return std::string("reference");
#endif
    }, "Return the currently active compute backend");

    m.def("set_backend", [](const std::string& name) {
        // Stub: backend selection is currently compile-time only
        if (name != "cpu" && name != "reference" && name != "blas" && name != "kpu")
            throw std::runtime_error("Unknown backend: '" + name +
                                     "'. Valid: cpu, reference, blas, kpu");
        if (name == "kpu")
            throw std::runtime_error("KPU backend not yet available. "
                                     "Hardware support is in development.");
        // No-op for cpu/reference/blas — selected at compile time
    }, "name"_a, "Set the active compute backend (currently compile-time only)");

    // ----- Native C++ types (zero-copy via nb::ndarray) ----------------------
    register_native_with_solve<float>(m);     // f32
    register_native_with_solve<double>(m);    // f64
    register_native<int32_t>(m);              // i32
    register_native<int64_t>(m);              // i64

    // ----- Sparse matrices (CSR via mtl::compressed2D) -----------------------
    register_sparse_matrix<float>(m);
    register_sparse_matrix<double>(m);

    // ----- Iterative solvers (CG, GMRES, BiCGSTAB) ---------------------------
    register_sparse_solvers<float>(m);
    register_sparse_solvers<double>(m);

    // ----- Preconditioners (ILU0, IC0) ---------------------------------------
    register_preconditioners<float>(m);
    register_preconditioners<double>(m);

    // ----- Universal number types (copy-converting from float64) -------------
    // Standard IEEE-style cfloat configurations
    register_universal<fp8>(m, "vector_fp8", "matrix_fp8");
    register_universal<fp16>(m, "vector_fp16", "matrix_fp16");

    // Posit types — tapered precision, ideal for ML on KPU
    register_universal<posit8>(m, "vector_posit8", "matrix_posit8");
    register_universal<posit16>(m, "vector_posit16", "matrix_posit16");
    register_universal<posit32>(m, "vector_posit32", "matrix_posit32");
    register_universal<posit64>(m, "vector_posit64", "matrix_posit64");

    // Fixed-point types
    register_universal<fixpnt8>(m, "vector_fixpnt8", "matrix_fixpnt8");
    register_universal<fixpnt16>(m, "vector_fixpnt16", "matrix_fixpnt16");

    // Logarithmic number system
    register_universal<lns16>(m, "vector_lns16", "matrix_lns16");
    register_universal<lns32>(m, "vector_lns32", "matrix_lns32");
}
