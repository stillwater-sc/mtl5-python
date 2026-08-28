// mtl5-python -- complex element types (c64 = complex64, c128 = complex128).
//
// MTL5 has complex support designed in (`magnitude_trait<std::complex<T>>`),
// but it reaches only part of the library. The bindings here cover exactly the
// operations that were verified to compile AND to give the right answer on
// complex input; everything else is deliberately absent so that a wrong result
// is never reachable from Python. The boundary, as measured:
//
//   bound      containers, vector/matrix factories, one/two/infinity/frobenius
//              norms, dot, dot_real, matvec, matmul, transpose, conj, adjoint,
//              solve, lu, inv, ldlt (guarded), is_hermitian, is_symmetric
//   absent     cholesky and qr      -- compare a complex against a complex
//                                      (`operator<=`, `operator>`) where a
//                                      magnitude is meant, so they do not
//                                      compile at all
//              eigenvalue and svd   -- would need complex<complex> internally
//              cg/gmres/bicgstab    -- the Krylov layer is real-only
//
// Three semantics are easy to get wrong and are pinned by tests:
//
//   `dot` is Hermitian -- sum(conj(a[i]) * b[i]), conjugating the FIRST
//   argument. That is NumPy's `vdot`, NOT NumPy's `dot`. `dot_real` is the
//   unconjugated sum(a[i] * b[i]), which is what `np.dot` does for 1-D complex.
//   Both are exposed because both are wanted, and the names say which is which.
//
//   `.T` is a plain transpose, not a conjugate transpose. MTL5's `trans` does
//   not conjugate. `.H` / `adjoint()` is the conjugate transpose. For real
//   element types the two coincide, which is exactly why the distinction has to
//   be explicit here.
//
//   `ldlt` is LDL^T, not LDL^H. On a complex *symmetric* matrix (A == A^T) it
//   is correct. On a *Hermitian* matrix (A == A^H) it returns info=0 and a
//   wrong answer -- silently. Since Hermitian is the case most callers reach
//   for, the binding refuses Hermitian-but-not-symmetric input rather than
//   passing it through. See the guard in register_complex_ldlt.

#include "mtl5_types.hpp"

#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <mtl/operation/cholesky.hpp>
#include <mtl/operation/conj.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/inv.hpp>
#include <mtl/operation/ldlt.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/matrix_properties.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/norms.hpp>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace nb::literals;

namespace {

// Deep copies. The views may borrow NumPy memory, and every kernel below wants
// an owning container it is free to overwrite.
template <typename T>
mtl::mat::dense2D<T> owned_copy(const MatrixView<T>& mv) {
    const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
    mtl::mat::dense2D<T> out(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            out(i, j) = mv.mat(i, j);
    return out;
}

template <typename T>
mtl::vec::dense_vector<T> owned_copy(const VectorView<T>& vv) {
    mtl::vec::dense_vector<T> out(vv.vec.size());
    for (std::size_t i = 0; i < vv.vec.size(); ++i) out[i] = vv.vec[i];
    return out;
}

template <typename T>
void require_square(const mtl::mat::dense2D<T>& A, const char* who) {
    if (A.num_rows() != A.num_cols())
        throw std::invalid_argument(std::string(who) + ": matrix must be square");
}

// ---------------------------------------------------------------------------
// LU factorization object, mirroring LUFactor_f64 but for complex elements.
// ---------------------------------------------------------------------------
template <typename T>
struct ComplexLU {
    mtl::mat::dense2D<T> LU;
    std::vector<std::size_t> pivot;
    std::size_t n;

    explicit ComplexLU(mtl::mat::dense2D<T> A) : LU(std::move(A)), n(LU.num_rows()) {
        int info = mtl::lu_factor(LU, pivot);
        if (info != 0)
            throw std::runtime_error("lu: singular matrix (zero pivot at row " +
                                     std::to_string(info - 1) + ")");
    }
};

// ---------------------------------------------------------------------------
// Containers
// ---------------------------------------------------------------------------
template <typename T>
void register_complex_vector(nb::module_& m) {
    using VV = VectorView<T>;
    const std::string name = std::string("DenseVector_") + type_suffix<T>();

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
            // nb::handle for the keep-alive owner -- see the note on
            // VectorView::to_numpy in mtl5_module.cpp.
            VV& vv = nb::cast<VV&>(self);
            std::size_t shape[1] = { vv.vec.size() };
            return nb::ndarray<nb::numpy, T, nb::ndim<1>>(
                vv.vec.data(), 1, shape, self);
        }, "Return a zero-copy NumPy array view of this vector")
        .def("copy", [](const VV& vv) {
            nogil guard;
            return VV(owned_copy(vv));
        }, "Return an owning copy of this vector")
        .def("conj", [](const VV& vv) {
            mtl::vec::dense_vector<T> out(vv.vec.size());
            {
                nogil guard;
                for (std::size_t i = 0; i < vv.vec.size(); ++i)
                    out[i] = std::conj(vv.vec[i]);
            }
            return VV(std::move(out));
        }, "Elementwise complex conjugate")
        .def_prop_ro("real", [](const VV& vv) {
            using R = typename T::value_type;
            mtl::vec::dense_vector<R> out(vv.vec.size());
            {
                nogil guard;
                for (std::size_t i = 0; i < vv.vec.size(); ++i)
                    out[i] = vv.vec[i].real();
            }
            return VectorView<R>(std::move(out));
        }, "Real part, as a new real-valued vector")
        .def_prop_ro("imag", [](const VV& vv) {
            using R = typename T::value_type;
            mtl::vec::dense_vector<R> out(vv.vec.size());
            {
                nogil guard;
                for (std::size_t i = 0; i < vv.vec.size(); ++i)
                    out[i] = vv.vec[i].imag();
            }
            return VectorView<R>(std::move(out));
        }, "Imaginary part, as a new real-valued vector")
        .def("__repr__", [](const VV& vv) {
            std::ostringstream os;
            os << "mtl5.DenseVector_" << type_suffix<T>() << "([";
            for (std::size_t i = 0; i < vv.vec.size(); ++i) {
                if (i) os << ", ";
                if (i >= 6 && vv.vec.size() > 8) { os << "..."; break; }
                os << vv.vec[i].real()
                   << (vv.vec[i].imag() < 0 ? "-" : "+")
                   << std::abs(vv.vec[i].imag()) << "j";
            }
            os << "], size=" << vv.vec.size()
               << ", device='" << vv.device_name << "'"
               << (vv.is_view() ? ", view" : "") << ")";
            return os.str();
        });
}

template <typename T>
void register_complex_matrix(nb::module_& m) {
    using MV = MatrixView<T>;
    using VV = VectorView<T>;
    const std::string name = std::string("DenseMatrix_") + type_suffix<T>();

    nb::class_<MV>(m, name.c_str())
        .def_prop_ro("num_rows", [](const MV& mv) { return mv.mat.num_rows(); })
        .def_prop_ro("num_cols", [](const MV& mv) { return mv.mat.num_cols(); })
        .def_prop_ro("shape", [](const MV& mv) {
            return std::pair<std::size_t, std::size_t>(
                mv.mat.num_rows(), mv.mat.num_cols());
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
            MV& mv = nb::cast<MV&>(self);
            std::size_t shape[2] = { mv.mat.num_rows(), mv.mat.num_cols() };
            return nb::ndarray<nb::numpy, T, nb::ndim<2>>(
                mv.mat.data(), 2, shape, self);
        }, "Return a zero-copy NumPy array view of this matrix")
        // .T is the PLAIN transpose. It does not conjugate -- see the file
        // header. Use .H for the adjoint.
        .def_prop_ro("T", [](const MV& mv) {
            const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
            mtl::mat::dense2D<T> out(c, r);
            {
                nogil guard;
                for (std::size_t i = 0; i < r; ++i)
                    for (std::size_t j = 0; j < c; ++j)
                        out(j, i) = mv.mat(i, j);
            }
            return MV(std::move(out));
        }, "Plain transpose (does NOT conjugate -- use .H for the adjoint)")
        .def_prop_ro("H", [](const MV& mv) {
            const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
            mtl::mat::dense2D<T> out(c, r);
            {
                nogil guard;
                for (std::size_t i = 0; i < r; ++i)
                    for (std::size_t j = 0; j < c; ++j)
                        out(j, i) = std::conj(mv.mat(i, j));
            }
            return MV(std::move(out));
        }, "Conjugate transpose (adjoint)")
        .def("conj", [](const MV& mv) {
            const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
            mtl::mat::dense2D<T> out(r, c);
            {
                nogil guard;
                for (std::size_t i = 0; i < r; ++i)
                    for (std::size_t j = 0; j < c; ++j)
                        out(i, j) = std::conj(mv.mat(i, j));
            }
            return MV(std::move(out));
        }, "Elementwise complex conjugate")
        .def_prop_ro("real", [](const MV& mv) {
            using R = typename T::value_type;
            const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
            mtl::mat::dense2D<R> out(r, c);
            {
                nogil guard;
                for (std::size_t i = 0; i < r; ++i)
                    for (std::size_t j = 0; j < c; ++j)
                        out(i, j) = mv.mat(i, j).real();
            }
            return MatrixView<R>(std::move(out));
        }, "Real part, as a new real-valued matrix")
        .def_prop_ro("imag", [](const MV& mv) {
            using R = typename T::value_type;
            const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
            mtl::mat::dense2D<R> out(r, c);
            {
                nogil guard;
                for (std::size_t i = 0; i < r; ++i)
                    for (std::size_t j = 0; j < c; ++j)
                        out(i, j) = mv.mat(i, j).imag();
            }
            return MatrixView<R>(std::move(out));
        }, "Imaginary part, as a new real-valued matrix")
        .def("__matmul__", [](const MV& A, const MV& B) {
            if (A.mat.num_cols() != B.mat.num_rows())
                throw std::invalid_argument("matmul: A.num_cols != B.num_rows");
            mtl::mat::dense2D<T> C(A.mat.num_rows(), B.mat.num_cols());
            {
                nogil guard;
                mtl::mult(owned_copy(A), owned_copy(B), C);
            }
            return MV(std::move(C));
        })
        .def("__matmul__", [](const MV& A, const VV& x) {
            if (A.mat.num_cols() != x.vec.size())
                throw std::invalid_argument("matmul: A.num_cols != len(x)");
            mtl::vec::dense_vector<T> y(A.mat.num_rows());
            {
                nogil guard;
                mtl::mult(owned_copy(A), owned_copy(x), y);
            }
            return VV(std::move(y));
        })
        .def("copy", [](const MV& mv) {
            nogil guard;
            return MV(owned_copy(mv));
        }, "Return an owning copy of this matrix")
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
// Factories -- zero-copy views over NumPy complex64/complex128 buffers
// ---------------------------------------------------------------------------
template <typename T>
void register_complex_factories(nb::module_& m) {
    m.def("vector", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        return VectorView<T>(a.shape(0), a.data(), nb::cast(a));
    }, nb::arg("a").noconvert(), "Create a zero-copy MTL5 vector view of a 1-D complex NumPy array");

    m.def("vector_copy", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a) {
        const std::size_t n = a.shape(0);
        const T* src = a.data();
        mtl::vec::dense_vector<T> v(n);
        {
            nogil guard;
            for (std::size_t i = 0; i < n; ++i) v[i] = src[i];
        }
        return VectorView<T>(std::move(v));
    }, nb::arg("a").noconvert(), "Create an owning MTL5 vector (copies from a complex NumPy array)");

    m.def("matrix", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        return MatrixView<T>(a.shape(0), a.shape(1), a.data(), nb::cast(a));
    }, nb::arg("a").noconvert(), "Create a zero-copy MTL5 matrix view of a 2-D complex NumPy array");

    m.def("matrix_copy", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        const std::size_t r = a.shape(0), c = a.shape(1);
        const T* src = a.data();
        mtl::mat::dense2D<T> M(r, c);
        {
            nogil guard;
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t j = 0; j < c; ++j)
                    M(i, j) = src[i * c + j];
        }
        return MatrixView<T>(std::move(M));
    }, nb::arg("a").noconvert(), "Create an owning MTL5 matrix (copies from a complex NumPy array)");
}

// ---------------------------------------------------------------------------
// Norms and inner products
//
// Every norm of a complex container is a REAL number -- MTL5 returns the
// magnitude type, and these return a Python float, not a complex with a zero
// imaginary part.
// ---------------------------------------------------------------------------
template <typename T>
void register_complex_norms(nb::module_& m) {
    // The ndarray overloads matter for more than convenience. `norm` and `dot`
    // already have nb::ndarray<double> overloads for real input; without a
    // complex one registered, a complex array could reach those through
    // nanobind's second (converting) pass and quietly lose its imaginary part.
    // Registering these means complex matches in the first, no-convert pass.
    m.def("norm", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                     int ord) -> double {
        if (ord != 1 && ord != 2 && ord != -1)
            throw std::invalid_argument("ord must be 1, 2, or -1 (infinity)");
        const std::size_t n = a.shape(0);
        T* data = const_cast<T*>(a.data());
        nogil guard;
        auto v = mtl::vec::dense_vector<T>(n, data);
        if (ord == 1) return static_cast<double>(mtl::one_norm(v));
        if (ord == 2) return static_cast<double>(mtl::two_norm(v));
        return static_cast<double>(mtl::infinity_norm(v));
    }, "x"_a, "ord"_a = 2);

    m.def("norm", [](const VectorView<T>& x, int ord) -> double {
        if (ord != 1 && ord != 2 && ord != -1)
            throw std::invalid_argument("ord must be 1, 2, or -1 (infinity)");
        nogil guard;
        if (ord == 1) return static_cast<double>(mtl::one_norm(x.vec));
        if (ord == 2) return static_cast<double>(mtl::two_norm(x.vec));
        return static_cast<double>(mtl::infinity_norm(x.vec));
    }, "x"_a, "ord"_a = 2,
       "Vector norm. 1 = sum|z|, 2 = Euclidean, -1 = max|z|. Always real.");

    m.def("norm", [](const MatrixView<T>& A, int ord) -> double {
        if (ord != 1 && ord != 2 && ord != -1)
            throw std::invalid_argument("ord must be 1, 2, or -1 (infinity)");
        nogil guard;
        if (ord == 1) return static_cast<double>(mtl::one_norm(A.mat));
        if (ord == 2) return static_cast<double>(mtl::frobenius_norm(A.mat));
        return static_cast<double>(mtl::infinity_norm(A.mat));
    }, "A"_a, "ord"_a = 2,
       "Matrix norm. 1 = max column sum, 2 = Frobenius, -1 = max row sum.");

    m.def("frobenius_norm", [](const MatrixView<T>& A) -> double {
        nogil guard;
        return static_cast<double>(mtl::frobenius_norm(A.mat));
    }, "A"_a, "Frobenius norm, sqrt(sum |a_ij|^2). Always real.");
}

template <typename T>
void register_complex_dot(nb::module_& m) {
    m.def("dot", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                    nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b) -> T {
        if (a.shape(0) != b.shape(0))
            throw std::invalid_argument("vectors must have the same length");
        const std::size_t n = a.shape(0);
        T* ap = const_cast<T*>(a.data());
        T* bp = const_cast<T*>(b.data());
        nogil guard;
        return mtl::dot(mtl::vec::dense_vector<T>(n, ap),
                        mtl::vec::dense_vector<T>(n, bp));
    }, "a"_a, "b"_a);

    m.def("dot_real", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                         nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b) -> T {
        if (a.shape(0) != b.shape(0))
            throw std::invalid_argument("vectors must have the same length");
        const std::size_t n = a.shape(0);
        T* ap = const_cast<T*>(a.data());
        T* bp = const_cast<T*>(b.data());
        nogil guard;
        return mtl::dot_real(mtl::vec::dense_vector<T>(n, ap),
                             mtl::vec::dense_vector<T>(n, bp));
    }, "a"_a, "b"_a);

    m.def("dot", [](const VectorView<T>& a, const VectorView<T>& b) -> T {
        if (a.vec.size() != b.vec.size())
            throw std::invalid_argument("vectors must have the same length");
        nogil guard;
        return mtl::dot(a.vec, b.vec);
    }, "a"_a, "b"_a,
       "Hermitian inner product sum(conj(a[i]) * b[i]) -- conjugates the FIRST "
       "argument, matching numpy.vdot. For the unconjugated product (what "
       "numpy.dot does for 1-D complex) use mtl5.dot_real.");

    m.def("dot_real", [](const VectorView<T>& a, const VectorView<T>& b) -> T {
        if (a.vec.size() != b.vec.size())
            throw std::invalid_argument("vectors must have the same length");
        nogil guard;
        return mtl::dot_real(a.vec, b.vec);
    }, "a"_a, "b"_a,
       "Unconjugated bilinear product sum(a[i] * b[i]), matching numpy.dot.");
}

// ---------------------------------------------------------------------------
// Products, solves, inverse
// ---------------------------------------------------------------------------
template <typename T>
void register_complex_ops(nb::module_& m) {
    using Mat = mtl::mat::dense2D<T>;
    using Vec = mtl::vec::dense_vector<T>;
    using MV  = MatrixView<T>;
    using VV  = VectorView<T>;

    m.def("matmul", [](const MV& A, const MV& B) {
        if (A.mat.num_cols() != B.mat.num_rows())
            throw std::invalid_argument("matmul: A.num_cols must equal B.num_rows");
        Mat C(A.mat.num_rows(), B.mat.num_cols());
        {
            nogil guard;
            mtl::mult(owned_copy(A), owned_copy(B), C);
        }
        return MV(std::move(C));
    }, "A"_a, "B"_a, "Matrix-matrix multiplication: C = A @ B");

    m.def("matvec", [](const MV& A, const VV& x) {
        if (A.mat.num_cols() != x.vec.size())
            throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
        Vec y(A.mat.num_rows());
        {
            nogil guard;
            mtl::mult(owned_copy(A), owned_copy(x), y);
        }
        return VV(std::move(y));
    }, "A"_a, "x"_a, "Matrix-vector multiplication: y = A @ x");

    m.def("transpose", [](const MV& mv) {
        const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
        Mat out(c, r);
        {
            nogil guard;
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t j = 0; j < c; ++j)
                    out(j, i) = mv.mat(i, j);
        }
        return MV(std::move(out));
    }, "A"_a, "Plain transpose (does NOT conjugate -- use mtl5.adjoint)");

    m.def("adjoint", [](const MV& mv) {
        const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
        Mat out(c, r);
        {
            nogil guard;
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t j = 0; j < c; ++j)
                    out(j, i) = std::conj(mv.mat(i, j));
        }
        return MV(std::move(out));
    }, "A"_a, "Conjugate transpose A^H");

    m.def("conj", [](const MV& mv) {
        const std::size_t r = mv.mat.num_rows(), c = mv.mat.num_cols();
        Mat out(r, c);
        {
            nogil guard;
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t j = 0; j < c; ++j)
                    out(i, j) = std::conj(mv.mat(i, j));
        }
        return MV(std::move(out));
    }, "A"_a, "Elementwise complex conjugate");

    m.def("conj", [](const VV& vv) {
        Vec out(vv.vec.size());
        {
            nogil guard;
            for (std::size_t i = 0; i < vv.vec.size(); ++i)
                out[i] = std::conj(vv.vec[i]);
        }
        return VV(std::move(out));
    }, "x"_a, "Elementwise complex conjugate");

    m.def("solve", [](const MV& A_mv, const VV& b_vv) {
        require_square(A_mv.mat, "solve");
        const std::size_t n = A_mv.mat.num_rows();
        if (b_vv.vec.size() != n)
            throw std::invalid_argument("b length must match A dimensions");
        Vec x(n);
        {
            nogil guard;
            Mat A = owned_copy(A_mv);
            std::vector<std::size_t> pivot;
            int info = mtl::lu_factor(A, pivot);
            if (info != 0)
                throw std::runtime_error("Singular matrix (pivot " +
                                         std::to_string(info) + " is zero)");
            mtl::lu_solve(A, pivot, x, owned_copy(b_vv));
        }
        return VV(std::move(x));
    }, "A"_a, "b"_a, "Solve A x = b by complex LU with partial pivoting");

    m.def("inv", [](const MV& A_mv) {
        require_square(A_mv.mat, "inv");
        Mat out(A_mv.mat.num_rows(), A_mv.mat.num_cols());
        {
            nogil guard;
            out = mtl::inv(owned_copy(A_mv));
        }
        return MV(std::move(out));
    }, "A"_a, "Explicit matrix inverse (prefer solve() for a single right-hand side)");

    m.def("is_hermitian", [](const MV& A) {
        nogil guard;
        return mtl::is_hermitian(A.mat);
    }, "A"_a, "True if A == A^H (conjugate transpose)");

    m.def("is_symmetric", [](const MV& A) {
        nogil guard;
        return mtl::is_symmetric(A.mat);
    }, "A"_a, "True if A == A^T (plain transpose -- for complex this is NOT "
              "the same as Hermitian)");
}

// ---------------------------------------------------------------------------
// LU factorization object
// ---------------------------------------------------------------------------
template <typename T>
void register_complex_lu(nb::module_& m) {
    using LU = ComplexLU<T>;
    const std::string name = std::string("LUFactor_") + type_suffix<T>();

    nb::class_<LU>(m, name.c_str())
        .def_prop_ro("shape", [](const LU& f) {
            return std::pair<std::size_t, std::size_t>(f.n, f.n);
        })
        .def_prop_ro("dtype", [](const LU&) { return type_suffix<T>(); })
        .def("solve", [](const LU& f, const VectorView<T>& b) {
            if (b.vec.size() != f.n)
                throw std::invalid_argument("b length must match the factorization");
            mtl::vec::dense_vector<T> x(f.n);
            {
                nogil guard;
                mtl::lu_solve(f.LU, f.pivot, x, owned_copy(b));
            }
            return VectorView<T>(std::move(x));
        }, "b"_a, "Solve A x = b reusing this factorization")
        .def("__repr__", [](const LU& f) {
            std::ostringstream os;
            os << "mtl5.LUFactor_" << type_suffix<T>()
               << "(shape=(" << f.n << ", " << f.n << "))";
            return os.str();
        });

    m.def("lu", [](const MatrixView<T>& A) {
        require_square(A.mat, "lu");
        nogil guard;
        return LU(owned_copy(A));
    }, "A"_a, "LU factorization with partial pivoting, reusable across solves");
}

// ---------------------------------------------------------------------------
// Hermitian Cholesky, A = L L^H
//
// MTL5's plain `cholesky_factor` computes A = L L^T and orders the diagonal to
// test positive definiteness, neither of which means anything for a complex
// element type -- it static_asserts against complex and points here. This is
// upstream's answer to stillwater-sc/mtl5#353, and it also accepts real
// symmetric input, so it is a superset rather than a parallel path.
// ---------------------------------------------------------------------------
template <typename T>
struct ComplexCholesky {
    mtl::mat::dense2D<T> L;
    std::size_t n;

    explicit ComplexCholesky(mtl::mat::dense2D<T> A) : L(std::move(A)), n(L.num_rows()) {
        const int info = mtl::cholesky_h_factor(L);
        if (info == mtl::CHOLESKY_NOT_HERMITIAN)
            throw std::invalid_argument(
                "cholesky: the matrix is not Hermitian -- its diagonal has a "
                "non-real entry, which A = L L^H cannot produce. Use "
                "mtl5.solve() for a general complex system.");
        if (info != 0)
            throw std::runtime_error(
                "cholesky: matrix is not Hermitian positive definite (failure at "
                "row " + std::to_string(info - 1) + ")");
    }
};

template <typename T>
void register_complex_cholesky(nb::module_& m) {
    using CF = ComplexCholesky<T>;
    const std::string name = std::string("CholeskyFactor_") + type_suffix<T>();

    nb::class_<CF>(m, name.c_str())
        .def_prop_ro("n", [](const CF& f) { return f.n; })
        .def_prop_ro("dtype", [](const CF&) { return type_suffix<T>(); })
        .def_prop_ro("shape", [](const CF& f) {
            return std::pair<std::size_t, std::size_t>(f.n, f.n);
        })
        .def("solve", [](const CF& f, const VectorView<T>& b) {
            if (b.vec.size() != f.n)
                throw std::invalid_argument("cholesky.solve: dimension mismatch");
            mtl::vec::dense_vector<T> x(f.n);
            {
                nogil guard;
                mtl::cholesky_h_solve(f.L, x, owned_copy(b));
            }
            return VectorView<T>(std::move(x));
        }, "b"_a, "Solve A x = b using the Hermitian Cholesky factor")
        .def("__repr__", [name](const CF& f) {
            return "mtl5." + name + "(n=" + std::to_string(f.n) + ")";
        });

    m.def("cholesky", [](const MatrixView<T>& A) {
        require_square(A.mat, "cholesky");
        nogil guard;
        return CF(owned_copy(A));
    }, "A"_a,
       "Cholesky factorization A = L L^H of a Hermitian positive definite "
       "matrix. Complex uses MTL5's cholesky_h; the plain L L^T form is not "
       "meaningful for complex elements.");

    m.def("cholesky", [](nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> a) {
        if (a.shape(0) != a.shape(1))
            throw std::invalid_argument("cholesky: A must be square");
        const std::size_t n = a.shape(0);
        const T* src = a.data();
        mtl::mat::dense2D<T> A(n, n);
        {
            nogil guard;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    A(i, j) = src[i * n + j];
        }
        return CF(std::move(A));
    }, "A"_a);
}

// ---------------------------------------------------------------------------
// LDL^T / LDL^H -- dispatched on the matrix
//
// MTL5 has both: `ldlt` computes A = L D L^T, correct for a complex *symmetric*
// matrix, and `ldlt_h` computes A = L D L^H, correct for a *Hermitian* one.
// Those are different factorizations of different matrices, and feeding either
// the other's input gives a wrong answer -- which is what stillwater-sc/mtl5#352
// was: before `ldlt_h` existed, plain `ldlt` accepted Hermitian input and
// returned x = [1.9+0.3i, -0.2+0.6i] on [[2, 1-i], [1+i, 3]] where the answer is
// [1, i], reporting info = 0 throughout.
//
// This binding used to refuse Hermitian input for exactly that reason. Now that
// upstream provides LDL^H it dispatches instead, so both cases are served and
// neither can reach the wrong kernel. A matrix that is both (a real-valued
// complex matrix) takes the Hermitian branch, where the two coincide.
// ---------------------------------------------------------------------------
template <typename T>
void register_complex_ldlt(nb::module_& m) {
    m.def("ldlt_solve", [](const MatrixView<T>& A_mv, const VectorView<T>& b_vv) {
        require_square(A_mv.mat, "ldlt_solve");
        const std::size_t n = A_mv.mat.num_rows();
        if (b_vv.vec.size() != n)
            throw std::invalid_argument("b length must match A dimensions");

        bool symmetric, hermitian;
        {
            nogil guard;
            symmetric = mtl::is_symmetric(A_mv.mat);
            hermitian = mtl::is_hermitian(A_mv.mat);
        }
        if (!symmetric && !hermitian)
            throw std::invalid_argument(
                "ldlt_solve: matrix must be complex symmetric (A == A^T) or "
                "Hermitian (A == A^H). Use mtl5.solve() for a general complex "
                "system.");

        mtl::vec::dense_vector<T> x(n);
        {
            nogil guard;
            auto LD = owned_copy(A_mv);
            // Hermitian first: when a matrix is both, the two factorizations
            // agree, and LDL^H is the one whose D is real.
            const int info = hermitian ? mtl::ldlt_h_factor(LD)
                                       : mtl::ldlt_factor(LD);
            if (info != 0)
                throw std::runtime_error(
                    "ldlt_solve: zero pivot at row " + std::to_string(info - 1));
            if (hermitian) mtl::ldlt_h_solve(LD, x, owned_copy(b_vv));
            else           mtl::ldlt_solve(LD, x, owned_copy(b_vv));
        }
        return VectorView<T>(std::move(x));
    }, "A"_a, "b"_a,
       "Solve A x = b by LDL^T for a complex symmetric matrix (A == A^T) or by "
       "LDL^H for a Hermitian one (A == A^H), chosen from the matrix. Those are "
       "different factorizations; sending either input to the other's kernel "
       "gives a wrong answer, so anything that is neither is refused.");
}

template <typename T>
void register_for(nb::module_& m) {
    register_complex_vector<T>(m);
    register_complex_matrix<T>(m);
    register_complex_factories<T>(m);
    register_complex_norms<T>(m);
    register_complex_dot<T>(m);
    register_complex_ops<T>(m);
    register_complex_lu<T>(m);
    register_complex_cholesky<T>(m);
    register_complex_ldlt<T>(m);
}

}  // namespace

// ===========================================================================
void register_complex(nb::module_& m) {
    register_for<c64>(m);
    register_for<c128>(m);
}
