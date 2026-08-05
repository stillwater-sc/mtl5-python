#pragma once
// Shared across the mtl5-python translation units: the Universal type aliases,
// the Python-facing dtype naming, the zero-copy view wrappers, and the GIL
// policy. Split out when the mixed-precision bindings moved into their own TU
// (they carry the heavy template instantiations, so keeping them separate keeps
// incremental builds tolerable).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>

#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>
#include <universal/number/fixpnt/fixpnt.hpp>
#include <universal/number/lns/lns.hpp>

#include <complex>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

// ===========================================================================
// GIL policy
//
// `nogil` marks a region that runs pure C++ on memory we own (or on a NumPy
// buffer the caller handed us) with the interpreter released, so MTL5's
// threaded kernels (#221/#297) can actually overlap with Python and a long
// factorization no longer freezes the interpreter.
//
// Rules for every use:
//   * No Python C-API inside the region. That means no nb::object, no
//     nb::cast, no nb::ndarray construction, and no destruction of a
//     VectorView/MatrixView (its `source` member is an nb::object).
//   * Raw pointers and shapes are read from an ndarray BEFORE the region;
//     dereferencing them inside is fine.
//   * Throwing out of the region is safe — the guard reacquires the GIL while
//     unwinding, before nanobind translates the exception.
//   * Reading a borrowed NumPy buffer without the GIL carries the same
//     contract as any nogil extension: the caller must not mutate the array
//     from another thread for the duration of the call.
// ===========================================================================
using nogil = nb::gil_scoped_release;

// ---------------------------------------------------------------------------
// Universal type aliases
// ---------------------------------------------------------------------------
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

// Complex, named for NumPy's convention: the digits are the TOTAL width, so
// c64 is a pair of float32 and c128 a pair of float64.
using c64  = std::complex<float>;
using c128 = std::complex<double>;

template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};
template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;

// True when element type T is bound to Python through a VectorView/MatrixView
// wrapper rather than as a bare MTL5 container. The native and complex types
// are; the Universal number systems are bound as containers directly. Several
// helpers need to know which, and `std::is_arithmetic_v` was standing in for it
// before complex arrived -- which is no longer the same question.
template <typename T>
inline constexpr bool uses_view_wrapper_v = std::is_arithmetic_v<T> || is_complex_v<T>;

// ---------------------------------------------------------------------------
// Human-readable suffix for Python class names and dtype strings
// ---------------------------------------------------------------------------
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
template <> constexpr const char* type_suffix<c64>()     { return "c64"; }
template <> constexpr const char* type_suffix<c128>()    { return "c128"; }

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
// PreconditionerWrapper — an MTL5 preconditioner plus the factor dimension, so
// a mismatched RHS fails with a clean Python error before reaching the kernel.
// Shared because the mixed-precision TU passes these to sparse iterative
// refinement (any object exposing solve(x, b) satisfies that contract).
// ===========================================================================
template <typename PC, typename T>
struct PreconditionerWrapper {
    PC pc;
    std::size_t n;

    PreconditionerWrapper(const mtl::mat::compressed2D<T>& A)
        : pc(A), n(A.num_rows()) {}

    /// The object mtl::sparse::iterative_refine drives via solve(x, b).
    /// Every wrapper that can act as a refinement factorization exposes this,
    /// so one binding template serves preconditioners and direct factors alike.
    const PC& factor() const { return pc; }
};

// Registered by mtl5_mixed_precision.cpp — convert(), the accumulator-policy
// operations, and the iterative-refinement entry points.
void register_mixed_precision(nb::module_& m);

// Registered by mtl5_dense_factor.cpp — QR, LQ and LDL^T. Bunch-Kaufman is
// withheld pending stillwater-sc/mtl5#335.
void register_dense_factorizations(nb::module_& m);

// Registered by mtl5_dense_ops.cpp — eigen, BLAS L2/L3, property predicates.
// SVD (and the condition/rank queries built on it) is withheld pending
// stillwater-sc/mtl5#337.
void register_dense_ops(nb::module_& m);

// ===========================================================================
// Accumulator vocabulary
//
// Shared between the mixed-precision operations and the sparse factorizations
// so that `accumulator=` means the same thing everywhere it appears.
// ===========================================================================
enum class AccKind { Default, F32, F64, FMA32, FMA64, Quire };

inline constexpr const char* kAccumulatorHelp =
    "valid accumulators: None (element precision), 'f32', 'f64', "
    "'fma32', 'fma64'/'fma', 'quire'";

inline AccKind parse_acc(const std::optional<std::string>& spec,
                         const char* dtype, bool quire_ok) {
    if (!spec || *spec == "none" || *spec == "default") return AccKind::Default;
    const std::string& a = *spec;
    if (a == "f32" || a == "float32") return AccKind::F32;
    if (a == "f64" || a == "float64") return AccKind::F64;
    if (a == "fma32")                 return AccKind::FMA32;
    if (a == "fma" || a == "fma64")   return AccKind::FMA64;
    if (a == "quire") {
        if (!quire_ok)
            throw std::invalid_argument(
                std::string("accumulator='quire' is not available for dtype '") + dtype +
                "': Universal provides a quire only for the posit, cfloat, lns and "
                "fixpnt families. Use 'f64' to accumulate in double instead.");
        return AccKind::Quire;
    }
    throw std::invalid_argument("unknown accumulator '" + a + "'; " + kAccumulatorHelp);
}

// Registered by mtl5_tensor.cpp — index-notation tensor algebra.
void register_tensor_algebra(nb::module_& m);

// Registered by mtl5_views.cpp — matrix views.
void register_views(nb::module_& m);

// Registered by mtl5_multigrid.cpp — smoothers, grid transfer, multigrid.
void register_multigrid(nb::module_& m);

// Registered by mtl5_krylov.cpp — Krylov solver / preconditioner dispatch.
void register_krylov(nb::module_& m);

// Registered by mtl5_ndarray.cpp — the N-D array layer (mtl/array).
void register_ndarray_layer(nb::module_& m);

// Registered by mtl5_sparse_formats.cpp — COO and ELL storage formats.
void register_sparse_formats(nb::module_& m);

// Registered by mtl5_complex.cpp — complex element types.
void register_complex(nb::module_& m);

// Registered by mtl5_io.cpp — Matrix Market I/O and spy PNG output.
void register_io(nb::module_& m);

// Registered by mtl5_generators.cpp — named test matrices and range vectors.
void register_generators(nb::module_& m);

// Registered by mtl5_sparse_direct.cpp — the sparse direct factorizations and
// the fill-reducing orderings.
void register_sparse_direct(nb::module_& m);
