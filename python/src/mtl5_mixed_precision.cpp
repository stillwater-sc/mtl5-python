// mtl5-python -- mixed-precision bindings (MTL5 epic #157 + #119/#273).
//
// This is the composition layer: MTL5 supplies precision-generic kernels and
// never depends on a number library, Universal supplies the number systems, and
// this file is where the two meet and get a Python surface.
//
// The three precisions of a mixed-precision operation are independent:
//
//     element (storage)  -- fixed by the container you pass in
//     accumulator        -- the `accumulator=` argument
//     result (delivery)  -- the `result=` argument
//
// e.g. dot over posit16 vectors, accumulated in an exact quire, delivered as a
// Python float:  mtl5.mixed.dot(a, b, accumulator="quire")
//
// Split into its own translation unit because the accumulator dispatch
// instantiates each kernel once per (element type, accumulator) pair.

#include "mtl5_types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <mtl/operation/convert.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/lu_iterative_refine.hpp>
#include <mtl/operation/backward_error.hpp>
#include "mtl5_sparse_refine.hpp"
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>

#include <mtl/math/accumulator_traits.hpp>
// The MTL5 + Universal bridge. MTL5 deliberately ships only the contract
// (accumulator_traits) and leaves the quire specializations to the composition
// layer; this header is that specialization, mirrored across the mp-* repos.
#include <mtl/math/quire_accumulator.hpp>

#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

// ===========================================================================
// Which element types have a Universal quire
//
// Mirrors the four families the quire_accumulator bridge specializes. Native
// float/double/int have no quire (Universal ships no fdp.hpp for them), so
// accumulator="quire" is rejected for those with an explanatory error rather
// than failing to compile or silently falling back.
// ===========================================================================
template <typename T>
struct quire_for {
    static constexpr bool ok = false;
    using type = void;
};

template <unsigned nbits, unsigned es>
struct quire_for<sw::universal::posit<nbits, es>> {
    static constexpr bool ok = true;
    using type = sw::universal::quire<sw::universal::posit<nbits, es>>;
};

template <unsigned nbits, unsigned es, typename bt, bool sub, bool max, bool sat>
struct quire_for<sw::universal::cfloat<nbits, es, bt, sub, max, sat>> {
    static constexpr bool ok = true;
    using type = sw::universal::quire<sw::universal::cfloat<nbits, es, bt, sub, max, sat>>;
};

template <unsigned nbits, unsigned rbits, typename bt, auto... xtra>
struct quire_for<sw::universal::lns<nbits, rbits, bt, xtra...>> {
    static constexpr bool ok = true;
    using type = sw::universal::quire<sw::universal::lns<nbits, rbits, bt, xtra...>>;
};

template <unsigned nbits, unsigned rbits, bool arithmetic, typename bt>
struct quire_for<sw::universal::fixpnt<nbits, rbits, arithmetic, bt>> {
    static constexpr bool ok = true;
    using type = sw::universal::quire<sw::universal::fixpnt<nbits, rbits, arithmetic, bt>>;
};

// Which bound dtypes accept accumulator='quire', keyed by the Python-facing
// dtype string. Filled by the registration templates below from
// quire_for<T>::ok so that accumulators() and dispatch_acc() answer from one
// fact rather than two hand-maintained lists that can disagree.
inline std::map<std::string, bool>& quire_support_registry() {
    static std::map<std::string, bool> registry;
    return registry;
}

template <typename T>
void record_quire_support() {
    quire_support_registry()[type_suffix<T>()] = quire_for<T>::ok;
}

bool parse_result_is_element(const std::optional<std::string>& spec) {
    if (!spec || *spec == "f64" || *spec == "float64") return false;
    if (*spec == "element") return true;
    throw std::invalid_argument(
        "unknown result '" + *spec + "'; valid: 'f64' (deliver the accumulator "
        "rounded to double) or 'element' (round back to the element type, which "
        "is the single-rounding exact-dot-product semantics when the accumulator "
        "is a quire)");
}

// Apply `f.template operator()<Acc>()` for the selected accumulator. Every
// branch must yield the same type, so the callers return double.
template <typename T, typename F>
double dispatch_acc(AccKind kind, F&& f) {
    switch (kind) {
        case AccKind::Default: return f.template operator()<void>();
        case AccKind::F32:     return f.template operator()<float>();
        case AccKind::F64:     return f.template operator()<double>();
        case AccKind::FMA32:   return f.template operator()<mtl::math::fma_accumulator<float>>();
        case AccKind::FMA64:   return f.template operator()<mtl::math::fma_accumulator<double>>();
        case AccKind::Quire:
            if constexpr (quire_for<T>::ok)
                return f.template operator()<typename quire_for<T>::type>();
            else
                throw std::invalid_argument("accumulator='quire' unavailable for this dtype");
        case AccKind::I32:
            // Guarded exactly as Quire is, and for the same reason: a discarded
            // if-constexpr branch is not instantiated, so `mtl::dot<int32_t>`
            // over a posit or cfloat vector is never formed. parse_acc rejects
            // 'i32' for those dtypes first; this is the second line of defence
            // and the one the compiler enforces.
            //
            // int32 is exactly representable in a double's 53-bit significand,
            // so returning it through this function's double is lossless. That
            // is worth stating because it looks like a narrowing bug and is not.
            if constexpr (std::is_integral_v<T>)
                return f.template operator()<std::int32_t>();
            else
                throw std::invalid_argument("accumulator='i32' unavailable for this dtype");
    }
    throw std::invalid_argument("unreachable accumulator kind");
}

// ===========================================================================
// dot / norm with an accumulator policy
// ===========================================================================
template <typename T, typename V>
double mixed_dot(const V& a, const V& b, AccKind kind, bool result_element) {
    if (a.size() != b.size())
        throw std::invalid_argument("dot: vectors must have the same length");
    nogil guard;
    return dispatch_acc<T>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>) {
            // No policy: MTL5's default path (BLAS / SIMD / common-type loop).
            return static_cast<double>(mtl::dot<>(a, b));
        } else if (result_element) {
            // Round the accumulator out to the element type once — the
            // exact-dot-product semantics when Acc is a quire.
            return static_cast<double>(mtl::dot<Acc, T>(a, b));
        } else {
            return static_cast<double>(mtl::dot<Acc, double>(a, b));
        }
    });
}

// These went through local sum-of-squares loops until stillwater-sc/mtl5#379
// landed (MTL5 03c77ee). Before it, `two_norm<Acc>` rounded the accumulated
// value out to the ELEMENT type -- correct for a C++ caller, and what #324
// settled on -- which at the Python boundary threw away exactly what the
// accumulator bought: measured on 4000 posit16 values, f64 and quire
// accumulation both landed on rel err 1.19e-04, so `accumulator=` was
// unobservable. #379 added the Result parameter that `dot<Accumulator, Result>`
// already had, so `two_norm<Acc, double>` now delivers the accumulated value at
// double precision and the loops are gone.
template <typename T, typename V>
double mixed_two_norm(const V& v, AccKind kind) {
    nogil guard;
    return dispatch_acc<T>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>)
            return static_cast<double>(mtl::two_norm(v));
        else
            return static_cast<double>(mtl::two_norm<Acc, double>(v));
    });
}

template <typename T, typename M>
double mixed_frobenius_norm(const M& m, AccKind kind) {
    nogil guard;
    return dispatch_acc<T>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>)
            return static_cast<double>(mtl::frobenius_norm(m));
        else
            return static_cast<double>(mtl::frobenius_norm<Acc, double>(m));
    });
}

// ===========================================================================
// matvec / matmul with an accumulator policy
//
// The output container's element type selects the result precision, matching
// MTL5's `mult<Acc>(A, B, C)` contract (C's type is the delivery type).
// ===========================================================================
template <typename T>
void mixed_mult_mv(const mtl::mat::dense2D<T>& A,
                   const mtl::vec::dense_vector<T>& x,
                   mtl::vec::dense_vector<T>& y, AccKind kind) {
    dispatch_acc<T>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>) mtl::mult(A, x, y);
        else                                     mtl::mult<Acc>(A, x, y);
        return 0.0;
    });
}

template <typename T>
void mixed_mult_mm(const mtl::mat::dense2D<T>& A,
                   const mtl::mat::dense2D<T>& B,
                   mtl::mat::dense2D<T>& C, AccKind kind) {
    dispatch_acc<T>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>) mtl::mult(A, B, C);
        else                                     mtl::mult<Acc>(A, B, C);
        return 0.0;
    });
}

// ===========================================================================
// Registration helpers
// ===========================================================================

/// Universal element types: containers are bound as bare dense_vector<T> /
/// dense2D<T>, so the mixed ops take those directly.
template <typename T>
void register_mixed_universal(nb::module_& mx) {
    using Vec = mtl::vec::dense_vector<T>;
    using Mat = mtl::mat::dense2D<T>;
    constexpr bool quire_ok = quire_for<T>::ok;
    record_quire_support<T>();

    mx.def("dot", [](const Vec& a, const Vec& b,
                     std::optional<std::string> accumulator,
                     std::optional<std::string> result) {
        return mixed_dot<T>(a, b, parse_acc(accumulator, type_suffix<T>(), quire_ok),
                            parse_result_is_element(result));
    }, "a"_a, "b"_a, "accumulator"_a = nb::none(), "result"_a = nb::none());

    mx.def("norm", [](const Vec& v, int ord, std::optional<std::string> accumulator) {
        if (ord != 2)
            throw std::invalid_argument(
                "mixed.norm: only ord=2 has an accumulator policy (the one- and "
                "infinity-norms are max/sum of magnitudes, not sums of products). "
                "Use mtl5.norm() for those.");
        return mixed_two_norm<T>(v, parse_acc(accumulator, type_suffix<T>(), quire_ok));
    }, "x"_a, "ord"_a = 2, "accumulator"_a = nb::none());

    mx.def("frobenius_norm", [](const Mat& m, std::optional<std::string> accumulator) {
        return mixed_frobenius_norm<T>(m, parse_acc(accumulator, type_suffix<T>(), quire_ok));
    }, "A"_a, "accumulator"_a = nb::none());

    mx.def("matvec", [](const Mat& A, const Vec& x, std::optional<std::string> accumulator) {
        if (A.num_cols() != x.size())
            throw std::invalid_argument("matvec: A.num_cols must equal len(x)");
        const AccKind kind = parse_acc(accumulator, type_suffix<T>(), quire_ok);
        Vec y(A.num_rows());
        {
            nogil guard;
            mixed_mult_mv<T>(A, x, y, kind);
        }
        return y;
    }, "A"_a, "x"_a, "accumulator"_a = nb::none());

    mx.def("matmul", [](const Mat& A, const Mat& B, std::optional<std::string> accumulator) {
        if (A.num_cols() != B.num_rows())
            throw std::invalid_argument("matmul: A.num_cols must equal B.num_rows");
        const AccKind kind = parse_acc(accumulator, type_suffix<T>(), quire_ok);
        Mat C(A.num_rows(), B.num_cols());
        {
            nogil guard;
            mixed_mult_mm<T>(A, B, C, kind);
        }
        return C;
    }, "A"_a, "B"_a, "accumulator"_a = nb::none());
}

/// Native float/double: containers are the zero-copy view wrappers.
template <typename T>
void register_mixed_native(nb::module_& mx) {
    using VV = VectorView<T>;
    using MV = MatrixView<T>;
    record_quire_support<T>();

    mx.def("dot", [](const VV& a, const VV& b,
                     std::optional<std::string> accumulator,
                     std::optional<std::string> result) {
        return mixed_dot<T>(a.vec, b.vec,
                            parse_acc(accumulator, type_suffix<T>(), /*quire_ok=*/false),
                            parse_result_is_element(result));
    }, "a"_a, "b"_a, "accumulator"_a = nb::none(), "result"_a = nb::none());

    mx.def("dot", [](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                     nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b,
                     std::optional<std::string> accumulator,
                     std::optional<std::string> result) {
        if (a.shape(0) != b.shape(0))
            throw std::invalid_argument("dot: vectors must have the same length");
        const AccKind kind = parse_acc(accumulator, type_suffix<T>(), false);
        const bool elem = parse_result_is_element(result);
        const std::size_t n = a.shape(0);
        T* ap = const_cast<T*>(a.data());
        T* bp = const_cast<T*>(b.data());
        auto va = mtl::vec::dense_vector<T>(n, ap);
        auto vb = mtl::vec::dense_vector<T>(n, bp);
        return mixed_dot<T>(va, vb, kind, elem);
    }, "a"_a, "b"_a, "accumulator"_a = nb::none(), "result"_a = nb::none());

    mx.def("norm", [](const VV& v, int ord, std::optional<std::string> accumulator) {
        if (ord != 2)
            throw std::invalid_argument(
                "mixed.norm: only ord=2 has an accumulator policy. Use mtl5.norm() "
                "for the one- and infinity-norms.");
        return mixed_two_norm<T>(v.vec, parse_acc(accumulator, type_suffix<T>(), false));
    }, "x"_a, "ord"_a = 2, "accumulator"_a = nb::none());

    mx.def("frobenius_norm", [](const MV& m, std::optional<std::string> accumulator) {
        return mixed_frobenius_norm<T>(m.mat, parse_acc(accumulator, type_suffix<T>(), false));
    }, "A"_a, "accumulator"_a = nb::none());
}

/// Dot over two DIFFERENT narrow integer operand types.
///
/// The signedness pairing is not a detail: `u8 x i8` -- unsigned activations
/// against signed weights -- is VNNI's NATIVE shape on x86 and the one quantized
/// inference is written in. A same-type-only surface would put the whole point
/// of the 8-bit path out of reach.
///
/// MTL5 accepts every pairing at the `dot` level and swaps the operands onto
/// whichever the machine implements, because a dot product is symmetric. The
/// asymmetry is real one level down -- `simd::reduce_dot_widen` rejects
/// `(int8, uint8)` at compile time -- but that restriction is the kernel's, and
/// re-exposing it here would refuse a call the library can serve.
template <typename TA, typename TB, typename VA, typename VB>
double mixed_dot_pair(const VA& a, const VB& b, AccKind kind) {
    if (a.size() != b.size())
        throw std::invalid_argument("dot: vectors must have the same length");
    nogil guard;
    return dispatch_acc<TA>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>)
            return static_cast<double>(mtl::dot<>(a, b));
        else
            return static_cast<double>(mtl::dot<Acc, double>(a, b));
    });
}

/// The narrow integer element types (i8, i16, u8): `dot` only, and only with an
/// accumulator wider than the element.
///
/// This is what phase 1 registered the containers for. MTL5 v5.11.0 routes
/// `dot<int32_t>` over 8- and 16-bit operands onto the hardware widening
/// multiply-accumulate -- vpmaddwd / vpdpbusd on x86, SMLAL / SDOT on NEON --
/// and an int32 accumulator is what those instructions accumulate into.
///
/// `norm` and `frobenius_norm` are deliberately absent, for the reason phase 1
/// gave: two_norm takes sqrt of the accumulated sum, and on a wrapped -- and for
/// i16 negative -- sum that yields nan. An int32 accumulator would fix the sum
/// but the API has no way to say "accumulate in int32, deliver a real square
/// root", so the honest surface is not to offer it yet.
///
/// `accumulator` is REQUIRED here, unlike every other dtype. Omitting it means
/// element precision, which for these widths is exactly the wrapping that phase
/// 1 refused to expose: six products of 100 give 96 for i8 where the answer is
/// 60000. Rather than silently redefine None for these dtypes alone, ask.
/// Register `dot` for one ordered pair of distinct narrow integer operands.
template <typename TA, typename TB>
void register_mixed_narrow_int_pair(nb::module_& mx) {
    const char* kPairDoc =
        "Dot product over mixed-signedness 8-bit operands, accumulated in int32.\n\n"
        "accumulator='i32' is required. u8 x i8 is VNNI's native pairing on x86\n"
        "(unsigned activations against signed weights); ARM implements the\n"
        "symmetric pairings first instead. Either order is accepted here -- a dot\n"
        "product is symmetric, so MTL5 swaps the operands onto whichever form the\n"
        "machine has rather than dropping to the generic loop.\n\n"
        "The overflow contract is the same as the same-type form: products are\n"
        "exact, the sum wraps, and 8-bit operands give roughly 131000 terms of\n"
        "headroom in an int32.";

    auto acc_of = [](const std::optional<std::string>& spec) {
        if (!spec)
            throw std::invalid_argument(
                "mixed.dot on mixed 8-bit operands requires an explicit "
                "accumulator: pass accumulator='i32'.");
        return parse_acc(spec, type_suffix<TA>(), /*quire_ok=*/false, /*i32_ok=*/true);
    };

    mx.def("dot", [acc_of](const VectorView<TA>& a, const VectorView<TB>& b,
                           std::optional<std::string> accumulator,
                           std::optional<std::string> result) {
        if (result)
            throw std::invalid_argument(
                "mixed.dot on narrow integer operands does not accept result=.");
        return mixed_dot_pair<TA, TB>(a.vec, b.vec, acc_of(accumulator));
    }, "a"_a, "b"_a, "accumulator"_a = nb::none(), "result"_a = nb::none(), kPairDoc);

    mx.def("dot", [acc_of](nb::ndarray<TA, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                           nb::ndarray<TB, nb::ndim<1>, nb::c_contig, nb::device::cpu> b,
                           std::optional<std::string> accumulator,
                           std::optional<std::string> result) {
        if (result)
            throw std::invalid_argument(
                "mixed.dot on narrow integer operands does not accept result=.");
        const AccKind kind = acc_of(accumulator);
        const std::size_t n = a.shape(0);
        auto va = mtl::vec::dense_vector<TA>(n, const_cast<TA*>(a.data()));
        auto vb = mtl::vec::dense_vector<TB>(n, const_cast<TB*>(b.data()));
        return mixed_dot_pair<TA, TB>(va, vb, kind);
    }, nb::arg("a").noconvert(), nb::arg("b").noconvert(),
       "accumulator"_a = nb::none(), "result"_a = nb::none(), kPairDoc);
}

template <typename T>
void register_mixed_narrow_int(nb::module_& mx) {
    using VV = VectorView<T>;
    record_quire_support<T>();

    auto acc_of = [](const std::optional<std::string>& spec) {
        if (!spec)
            throw std::invalid_argument(
                std::string("mixed.dot on '") + type_suffix<T>() + "' requires an "
                "explicit accumulator: pass accumulator='i32'. The default is "
                "element precision, and an 8- or 16-bit accumulator overflows "
                "almost immediately -- six products of 100 wrap to 96 where the "
                "exact answer is 60000. See mtl5.mixed.accumulators('" +
                type_suffix<T>() + "').");
        return parse_acc(spec, type_suffix<T>(), /*quire_ok=*/false, /*i32_ok=*/true);
    };

    // The overflow contract, stated where a caller will meet it. This is not
    // decoration: the headroom differs by four orders of magnitude across the
    // operand widths, and a caller who does not know that gets silent wraparound.
    const char* kDoc =
        "Dot product accumulated in a precision wider than the element.\n\n"
        "accumulator='i32' is required. On 8-bit operands this is the quad\n"
        "multiply-accumulate (vpdpbusd / SDOT); on 16-bit it is the widening\n"
        "multiply-accumulate (vpmaddwd / SMLAL).\n\n"
        "OVERFLOW IS THE CONTRACT, not an error case. Products are always\n"
        "exact -- two int16 cannot overflow an int32 product -- but the SUM\n"
        "wraps, and how soon depends on the operand magnitude rather than the\n"
        "vector length: at b bits of magnitude the headroom is about 2^(31-2b)\n"
        "terms. Measured at full range: one i16 x i16 product uses 2^30 of the\n"
        "int32 range, so TWO of them already overflow it, while i8 x i8 holds\n"
        "131071 terms. That five order of magnitude gap is why the\n"
        "quantized-inference instructions are 8-bit. Wrapping is two's\n"
        "complement and therefore bit-identical across lane counts, backends\n"
        "and thread partitions -- reproducible, but still wrapping.\n\n"
        "Every signedness pairing is accepted. The hardware implements only\n"
        "some of them (x86 does unsigned x signed first, ARM the symmetric\n"
        "ones), and a dot product is symmetric, so MTL5 swaps the operands onto\n"
        "whichever the machine has rather than refusing or falling back to the\n"
        "generic loop.";

    mx.def("dot", [acc_of](const VV& a, const VV& b,
                           std::optional<std::string> accumulator,
                           std::optional<std::string> result) {
        if (result)
            throw std::invalid_argument(
                std::string("mixed.dot on '") + type_suffix<T>() + "' does not accept "
                "result=: rounding an int32 sum back to an 8- or 16-bit element "
                "would re-introduce the wrap the accumulator exists to avoid. The "
                "result is delivered as a Python float, which holds every int32 "
                "exactly.");
        return mixed_dot<T>(a.vec, b.vec, acc_of(accumulator), /*result_element=*/false);
    }, "a"_a, "b"_a, "accumulator"_a = nb::none(), "result"_a = nb::none(), kDoc);

    mx.def("dot", [acc_of](nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                           nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu> b,
                           std::optional<std::string> accumulator,
                           std::optional<std::string> result) {
        if (a.shape(0) != b.shape(0))
            throw std::invalid_argument("dot: vectors must have the same length");
        if (result)
            throw std::invalid_argument(
                std::string("mixed.dot on '") + type_suffix<T>() + "' does not accept "
                "result=: see the docstring.");
        const AccKind kind = acc_of(accumulator);
        const std::size_t n = a.shape(0);
        auto va = mtl::vec::dense_vector<T>(n, const_cast<T*>(a.data()));
        auto vb = mtl::vec::dense_vector<T>(n, const_cast<T*>(b.data()));
        return mixed_dot<T>(va, vb, kind, /*result_element=*/false);
    }, nb::arg("a").noconvert(), nb::arg("b").noconvert(),
       "accumulator"_a = nb::none(), "result"_a = nb::none(), kDoc);
}

// ===========================================================================
// convert() — element-wise re-quantization into a target number system
//
// This is MTL5's NON-fused path: a separate pass over a stored tensor. The
// fused accumulate->store conversion is what `accumulator=`/`result=` do.
//
// The entry point is a float64 array, which is what NumPy hands us; the way
// back is the containers' own .to_numpy(). Container-to-container
// re-quantization therefore round-trips through float64 — exact for every
// bound type except posit64, whose significand exceeds float64's 53 bits.
// ===========================================================================
template <typename T>
nb::object vec_from_f64(const double* src, std::size_t n) {
    mtl::vec::dense_vector<T> out(n);
    {
        nogil guard;
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<T>(src[i]);
    }
    return nb::cast(std::move(out));
}

template <typename T>
nb::object mat_from_f64(const double* src, std::size_t rows, std::size_t cols) {
    mtl::mat::dense2D<T> out(rows, cols);
    {
        nogil guard;
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < cols; ++c)
                out(r, c) = static_cast<T>(src[r * cols + c]);
    }
    return nb::cast(std::move(out));
}

// Native types keep their zero-copy view wrappers rather than the bare
// containers, so `convert(a, "f32")` yields the same class as mtl5.vector().
template <typename T>
nb::object native_vec_from_f64(const double* src, std::size_t n) {
    mtl::vec::dense_vector<T> v(n);
    {
        nogil guard;
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<T>(src[i]);
    }
    return nb::cast(VectorView<T>(std::move(v)));
}

template <typename T>
nb::object native_mat_from_f64(const double* src, std::size_t rows, std::size_t cols) {
    mtl::mat::dense2D<T> m(rows, cols);
    {
        nogil guard;
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < cols; ++c)
                m(r, c) = static_cast<T>(src[r * cols + c]);
    }
    return nb::cast(MatrixView<T>(std::move(m)));
}

using vec_ctor = nb::object (*)(const double*, std::size_t);
using mat_ctor = nb::object (*)(const double*, std::size_t, std::size_t);

const char* const kDtypeHelp =
    "valid dtypes: f32, f64, fp8, fp16, posit8, posit16, posit32, posit64, "
    "fixpnt8, fixpnt16, lns16, lns32, cfloat32, takum32, dd_cascade, "
    "td_cascade, qd_cascade";

vec_ctor vec_ctor_for(const std::string& dtype) {
    if (dtype == "f32")      return &native_vec_from_f64<float>;
    if (dtype == "f64")      return &native_vec_from_f64<double>;
    if (dtype == "fp8")      return &vec_from_f64<fp8>;
    if (dtype == "fp16")     return &vec_from_f64<fp16>;
    if (dtype == "posit8")   return &vec_from_f64<posit8>;
    if (dtype == "posit16")  return &vec_from_f64<posit16>;
    if (dtype == "posit32")  return &vec_from_f64<posit32>;
    if (dtype == "posit64")  return &vec_from_f64<posit64>;
    if (dtype == "fixpnt8")  return &vec_from_f64<fixpnt8>;
    if (dtype == "fixpnt16") return &vec_from_f64<fixpnt16>;
    if (dtype == "lns16")    return &vec_from_f64<lns16>;
    if (dtype == "lns32")    return &vec_from_f64<lns32>;
    if (dtype == "cfloat32")   return &vec_from_f64<cfloat32>;
    if (dtype == "takum32")    return &vec_from_f64<takum32>;
    if (dtype == "dd_cascade") return &vec_from_f64<dd_cascade>;
    if (dtype == "td_cascade") return &vec_from_f64<td_cascade>;
    if (dtype == "qd_cascade") return &vec_from_f64<qd_cascade>;
    throw std::invalid_argument("unknown dtype '" + dtype + "'; " + kDtypeHelp);
}

mat_ctor mat_ctor_for(const std::string& dtype) {
    if (dtype == "f32")      return &native_mat_from_f64<float>;
    if (dtype == "f64")      return &native_mat_from_f64<double>;
    if (dtype == "fp8")      return &mat_from_f64<fp8>;
    if (dtype == "fp16")     return &mat_from_f64<fp16>;
    if (dtype == "posit8")   return &mat_from_f64<posit8>;
    if (dtype == "posit16")  return &mat_from_f64<posit16>;
    if (dtype == "posit32")  return &mat_from_f64<posit32>;
    if (dtype == "posit64")  return &mat_from_f64<posit64>;
    if (dtype == "fixpnt8")  return &mat_from_f64<fixpnt8>;
    if (dtype == "fixpnt16") return &mat_from_f64<fixpnt16>;
    if (dtype == "lns16")    return &mat_from_f64<lns16>;
    if (dtype == "lns32")    return &mat_from_f64<lns32>;
    if (dtype == "cfloat32")   return &mat_from_f64<cfloat32>;
    if (dtype == "takum32")    return &mat_from_f64<takum32>;
    if (dtype == "dd_cascade") return &mat_from_f64<dd_cascade>;
    if (dtype == "td_cascade") return &mat_from_f64<td_cascade>;
    if (dtype == "qd_cascade") return &mat_from_f64<qd_cascade>;
    throw std::invalid_argument("unknown dtype '" + dtype + "'; " + kDtypeHelp);
}

// ===========================================================================
// Dense mixed-precision LU iterative refinement (#273)
//
// A and b arrive in double (the Residual precision); `working` names the low
// precision the factorization runs in. That is the whole point: the O(n^3) work
// happens in the cheap type, an accurate residual recovers the accuracy.
// ===========================================================================
template <typename Working>
mtl::lu_refine_result run_lu_refine(const mtl::mat::dense2D<double>& A,
                                    const mtl::vec::dense_vector<double>& b,
                                    mtl::vec::dense_vector<double>& x,
                                    const mtl::lu_refine_options& opt) {
    return mtl::lu_iterative_refine<Working>(A, b, x, opt);
}

using refine_fn = mtl::lu_refine_result (*)(const mtl::mat::dense2D<double>&,
                                            const mtl::vec::dense_vector<double>&,
                                            mtl::vec::dense_vector<double>&,
                                            const mtl::lu_refine_options&);

refine_fn refine_for_dtype(const std::string& dtype) {
    if (dtype == "f32")      return &run_lu_refine<float>;
    if (dtype == "f64")      return &run_lu_refine<double>;
    if (dtype == "fp8")      return &run_lu_refine<fp8>;
    if (dtype == "fp16")     return &run_lu_refine<fp16>;
    if (dtype == "posit8")   return &run_lu_refine<posit8>;
    if (dtype == "posit16")  return &run_lu_refine<posit16>;
    if (dtype == "posit32")  return &run_lu_refine<posit32>;
    if (dtype == "posit64")  return &run_lu_refine<posit64>;
    if (dtype == "fixpnt8")  return &run_lu_refine<fixpnt8>;
    if (dtype == "fixpnt16") return &run_lu_refine<fixpnt16>;
    if (dtype == "lns16")    return &run_lu_refine<lns16>;
    if (dtype == "lns32")    return &run_lu_refine<lns32>;
    throw std::invalid_argument(
        "unknown working precision '" + dtype + "'; valid: f32, f64, fp8, fp16, "
        "posit8, posit16, posit32, posit64, fixpnt8, fixpnt16, lns16, lns32");
}

nb::dict refine_result_dict(const mtl::lu_refine_result& r) {
    nb::dict d;
    d["iters"]        = r.iters;
    d["rel_residual"] = r.rel_residual;
    d["converged"]    = r.converged;
    return d;
}

// ===========================================================================
// Sparse iterative refinement through a factorization
//
// The binding template lives in mtl5_sparse_refine.hpp so the sparse direct
// factorizations (registered in their own TU) can reuse it. Bound here for the
// preconditioners; a low-precision DIRECT factor is where the mixed-precision
// payoff actually lives.
// ===========================================================================
using ILU0d = PreconditionerWrapper<mtl::itl::pc::ilu_0<double>, double>;
using IC0d  = PreconditionerWrapper<mtl::itl::pc::ic_0<double>, double>;

}  // namespace

// ===========================================================================
void register_mixed_precision(nb::module_& m) {
    nb::module_ mx = m.def_submodule(
        "mixed", "Mixed-precision operations: element / accumulator / result "
                 "precisions chosen independently");

    // ----- Accumulator-policy kernels ---------------------------------------
    register_mixed_native<float>(mx);
    register_mixed_native<double>(mx);

    // Narrow integers: dot only, accumulator required. See
    // register_mixed_narrow_int for why norm/frobenius_norm are absent.
    register_mixed_narrow_int<std::int8_t>(mx);
    register_mixed_narrow_int<std::int16_t>(mx);
    register_mixed_narrow_int<std::uint8_t>(mx);
    // Mixed-signedness 8-bit pairings, both orders. u8 x i8 is what VNNI
    // implements natively on x86 and what quantized inference is written in.
    register_mixed_narrow_int_pair<std::uint8_t, std::int8_t>(mx);
    register_mixed_narrow_int_pair<std::int8_t, std::uint8_t>(mx);

    register_mixed_universal<fp8>(mx);
    register_mixed_universal<fp16>(mx);
    register_mixed_universal<posit8>(mx);
    register_mixed_universal<posit16>(mx);
    register_mixed_universal<posit32>(mx);
    register_mixed_universal<posit64>(mx);
    register_mixed_universal<fixpnt8>(mx);
    register_mixed_universal<fixpnt16>(mx);
    register_mixed_universal<lns16>(mx);
    register_mixed_universal<lns32>(mx);

    // Emulated IEEE binary32, takum, and the float cascades (#69). Only
    // cfloat32 gets a quire: quire_for<> specializes on the four families
    // Universal ships an fdp.hpp for, so accumulator='quire' on takum32 or a
    // cascade is rejected at runtime with the same message native float gets,
    // rather than failing to compile.
    register_mixed_universal<cfloat32>(mx);
    register_mixed_universal<takum32>(mx);
    register_mixed_universal<dd_cascade>(mx);
    register_mixed_universal<td_cascade>(mx);
    register_mixed_universal<qd_cascade>(mx);

    mx.def("accumulators", [](const std::string& dtype) {
        // The narrow integer types answer differently: an int32 accumulator is
        // the one that maps to hardware, and the float accumulators are not
        // offered because dot<float> over int8 operands takes the generic loop
        // and would quietly be the slow path dressed as a precision choice.
        if (dtype == "i8" || dtype == "i16" || dtype == "u8")
            return std::vector<std::string>{"i32"};
        std::vector<std::string> v{"f32", "f64", "fma32", "fma64"};
        // Answered from the registry the registration templates fill with
        // quire_for<T>::ok -- the SAME trait dispatch_acc() consults -- so what
        // is advertised cannot drift from what the operations accept.
        //
        // This used to be a denylist of the native dtypes, which failed open:
        // every dtype not named in it was advertised as having a quire. That
        // was right only by accident, because until takum32 and the cascades
        // arrived every non-native bound type happened to have one. Adding
        // them made accumulators() promise a quire that dot() rejects, and an
        // unknown dtype string was advertised a quire too.
        const auto& reg = quire_support_registry();
        auto it = reg.find(dtype);
        if (it == reg.end())
            throw std::invalid_argument("unknown dtype '" + dtype + "'; " + kDtypeHelp);
        if (it->second) v.push_back("quire");
        return v;
    }, "dtype"_a,
       "Accumulators available for an element dtype (None is always valid and "
       "means 'accumulate in the element precision')");

    // ----- convert() ---------------------------------------------------------
    m.def("convert", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a,
                        const std::string& dtype) {
        return vec_ctor_for(dtype)(a.data(), a.shape(0));
    }, "a"_a, "dtype"_a,
       "Re-quantize a 1-D float64 array into another number system");

    m.def("convert", [](nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> a,
                        const std::string& dtype) {
        return mat_ctor_for(dtype)(a.data(), a.shape(0), a.shape(1));
    }, "a"_a, "dtype"_a,
       "Re-quantize a 2-D float64 array into another number system");

    m.def("dtypes", []() {
        return std::vector<std::string>{
            "f32", "f64", "fp8", "fp16",
            "posit8", "posit16", "posit32", "posit64",
            "fixpnt8", "fixpnt16", "lns16", "lns32",
            "cfloat32", "takum32",
            "dd_cascade", "td_cascade", "qd_cascade"};
    }, "Element dtypes accepted by convert() and the mixed-precision operations.\n\n"
       "This is the set convert() can TARGET, and callers rely on that -- the\n"
       "test suite parametrizes over it and converts into every entry.\n\n"
       "The narrow integer element types (i8, i16, u8) are deliberately NOT\n"
       "here. They are real element types with containers and a mixed.dot\n"
       "(accumulator='i32'), but convert() cannot target them: it re-quantizes\n"
       "a float64 array, and rounding reals into an 8-bit integer is a\n"
       "quantization scheme -- scale, zero point, rounding mode -- rather than\n"
       "a cast. A naive version would silently clip everything outside\n"
       "[-128, 127]. Build them with NumPy and pass them in;\n"
       "mtl5.mixed.accumulators('i8') answers for them.");

    // ----- Dense mixed-precision iterative refinement -------------------------
    mx.def("lu_iterative_refine",
           [](nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np,
              nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np,
              const std::string& working, int max_iter, double rel_tol,
              bool scaled, int patience) {
        const std::size_t n = A_np.shape(0);
        if (A_np.shape(1) != n)
            throw std::invalid_argument("lu_iterative_refine: A must be square");
        if (b_np.shape(0) != n)
            throw std::invalid_argument("lu_iterative_refine: len(b) must match A");
        if (patience < 1)
            throw std::invalid_argument("lu_iterative_refine: patience must be >= 1");

        refine_fn fn = refine_for_dtype(working);   // may throw before any work
        const double* Ap = A_np.data();
        const double* bp = b_np.data();

        mtl::lu_refine_options opt;
        opt.max_iter = max_iter;
        opt.rel_tol  = rel_tol;
        opt.scaled   = scaled;
        opt.patience = patience;

        double* buf = new double[n];
        mtl::lu_refine_result res;
        try {
            nogil guard;
            mtl::mat::dense2D<double> A(n, n);
            for (std::size_t r = 0; r < n; ++r)
                for (std::size_t c = 0; c < n; ++c) A(r, c) = Ap[r * n + c];
            mtl::vec::dense_vector<double> b(n), x(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) b[i] = bp[i];

            res = fn(A, b, x, opt);
            for (std::size_t i = 0; i < n; ++i) buf[i] = x[i];
        } catch (...) {
            delete[] buf;   // no capsule owns it yet
            throw;
        }

        std::size_t shape[1] = { n };
        nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        auto out = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
        return std::make_pair(out, refine_result_dict(res));
    }, "A"_a, "b"_a, "working"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
       "scaled"_a = false, "patience"_a = 3,
       "Solve A x = b by factoring in `working` precision and refining with a "
       "float64 residual. Returns (x, info).");

    mx.def("backward_error",
           [](nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> A_np,
              nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> x_np,
              nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np) {
        const std::size_t n = A_np.shape(0);
        if (A_np.shape(1) != n)
            throw std::invalid_argument("backward_error: A must be square");
        if (x_np.shape(0) != n || b_np.shape(0) != n)
            throw std::invalid_argument("backward_error: len(x)/len(b) must match A");
        const double* Ap = A_np.data();
        const double* xp = x_np.data();
        const double* bp = b_np.data();
        nogil guard;
        mtl::mat::dense2D<double> A(n, n);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c) A(r, c) = Ap[r * n + c];
        mtl::vec::dense_vector<double> x(n), b(n);
        for (std::size_t i = 0; i < n; ++i) { x[i] = xp[i]; b[i] = bp[i]; }
        return mtl::normwise_backward_error(A, x, b);
    }, "A"_a, "x"_a, "b"_a,
       "Normwise backward error ||b - Ax||_inf / (||A||_inf ||x||_inf + ||b||_inf)");

    // ----- Sparse iterative refinement through a factorization ---------------
    mx.def("iterative_refine", &sparse_refine_through<ILU0d>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3, kRefineDoc);
    mx.def("iterative_refine", &sparse_refine_through<IC0d>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
}
