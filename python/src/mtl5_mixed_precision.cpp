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
#include <mtl/sparse/iterative_refine.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ic_0.hpp>

#include <mtl/math/accumulator_traits.hpp>
// The MTL5 + Universal bridge. MTL5 deliberately ships only the contract
// (accumulator_traits) and leaves the quire specializations to the composition
// layer; this header is that specialization, mirrored across the mp-* repos.
#include <mtl/math/quire_accumulator.hpp>

#include <cmath>
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

// ===========================================================================
// Runtime accumulator / result selection
// ===========================================================================
enum class AccKind { Default, F32, F64, FMA32, FMA64, Quire };

const char* const kAccumulatorHelp =
    "valid accumulators: None (element precision), 'f32', 'f64', "
    "'fma32', 'fma64'/'fma', 'quire'";

AccKind parse_acc(const std::optional<std::string>& spec,
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

// The sum-of-squares norms are accumulated here rather than through
// mtl::two_norm<Acc> / mtl::frobenius_norm<Acc>.
//
// Upstream those do `sqrt(AT::template value<Accumulator>(acc))` — they round
// the accumulator out to the ACCUMULATOR type and then take its square root.
// That only compiles when Accumulator is a plain arithmetic type, so it rejects
// both non-trivial configurations accumulator_traits documents: fma_accumulator
// and the quire. (`value<mag_t>` is what was meant.) Reported upstream; until
// it lands, these two loops are the accumulator-generic version, rounding the
// accumulated sum out to double before the square root so a narrow element type
// does not throw away the precision the accumulator just bought.
//
// All element types bound here are real, so |x|^2 == x*x and the magnitude type
// is the element type.
template <typename T, typename Acc, typename V>
double sumsq_vec(const V& v) {
    using AT = mtl::math::accumulator_traits<Acc, T>;
    Acc acc{};
    AT::clear(acc);
    for (typename V::size_type i = 0; i < v.size(); ++i) {
        using std::abs;
        T a = abs(v(i));
        AT::add_product(acc, a, a);
    }
    return AT::template value<double>(acc);
}

template <typename T, typename Acc, typename M>
double sumsq_mat(const M& m) {
    using AT = mtl::math::accumulator_traits<Acc, T>;
    Acc acc{};
    AT::clear(acc);
    for (typename M::size_type r = 0; r < m.num_rows(); ++r)
        for (typename M::size_type c = 0; c < m.num_cols(); ++c) {
            using std::abs;
            T a = abs(m(r, c));
            AT::add_product(acc, a, a);
        }
    return AT::template value<double>(acc);
}

template <typename T, typename V>
double mixed_two_norm(const V& v, AccKind kind) {
    nogil guard;
    return dispatch_acc<T>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>)
            return static_cast<double>(mtl::two_norm(v));
        else
            return std::sqrt(sumsq_vec<T, Acc>(v));
    });
}

template <typename T, typename M>
double mixed_frobenius_norm(const M& m, AccKind kind) {
    nogil guard;
    return dispatch_acc<T>(kind, [&]<typename Acc>() -> double {
        if constexpr (std::is_same_v<Acc, void>)
            return static_cast<double>(mtl::frobenius_norm(m));
        else
            return std::sqrt(sumsq_mat<T, Acc>(m));
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
    "fixpnt8, fixpnt16, lns16, lns32";

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
// mtl::sparse::iterative_refine accepts anything exposing solve(dx, r). Today
// the bound candidates are ILU(0)/IC(0); the sparse direct factorizations
// (sparse_lu, native_klu) land in Phase 2 and are where a genuinely
// low-precision factor makes this the mixed-precision workhorse it is meant to
// be. With an incomplete factor this is preconditioned Richardson — a real
// method, but not yet the mixed-precision story.
// ===========================================================================
using ILU0d = PreconditionerWrapper<mtl::itl::pc::ilu_0<double>, double>;
using IC0d  = PreconditionerWrapper<mtl::itl::pc::ic_0<double>, double>;

template <typename PCWrap>
std::pair<nb::ndarray<nb::numpy, double, nb::ndim<1>>, nb::dict>
sparse_refine(const mtl::mat::compressed2D<double>& A, const PCWrap& M,
              nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> b_np,
              int max_iter, double rel_tol, bool scaled, int patience) {
    const std::size_t n = A.num_rows();
    if (A.num_cols() != n)
        throw std::invalid_argument("iterative_refine: A must be square");
    if (b_np.shape(0) != n)
        throw std::invalid_argument("iterative_refine: len(b) must match A");
    if (M.n != n)
        throw std::invalid_argument(
            "iterative_refine: factorization size " + std::to_string(M.n) +
            " does not match A (" + std::to_string(n) + ")");
    if (patience < 1)
        throw std::invalid_argument("iterative_refine: patience must be >= 1");
    const double* bp = b_np.data();

    mtl::sparse::refine_options opt;
    opt.max_iter = max_iter;
    opt.rel_tol  = rel_tol;
    opt.scaled   = scaled;
    opt.patience = patience;

    double* buf = new double[n];
    mtl::sparse::refine_result res;
    try {
        nogil guard;
        mtl::vec::dense_vector<double> b(n), x(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) b[i] = bp[i];
        res = mtl::sparse::iterative_refine(A, M.pc, b, x, opt);
        for (std::size_t i = 0; i < n; ++i) buf[i] = x[i];
    } catch (...) {
        delete[] buf;   // no capsule owns it yet
        throw;
    }

    std::size_t shape[1] = { n };
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    auto out = nb::ndarray<nb::numpy, double, nb::ndim<1>>(buf, 1, shape, owner);
    nb::dict d;
    d["iters"]        = res.iters;
    d["rel_residual"] = res.rel_residual;
    d["converged"]    = res.converged;
    return std::make_pair(out, d);
}

}  // namespace

// ===========================================================================
void register_mixed_precision(nb::module_& m) {
    nb::module_ mx = m.def_submodule(
        "mixed", "Mixed-precision operations: element / accumulator / result "
                 "precisions chosen independently");

    // ----- Accumulator-policy kernels ---------------------------------------
    register_mixed_native<float>(mx);
    register_mixed_native<double>(mx);

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

    mx.def("accumulators", [](const std::string& dtype) {
        std::vector<std::string> v{"f32", "f64", "fma32", "fma64"};
        // Quire availability follows Universal's fdp.hpp coverage.
        if (dtype != "f32" && dtype != "f64" && dtype != "i32" && dtype != "i64")
            v.push_back("quire");
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
            "fixpnt8", "fixpnt16", "lns16", "lns32"};
    }, "Element dtypes accepted by convert() and the mixed-precision operations");

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
    mx.def("iterative_refine", &sparse_refine<ILU0d>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3,
           "Refine a solution of A x = b through a factorization M exposing "
           "solve(). Returns (x, info).");
    mx.def("iterative_refine", &sparse_refine<IC0d>,
           "A"_a, "M"_a, "b"_a, "max_iter"_a = 20, "rel_tol"_a = 0.0,
           "scaled"_a = false, "patience"_a = 3);
}
