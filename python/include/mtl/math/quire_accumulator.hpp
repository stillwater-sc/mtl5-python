#pragma once
// mp-* composition layer -- accumulator_traits bridge to Universal's quire
// super-accumulators. (Mirrored byte-for-byte across mp-blas, mp-iterative,
// mp-ir, mp-numerics, and mtl5-python; keep the copies in sync.)
//
// A quire is a fixed-size super-accumulator that captures sums of products of a
// fixed-size arithmetic type EXACTLY, deferring rounding to a single conversion
// at the end. The pattern applies to any fixed-size arithmetic that Universal
// exposes a quire for -- posit, cfloat, lns, fixpnt, dbns.
//
// Specializes mtl::math::accumulator_traits<Acc, Value> so that MTL5's
// mixed-precision kernels -- dot()/mult()/norms(), see
// mtl/math/accumulator_traits.hpp -- accumulate inner products in an exact
// quire and round once, instead of accumulating in the value type's arithmetic
// directly. This is the "super-accumulator" configuration (single-event
// rounding of an exact sum of products).
//
// This header lives in the mp-* composition layer, NOT in MTL5: MTL5 is the
// general linear-algebra layer and never depends on Universal, so all MTL5 +
// Universal coupling belongs here. It keeps the <mtl/math/...> path so the
// specialization sits next to the generic accumulator_traits it extends. No
// opt-in macro is needed -- if you are building against the composition layer,
// the composition is what you asked for.
//
// Contract mapping (mtl::math::accumulator_traits):
//   clear(a)          -> a.clear()
//   assign(a, v)      -> a.clear(); a += quire_mul(v, Value(1))   (exact v)
//   add_product(a,m,v)-> a += quire_mul(m, v)   (exact, unrounded blocktriple)
//   value<Result>(a)  -> quire_resolve(a) when Result == Value (exact single
//                        rounding to the element type), else a.convert_to<Result>()
//                        (round the exact accumulator out to another type, e.g.
//                        double). quire_resolve does exact blocktriple rounding;
//                        convert_to uses a double intermediate (<= 53 bits), so
//                        preferring quire_resolve keeps the exact-dot-product
//                        promise for element types wider than double.
//
// The quire-capable aggregator headers (posit.hpp / cfloat.hpp / lns.hpp /
// fixpnt.hpp) now pull in their own fdp.hpp, so quire_mul / quire_resolve are
// reachable from the number-system header alone -- no explicit fdp.hpp include
// needed (stillwater-sc/universal#1201, #1244, #1245).
//
// NOT wired up:
//   - integer, native (float/double): Universal has no fdp.hpp (no quire_mul /
//     quire_resolve), so there is no quire pattern to bridge.
//   - dbns: Universal's dbns has an fdp.hpp, but the dbns number system itself
//     misrepresents even exactly-representable integers in the configurations
//     that satisfy dbns/fdp.hpp's quire static_assert (e.g. dbns<10,4>(2) == 2.40,
//     dbns<8,3>(2) == 0, dbns<16,8>(2) == garbage), so a bridge would silently
//     produce wrong results. Add the specialization (same shape as below) once
//     the upstream dbns defect is fixed.
//
// Known Universal exactness limitations these element types inherit (the study
// applications/level1/dot_accumulator_study quantifies them):
//   - posit           : exact -- the quire delivers true single-event rounding.
//   - cfloat (no subn.): quire_traits<cfloat>::radix_point is undersized, so a
//                        quire dot floors at ~2^-radix_point (~1e-9 for
//                        cfloat<16,5>) instead of exact (stillwater-sc/universal#1202).
//   - lns             : quire_mul routes the product through double, so the lns
//                        quire is NOT exact and can be worse than a promoted
//                        double accumulator (stillwater-sc/universal#1203).
//   - fixpnt          : exact within the accumulator's dynamic range.

#include <type_traits>  // std::is_same_v

#include <universal/number/posit/posit.hpp>     // aggregator pulls posit/fdp.hpp (quire_mul/quire_resolve)
#include <universal/number/cfloat/cfloat.hpp>   // aggregator pulls cfloat/fdp.hpp
#include <universal/number/lns/lns.hpp>         // aggregator pulls lns/fdp.hpp
#include <universal/number/fixpnt/fixpnt.hpp>   // aggregator pulls fixpnt/fdp.hpp

#include <mtl/math/accumulator_traits.hpp>

namespace mtl::math {

// A quire<Value> accumulates products of `Value` exactly. Each specialization
// below maps the four accumulator_traits members onto Universal's quire API;
// they differ only in the element type's template signature.

/// posit quire. NOTE: posit's third template parameter `bt` (limb type) defaults
/// to std::uint8_t in Universal's posit_impl.hpp; this binds to that default. If
/// your project overrides `bt`, extend the pattern to include it explicitly.
template <unsigned nbits, unsigned es, unsigned capacity, typename LimbType>
struct accumulator_traits<sw::universal::quire<sw::universal::posit<nbits, es>, capacity, LimbType>,
                           sw::universal::posit<nbits, es>> {
    using Value = sw::universal::posit<nbits, es>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        if constexpr (std::is_same_v<Result, Value>) {
            return sw::universal::quire_resolve(a);   // exact single-event rounding
        } else {
            return a.template convert_to<Result>();   // round the exact sum out to Result
        }
    }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

/// cfloat quire. cfloat uses blocktriple as its arithmetic engine, so quire_mul
/// returns the exact (unrounded) full-precision MUL blocktriple.
template <unsigned nbits, unsigned es, typename bt,
          bool hasSubnormals, bool hasMaxExpValues, bool isSaturating,
          unsigned capacity, typename LimbType>
struct accumulator_traits<
        sw::universal::quire<sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasMaxExpValues, isSaturating>,
                             capacity, LimbType>,
        sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasMaxExpValues, isSaturating>> {
    using Value = sw::universal::cfloat<nbits, es, bt, hasSubnormals, hasMaxExpValues, isSaturating>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        if constexpr (std::is_same_v<Result, Value>) {
            return sw::universal::quire_resolve(a);
        } else {
            return a.template convert_to<Result>();
        }
    }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

/// lns quire. CAVEAT: Universal's lns quire_mul forms the product by converting
/// both operands to double (number/lns/fdp.hpp), so the per-term product is exact
/// only to ~53 significand bits -- the quire still removes cross-term accumulation
/// error but does not give the single-event-rounding guarantee posit/cfloat do
/// (stillwater-sc/universal#1203). `bt` defaults to std::uint8_t; the pattern
/// binds Universal's single optional `xtra`.
template <unsigned nbits, unsigned rbits, typename bt, auto... xtra,
          unsigned capacity, typename LimbType>
struct accumulator_traits<
        sw::universal::quire<sw::universal::lns<nbits, rbits, bt, xtra...>, capacity, LimbType>,
        sw::universal::lns<nbits, rbits, bt, xtra...>> {
    using Value = sw::universal::lns<nbits, rbits, bt, xtra...>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        if constexpr (std::is_same_v<Result, Value>) {
            return sw::universal::quire_resolve(a);
        } else {
            return a.template convert_to<Result>();
        }
    }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

/// fixpnt quire. The `arithmetic` template parameter is a bool (Modulo vs
/// Saturate) in Universal's fixpnt<nbits, rbits, arithmetic, bt>.
template <unsigned nbits, unsigned rbits, bool arithmetic, typename bt,
          unsigned capacity, typename LimbType>
struct accumulator_traits<
        sw::universal::quire<sw::universal::fixpnt<nbits, rbits, arithmetic, bt>, capacity, LimbType>,
        sw::universal::fixpnt<nbits, rbits, arithmetic, bt>> {
    using Value = sw::universal::fixpnt<nbits, rbits, arithmetic, bt>;
    using Acc   = sw::universal::quire<Value, capacity, LimbType>;

    static void clear(Acc& a) { a.clear(); }

    static void assign(Acc& a, const Value& v) {
        a.clear();
        a += sw::universal::quire_mul(v, Value(1));
    }

    template <typename Result = Value>
    static Result value(const Acc& a) {
        if constexpr (std::is_same_v<Result, Value>) {
            return sw::universal::quire_resolve(a);
        } else {
            return a.template convert_to<Result>();
        }
    }

    static void add_product(Acc& a, const Value& m, const Value& v) {
        a += sw::universal::quire_mul(m, v);
    }
};

} // namespace mtl::math
