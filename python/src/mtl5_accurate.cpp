// mtl5-python -- sums and dot products as if in K-fold precision (MTL5 #611).
//
// Ogita, Rump and Oishi, "Accurate sum and dot product", SIAM J. Sci. Comput.
// 26(6), 2005: SumK and DotK from mtl/operation/sum_dot_k.hpp.
//
//   mtl5.sum_k(x, k=2)     the sum of x, as if computed in k-fold working
//                          precision and rounded once (k = 2 is Sum2)
//   mtl5.dot_k(x, y, k=2)  the dot product, likewise (k = 2 is Dot2)
//
// The relative error is about u + (phi u)^k cond, so each extra pass buys
// another factor of the unit roundoff u against the condition number. They
// keep every error term, so they work on whole vectors and are not
// accumulators -- except k = 2, which needs no storage and IS one:
// mtl5.mixed.dot(x, y, accumulator="compensated") computes the same loop.
//
// float32 and float64 only: the error-free transformations under them
// (two_sum, the FMA two_product) are defined for binary IEEE types.
//
// K is a template parameter upstream. It is dispatched here over 1..kMaxK for
// the sum and 2..kMaxK for the dot product: each pass gains a factor of u, so
// in float64 k = 10 already covers condition numbers far past the 1e307 the
// range allows, and in float32 the same.

#include "mtl5_types.hpp"

#include <mtl/operation/sum_dot_k.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

using namespace nb::literals;

namespace {

constexpr int kMaxK = 10;

template <typename T>
using Arr1 = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void require_k(int k, int lo, const char* who) {
    if (k < lo || k > kMaxK)
        throw std::invalid_argument(std::string(who) + ": k must be between " +
                                    std::to_string(lo) + " and " + std::to_string(kMaxK) +
                                    ", got " + std::to_string(k));
}

/// Call `f.template operator()<K>()` for the runtime k, K in [Lo, kMaxK].
template <int Lo, typename F>
auto dispatch_k(int k, F&& f) {
    return [&]<int... I>(std::integer_sequence<int, I...>) {
        using R = decltype(f.template operator()<Lo>());
        R out{};
        // Exactly one I matches; require_k has range-checked k already.
        (void)((k == Lo + I ? (out = f.template operator()<Lo + I>(), true) : false) || ...);
        return out;
    }(std::make_integer_sequence<int, kMaxK - Lo + 1>{});
}

/// A non-owning dense_vector over an ndarray's buffer. MTL5 never writes
/// through it here; the const_cast only satisfies the constructor.
template <typename T>
mtl::vec::dense_vector<T> borrow(const Arr1<T>& a) {
    return mtl::vec::dense_vector<T>(a.shape(0), const_cast<T*>(a.data()));
}

template <typename T>
double sum_k_impl(const mtl::vec::dense_vector<T>& x, int k) {
    require_k(k, 1, "sum_k");
    nogil guard;
    return static_cast<double>(
        dispatch_k<1>(k, [&]<int K>() { return mtl::sum_k<K>(x); }));
}

template <typename T>
double dot_k_impl(const mtl::vec::dense_vector<T>& x, const mtl::vec::dense_vector<T>& y,
                  int k) {
    require_k(k, 2, "dot_k");
    if (x.size() != y.size())
        throw std::invalid_argument("dot_k: vectors must have the same length");
    nogil guard;
    return static_cast<double>(
        dispatch_k<2>(k, [&]<int K>() { return mtl::dot_k<K>(x, y); }));
}

constexpr const char* kSumDoc =
    "The sum of x as if computed in k-fold working precision, rounded once to "
    "x's type: Ogita-Rump-Oishi SumK, and Sum2 for k = 2 (the default). k = 1 "
    "is the plain recursive sum.\n\n"
    "The error is about u abs(s) + (phi u)^k sum(abs(x)), with u the unit "
    "roundoff and s the exact sum: each pass buys another factor of u against "
    "the condition number sum(abs(x)) / abs(s). Cost: (6k - 5)(n - 1) flops "
    "and one copy of x.\n\n"
    "float32 and float64, as an ndarray or a DenseVector; k from 1 to 10. "
    "Needs strict IEEE arithmetic: no -ffast-math.";

constexpr const char* kDotDoc =
    "The dot product of x and y as if computed in k-fold working precision, "
    "rounded once: Ogita-Rump-Oishi Dot2 for k = 2 (the default), DotK for "
    "k >= 3.\n\n"
    "The error is about u abs(x.y) + (phi u)^k abs(x).abs(y). Dot2 is the same "
    "loop as "
    "mtl5.mixed.dot(x, y, accumulator='compensated'), and gives the same bits. "
    "For larger k it keeps 2n error terms. For the correctly rounded value, at "
    "any condition number, use accumulator='exact'.\n\n"
    "float32 and float64 vectors of one type, as ndarrays or DenseVectors; "
    "k from 2 to 10. Needs strict IEEE arithmetic: no -ffast-math.";

template <typename T>
void register_for(nb::module_& m) {
    // noconvert: nanobind's conversion pass would otherwise hand a strided or
    // mixed-dtype array to the first overload that accepts a cast, which can
    // change the working precision. The Python wrappers in mtl5/__init__.py
    // normalise the layout, keeping the dtype, before calling these.
    m.def("sum_k", [](Arr1<T> x, int k) { return sum_k_impl<T>(borrow<T>(x), k); },
          "x"_a.noconvert(), "k"_a = 2, kSumDoc);
    m.def("sum_k", [](const VectorView<T>& x, int k) { return sum_k_impl<T>(x.vec, k); },
          "x"_a, "k"_a = 2);
    m.def("dot_k",
          [](Arr1<T> x, Arr1<T> y, int k) { return dot_k_impl<T>(borrow<T>(x), borrow<T>(y), k); },
          "x"_a.noconvert(), "y"_a.noconvert(), "k"_a = 2, kDotDoc);
    m.def("dot_k",
          [](const VectorView<T>& x, const VectorView<T>& y, int k) {
              return dot_k_impl<T>(x.vec, y.vec, k);
          },
          "x"_a, "y"_a, "k"_a = 2);
}

}  // namespace

void register_accurate(nb::module_& m) {
    register_for<double>(m);
    register_for<float>(m);
}
