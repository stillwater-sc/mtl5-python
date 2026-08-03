// mtl5-python -- the N-dimensional array layer (mtl/array).
//
// `mtl::array::ndarray<T, N, Order>` carries its rank as a template parameter,
// so a binding has to fix the ranks up front. Ranks 1-4 are instantiated for
// float32 and float64; `mtl5.array.asarray` dispatches on the input's ndim and
// refuses rank 5+ rather than silently flattening.
//
// Two upstream behaviours are worked around rather than passed through.
//
// **reshape.** `ndarray::reshape` guards on `is_contiguous()`, which means
// "contiguous in *either* C or F order" (`shape.hpp:158`), and then builds the
// result using the array's own `Order`. A transposed C-order array is
// F-contiguous, so it passes that guard and is then read back with C-order
// strides. Measured: `[[1,2,3],[4,5,6]].transpose().reshape(6)` gives
// `[1,2,3,4,5,6]` -- raw memory order -- where NumPy gives `[1,4,2,5,3,6]`, and
// nothing is raised. So `reshape` here checks contiguity in the array's *own*
// order, returns a view when that holds and materialises a logical-order copy
// otherwise. That is also NumPy's contract, which never errors on reshape.
//
// **Slicing.** `mtl::array::slice` derives the result rank at compile time from
// the argument *types* (`count_kept_dims<Args...>`), so a Python caller, whose
// index tuple is a runtime value, cannot reach it without instantiating every
// combination of {int, range, all} over every rank -- 3^N per rank per dtype.
// `__getitem__` instead computes the sliced shape, strides and base offset and
// builds the view through ndarray's (pointer, shape, strides) constructor,
// which is the same primitive `slice` is built on.
//
// Lifetime: every method returning a view uses nb::keep_alive<0, 1>, so the
// array it borrows from cannot be collected first. `asarray` additionally holds
// the NumPy object, which is what keeps the underlying buffer alive.

#include "mtl5_types.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mtl/array/broadcast.hpp>
#include <mtl/array/ndarray.hpp>
#include <mtl/array/operations.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

namespace ma = mtl::array;

/// An ndarray plus, when it borrows NumPy memory, the object owning it.
///
/// `owner` defaults to a NULL handle rather than `nb::none()`. That matters:
/// `copy`, the arithmetic operators and the axis reductions all construct an
/// NDArrayView inside a `nogil` scope, and `nb::none()` would incref Py_None
/// with the GIL released -- a data race on any interpreter that refcounts it
/// (CPython < 3.12, and free-threaded builds). A NULL handle touches nothing:
/// nanobind's copy path is Py_XINCREF, which is a no-op on null.
template <typename T, std::size_t N>
struct NDArrayView {
    ma::ndarray<T, N> arr;
    nb::object owner;   // NULL when this array owns its memory

    NDArrayView() = default;
    explicit NDArrayView(ma::ndarray<T, N> a, nb::object o = nb::object())
        : arr(std::move(a)), owner(std::move(o)) {}

    bool is_view() const { return arr.is_view() || owner.is_valid(); }
};

template <std::size_t N>
ma::shape<N> shape_from(const std::vector<std::size_t>& v) {
    if (v.size() != N)
        throw nb::value_error(
            ("shape must have " + std::to_string(N) + " entries").c_str());
    ma::shape<N> sh;
    for (std::size_t i = 0; i < N; ++i) sh[i] = v[i];
    return sh;
}

template <typename T, std::size_t N>
std::vector<std::size_t> shape_vec(const ma::ndarray<T, N>& a) {
    std::vector<std::size_t> out(N);
    for (std::size_t i = 0; i < N; ++i) out[i] = a.extent(i);
    return out;
}

/// Contiguity in the array's OWN order. `ndarray::is_contiguous()` answers the
/// weaker "in either order", which is what makes its reshape unsound.
template <typename T, std::size_t N>
bool is_c_contiguous(const ma::ndarray<T, N>& a) {
    return a.get_strides() == ma::c_order_strides(a.get_shape());
}

/// Visit every element in logical (C) order, passing its memory offset.
template <typename T, std::size_t N, typename F>
void each_offset(const ma::ndarray<T, N>& a, F&& f) {
    const auto& sh = a.get_shape();
    const auto& st = a.get_strides();
    std::array<std::size_t, N> idx{};
    const std::size_t total = a.size();
    for (std::size_t k = 0; k < total; ++k) {
        std::size_t off = 0;
        for (std::size_t d = 0; d < N; ++d) off += idx[d] * st[d];
        f(off);
        for (std::size_t d = N; d-- > 0;) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
}

/// A fresh C-contiguous array holding the elements in logical order.
template <typename T, std::size_t N>
ma::ndarray<T, N> dense_copy(const ma::ndarray<T, N>& a) {
    ma::ndarray<T, N> out(a.get_shape());
    const T* src = a.data();
    T* dst = out.data();
    std::size_t k = 0;
    each_offset(a, [&](std::size_t off) { dst[k++] = src[off]; });
    return out;
}

/// One axis of an index expression: either a dropped integer or a kept range.
struct AxisSpec {
    bool keep;
    std::size_t start;
    std::size_t count;
    std::size_t step;
};

/// Resolve a Python index tuple against `extents`. Missing trailing entries
/// mean "the whole axis", as in NumPy.
template <std::size_t N>
std::array<AxisSpec, N> resolve_key(nb::handle key,
                                    const std::array<std::size_t, N>& extents) {
    std::vector<nb::object> items;
    if (nb::isinstance<nb::tuple>(key)) {
        for (nb::handle h : nb::cast<nb::tuple>(key)) items.push_back(nb::borrow(h));
    } else {
        items.push_back(nb::borrow(key));
    }
    if (items.size() > N)
        throw nb::index_error("too many indices for array");

    std::array<AxisSpec, N> out{};
    for (std::size_t d = 0; d < N; ++d) {
        if (d >= items.size()) {                      // implicit trailing ':'
            out[d] = {true, 0, extents[d], 1};
            continue;
        }
        nb::handle it = items[d];
        if (nb::isinstance<nb::slice>(it)) {
            auto [start, stop, step, count] =
                nb::cast<nb::slice>(it).compute(extents[d]);
            if (step <= 0)
                throw nb::value_error(
                    "negative or zero slice steps are not supported -- MTL5 "
                    "strides are unsigned; use .copy() and slice in NumPy");
            out[d] = {true, static_cast<std::size_t>(start),
                      static_cast<std::size_t>(count),
                      static_cast<std::size_t>(step)};
        } else {
            Py_ssize_t i = nb::cast<Py_ssize_t>(it);
            if (i < 0) i += static_cast<Py_ssize_t>(extents[d]);
            if (i < 0 || static_cast<std::size_t>(i) >= extents[d])
                throw nb::index_error(
                    ("index out of range for axis " + std::to_string(d)).c_str());
            out[d] = {false, static_cast<std::size_t>(i), 1, 1};
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
template <typename T, std::size_t N>
void register_ndarray(nb::module_& m) {
    using NV = NDArrayView<T, N>;
    const std::string name =
        "NDArray_" + std::to_string(N) + "d_" + type_suffix<T>();

    auto cls = nb::class_<NV>(m, name.c_str());

    cls.def_prop_ro("ndim", [](const NV&) { return N; })
        .def_prop_ro("shape", [](const NV& v) { return shape_vec(v.arr); })
        .def_prop_ro("strides", [](const NV& v) {
            std::vector<std::size_t> out(N);
            for (std::size_t i = 0; i < N; ++i) out[i] = v.arr.get_strides()[i];
            return out;
        }, "Strides in ELEMENTS, which is how MTL5 holds them. NumPy's .strides "
           "is in bytes; the two differ by itemsize.")
        .def_prop_ro("size", [](const NV& v) { return v.arr.size(); })
        .def_prop_ro("dtype", [](const NV&) { return type_suffix<T>(); })
        .def_prop_ro("is_view", [](const NV& v) { return v.is_view(); })
        .def_prop_ro("is_c_contiguous", [](const NV& v) { return is_c_contiguous(v.arr); })
        .def_prop_ro("is_f_contiguous", [](const NV& v) {
            return v.arr.get_strides() == ma::f_order_strides(v.arr.get_shape());
        })
        .def("__len__", [](const NV& v) { return v.arr.extent(0); })

        .def("to_numpy", [](nb::handle self) {
            // Zero-copy, strides included, so a transposed or sliced view maps
            // onto NumPy without materialising. nanobind's strides are in
            // elements, matching MTL5's.
            NV& v = nb::cast<NV&>(self);
            std::size_t shape[N];
            int64_t strides[N];
            for (std::size_t i = 0; i < N; ++i) {
                shape[i] = v.arr.extent(i);
                strides[i] = static_cast<int64_t>(v.arr.get_strides()[i]);
            }
            return nb::ndarray<nb::numpy, T, nb::ndim<N>>(
                v.arr.data(), N, shape, self, strides);
        }, "Zero-copy NumPy view, strides included")

        .def("copy", [](const NV& v) {
            nogil guard;
            return NV(dense_copy(v.arr));
        }, "An owning, C-contiguous copy in logical element order")

        .def("fill", [](NV& v, T val) {
            nogil guard;
            T* p = v.arr.data();
            each_offset(v.arr, [&](std::size_t off) { p[off] = val; });
        }, "value"_a, "Set every element, honouring strides")

        // -- whole-array reductions -----------------------------------------
        .def("sum", [](const NV& v) { nogil guard; return ma::sum(v.arr); })
        .def("prod", [](const NV& v) { nogil guard; return ma::prod(v.arr); })
        .def("mean", [](const NV& v) {
            if (v.arr.size() == 0) throw nb::value_error("mean of an empty array");
            nogil guard;
            return ma::mean(v.arr);
        })
        .def("min", [](const NV& v) {
            if (v.arr.size() == 0) throw nb::value_error("min of an empty array");
            nogil guard;
            return ma::min(v.arr);
        })
        .def("max", [](const NV& v) {
            if (v.arr.size() == 0) throw nb::value_error("max of an empty array");
            nogil guard;
            return ma::max(v.arr);
        })

        // -- elementwise -----------------------------------------------------
        // Same-rank broadcasting only: an extent of 1 stretches, but NumPy's
        // rank promotion does not, because broadcast_shape takes two shape<N>
        // of equal N. Reshape the operand first if you need that.
        .def("__add__", [](const NV& a, const NV& b) {
            nogil guard; return NV((a.arr + b.arr).eval());
        })
        .def("__sub__", [](const NV& a, const NV& b) {
            nogil guard; return NV((a.arr - b.arr).eval());
        })
        .def("__mul__", [](const NV& a, const NV& b) {
            nogil guard; return NV((a.arr * b.arr).eval());
        })
        .def("__truediv__", [](const NV& a, const NV& b) {
            nogil guard; return NV((a.arr / b.arr).eval());
        })

        .def("__repr__", [](const NV& v) {
            std::ostringstream os;
            os << "mtl5.array.NDArray_" << N << "d_" << type_suffix<T>() << "(shape=(";
            for (std::size_t i = 0; i < N; ++i) {
                if (i) os << ", ";
                os << v.arr.extent(i);
            }
            if (N == 1) os << ",";
            os << ")" << (v.is_view() ? ", view" : "") << ")";
            return os.str();
        });

    // -- transpose (a view: shape and strides reversed) ----------------------
    cls.def("transpose", [](NV& v) { return NV(v.arr.transpose(), v.owner); },
            nb::keep_alive<0, 1>(), "Reverse the axes. Returns a view.");
    cls.def_prop_ro("T", [](NV& v) { return NV(v.arr.transpose(), v.owner); },
                    nb::keep_alive<0, 1>(), "Reverse the axes. Returns a view.");

    // -- element / slice access ---------------------------------------------
    // Signed, so that a negative index means the same thing it does in
    // __getitem__ (which normalises through resolve_key).
    cls.def("__setitem__", [](NV& v, const std::vector<Py_ssize_t>& idx, T val) {
        if (idx.size() != N)
            throw nb::index_error("assignment needs one index per axis");
        std::size_t off = 0;
        for (std::size_t d = 0; d < N; ++d) {
            Py_ssize_t i = idx[d];
            if (i < 0) i += static_cast<Py_ssize_t>(v.arr.extent(d));
            if (i < 0 || static_cast<std::size_t>(i) >= v.arr.extent(d))
                throw nb::index_error(
                    ("index out of range for axis " + std::to_string(d)).c_str());
            off += static_cast<std::size_t>(i) * v.arr.get_strides()[d];
        }
        v.arr.data()[off] = val;
    });

    cls.def("__getitem__", [](nb::handle self, nb::handle key) -> nb::object {
        NV& v = nb::cast<NV&>(self);
        std::array<std::size_t, N> extents{};
        for (std::size_t d = 0; d < N; ++d) extents[d] = v.arr.extent(d);
        const auto spec = resolve_key<N>(key, extents);

        std::size_t base = 0, kept = 0;
        for (std::size_t d = 0; d < N; ++d) {
            base += spec[d].start * v.arr.get_strides()[d];
            if (spec[d].keep) ++kept;
        }
        T* ptr = const_cast<T*>(v.arr.data()) + base;

        // Every axis indexed by an integer: a scalar, as in NumPy.
        if (kept == 0) return nb::cast(*ptr);

        // Otherwise a view whose rank is `kept` -- known only at runtime, so
        // dispatch over the instantiated ranks.
        auto build = [&](auto tag) -> nb::object {
            constexpr std::size_t M = decltype(tag)::value;
            ma::shape<M> sh;
            std::array<std::size_t, M> st{};
            std::size_t j = 0;
            for (std::size_t d = 0; d < N; ++d) {
                if (!spec[d].keep) continue;
                sh[j] = spec[d].count;
                st[j] = v.arr.get_strides()[d] * spec[d].step;
                ++j;
            }
            NDArrayView<T, M> out(ma::ndarray<T, M>(ptr, sh, st), v.owner);
            nb::object o = nb::cast(std::move(out));
            nb::detail::keep_alive(o.ptr(), self.ptr());
            return o;
        };
        switch (kept) {
            case 1: return build(std::integral_constant<std::size_t, 1>{});
            case 2: return build(std::integral_constant<std::size_t, 2>{});
            case 3: return build(std::integral_constant<std::size_t, 3>{});
            default: return build(std::integral_constant<std::size_t, 4>{});
        }
    }, "Index with ints and slices. An int drops its axis, a slice keeps it, "
       "and the result is a view -- never a copy. Negative steps are not "
       "supported (MTL5 strides are unsigned).");

    // -- reshape / ravel / flatten -------------------------------------------
    cls.def("reshape", [](nb::handle self, const std::vector<std::size_t>& want) -> nb::object {
        NV& v = nb::cast<NV&>(self);
        auto build = [&](auto tag) -> nb::object {
            constexpr std::size_t M = decltype(tag)::value;
            const auto sh = shape_from<M>(want);
            if (sh.total_size() != v.arr.size())
                throw nb::value_error(
                    ("cannot reshape an array of size " + std::to_string(v.arr.size()) +
                     " into the requested shape").c_str());
            if (is_c_contiguous(v.arr)) {
                NDArrayView<T, M> out(
                    ma::ndarray<T, M>(const_cast<T*>(v.arr.data()), sh), v.owner);
                nb::object o = nb::cast(std::move(out));
                nb::detail::keep_alive(o.ptr(), self.ptr());
                return o;
            }
            // Not contiguous in its own order: pack into logical order first.
            // This is where ndarray::reshape would have returned raw memory
            // order instead -- see the file header.
            auto packed = dense_copy(v.arr);
            NDArrayView<T, M> out{ma::ndarray<T, M>(sh)};
            std::copy_n(packed.data(), packed.size(), out.arr.data());
            return nb::cast(std::move(out));
        };
        switch (want.size()) {
            case 1: return build(std::integral_constant<std::size_t, 1>{});
            case 2: return build(std::integral_constant<std::size_t, 2>{});
            case 3: return build(std::integral_constant<std::size_t, 3>{});
            case 4: return build(std::integral_constant<std::size_t, 4>{});
            default: throw nb::value_error("reshape supports ranks 1 through 4");
        }
    }, "shape"_a,
       "Reshape to any rank in 1..4. A view when the array is C-contiguous, a "
       "copy otherwise -- as NumPy does, and never an error.");

    // MTL5's `flatten` has the same root cause as its reshape: it walks
    // elements through `for_each_element`, whose fast path triggers on
    // `is_contiguous()` (either order) and then indexes raw memory. On an
    // F-contiguous array -- any transpose of a C array -- that yields memory
    // order rather than logical order. Measured: flatten of the transpose of
    // [[1,2,3],[4,5,6]] gives [1,2,3,4,5,6] where NumPy gives [1,4,2,5,3,6].
    // (The commutative reductions are unaffected: sum/prod/min/max/mean do not
    // care what order they visit in.) Both entry points below go through the
    // logical-order path instead.
    cls.def("ravel", [](nb::handle self) {
        NV& v = nb::cast<NV&>(self);
        if (is_c_contiguous(v.arr)) {
            ma::shape<1> sh;
            sh[0] = v.arr.size();
            NDArrayView<T, 1> out(
                ma::ndarray<T, 1>(const_cast<T*>(v.arr.data()), sh), v.owner);
            nb::object o = nb::cast(std::move(out));
            nb::detail::keep_alive(o.ptr(), self.ptr());
            return o;
        }
        ma::shape<1> sh;
        sh[0] = v.arr.size();
        NDArrayView<T, 1> out{ma::ndarray<T, 1>(sh)};
        const T* src = v.arr.data();
        T* dst = out.arr.data();
        std::size_t k = 0;
        each_offset(v.arr, [&](std::size_t off) { dst[k++] = src[off]; });
        return nb::cast(std::move(out));
    }, "Flatten to rank 1, returning a view when the array is C-contiguous and "
       "a copy otherwise -- NumPy's ravel.");

    cls.def("flatten", [](const NV& v) {
        ma::shape<1> sh;
        sh[0] = v.arr.size();
        NDArrayView<T, 1> out{ma::ndarray<T, 1>(sh)};
        {
            nogil guard;
            const T* src = v.arr.data();
            T* dst = out.arr.data();
            std::size_t k = 0;
            each_offset(v.arr, [&](std::size_t off) { dst[k++] = src[off]; });
        }
        return out;
    }, "Flatten to rank 1. Always a copy, as NumPy's flatten is.");

    // -- axis reductions (rank >= 2; the result would otherwise be rank 0) ---
    if constexpr (N >= 2) {
        cls.def("sum_axis", [](const NV& v, std::size_t axis) {
            if (axis >= N) throw nb::value_error("axis out of range");
            nogil guard;
            return NDArrayView<T, N - 1>(ma::sum_axis(v.arr, axis));
        }, "axis"_a, "Sum along one axis, giving a rank N-1 array");

        cls.def("mean_axis", [](const NV& v, std::size_t axis) {
            if (axis >= N) throw nb::value_error("axis out of range");
            if (v.arr.extent(axis) == 0)
                throw nb::value_error("mean along a zero-length axis is undefined");
            nogil guard;
            return NDArrayView<T, N - 1>(ma::mean_axis(v.arr, axis));
        }, "axis"_a, "Mean along one axis, giving a rank N-1 array");
    }

    // -- zero-copy factory from NumPy ---------------------------------------
    // Any strided layout is accepted, so a NumPy transpose or slice comes
    // through without a copy.
    // .noconvert() is load-bearing, not a nicety. Without it nanobind's second
    // pass converts, and because the float overloads are registered first an
    // int64 array silently became a float32 array -- while still reporting
    // is_view=True, since the view is of the converted temporary. That breaks
    // both the dtype and the zero-copy contract at once. Requiring an exact
    // match makes anything else a TypeError.
    m.def("asarray", [](nb::ndarray<T, nb::ndim<N>, nb::device::cpu> a) {
        ma::shape<N> sh;
        std::array<std::size_t, N> st{};
        for (std::size_t i = 0; i < N; ++i) {
            sh[i] = a.shape(i);
            const int64_t s = a.stride(i);
            if (s < 0)
                throw nb::value_error(
                    "negative strides are not supported -- pass "
                    "np.ascontiguousarray(a) instead");
            st[i] = static_cast<std::size_t>(s);
        }
        return NDArrayView<T, N>(ma::ndarray<T, N>(a.data(), sh, st), nb::cast(a));
    }, nb::arg("a").noconvert(),
       "Zero-copy MTL5 ndarray view of a NumPy array (any strided layout).\n\n"
       "The dtype must match exactly -- float32 for an f32 array, float64 for "
       "an f64 one. Anything else raises TypeError rather than converting, "
       "because a converted array would be a view of a temporary and neither "
       "zero-copy nor the dtype you asked for.\n\n"
       "The array must also be writable: the view aliases it, so a read-only "
       "NumPy array raises TypeError. Pass a.copy() if that is what you have.");
}

// ---------------------------------------------------------------------------
// zeros(shape, dtype) -- one entry point, dispatching on len(shape).
// ---------------------------------------------------------------------------
template <typename T>
nb::object zeros_for(const std::vector<std::size_t>& want) {
    auto build = [&](auto tag) -> nb::object {
        constexpr std::size_t M = decltype(tag)::value;
        return nb::cast(NDArrayView<T, M>(ma::ndarray<T, M>(shape_from<M>(want))));
    };
    switch (want.size()) {
        case 1: return build(std::integral_constant<std::size_t, 1>{});
        case 2: return build(std::integral_constant<std::size_t, 2>{});
        case 3: return build(std::integral_constant<std::size_t, 3>{});
        case 4: return build(std::integral_constant<std::size_t, 4>{});
        default: throw nb::value_error("zeros supports ranks 1 through 4");
    }
}

// ---------------------------------------------------------------------------
// Interop with the dense containers. This is what earns the layer its place
// here: it makes a DenseMatrix sliceable along either axis without a copy.
// ---------------------------------------------------------------------------
template <typename T>
void register_interop(nb::module_& m) {
    m.def("as_ndarray", [](MatrixView<T>& mv) {
        ma::shape<2> sh;
        sh[0] = mv.mat.num_rows();
        sh[1] = mv.mat.num_cols();
        return NDArrayView<T, 2>(ma::ndarray<T, 2>(mv.mat.data(), sh));
    }, "A"_a, nb::keep_alive<0, 1>(),
       "Zero-copy rank-2 ndarray view of a DenseMatrix");

    m.def("as_ndarray", [](VectorView<T>& vv) {
        ma::shape<1> sh;
        sh[0] = vv.vec.size();
        return NDArrayView<T, 1>(ma::ndarray<T, 1>(vv.vec.data(), sh));
    }, "x"_a, nb::keep_alive<0, 1>(),
       "Zero-copy rank-1 ndarray view of a DenseVector");

    m.def("as_matrix", [](const NDArrayView<T, 2>& v) {
        mtl::mat::dense2D<T> out(v.arr.extent(0), v.arr.extent(1));
        {
            nogil guard;
            const T* src = v.arr.data();
            T* dst = out.data();
            std::size_t k = 0;
            each_offset(v.arr, [&](std::size_t off) { dst[k++] = src[off]; });
        }
        return MatrixView<T>(std::move(out));
    }, "a"_a, "Copy a rank-2 ndarray into a DenseMatrix (strides honoured)");

    m.def("as_vector", [](const NDArrayView<T, 1>& v) {
        mtl::vec::dense_vector<T> out(v.arr.size());
        {
            nogil guard;
            const T* src = v.arr.data();
            T* dst = out.data();
            std::size_t k = 0;
            each_offset(v.arr, [&](std::size_t off) { dst[k++] = src[off]; });
        }
        return VectorView<T>(std::move(out));
    }, "a"_a, "Copy a rank-1 ndarray into a DenseVector (strides honoured)");
}

}  // namespace

// ===========================================================================
void register_ndarray_layer(nb::module_& m) {
    nb::module_ arr = m.def_submodule(
        "array", "N-dimensional arrays (mtl/array), ranks 1-4");

    register_ndarray<float, 1>(arr);
    register_ndarray<float, 2>(arr);
    register_ndarray<float, 3>(arr);
    register_ndarray<float, 4>(arr);
    register_ndarray<double, 1>(arr);
    register_ndarray<double, 2>(arr);
    register_ndarray<double, 3>(arr);
    register_ndarray<double, 4>(arr);

    register_interop<float>(arr);
    register_interop<double>(arr);

    arr.def("zeros", [](const std::vector<std::size_t>& shape,
                        const std::string& dtype) {
        if (dtype == "f64") return zeros_for<double>(shape);
        if (dtype == "f32") return zeros_for<float>(shape);
        throw nb::value_error(
            "dtype must be 'f32' or 'f64' -- the array layer is instantiated "
            "for the native floating-point types only");
    }, "shape"_a, "dtype"_a = "f64", "A zero-filled owning array, rank 1-4");
}
