// mtl5-python -- index-notation tensor algebra (mtl/tensor).
//
// Distinct from mtl5.array: that is NumPy-shaped N-D data with a runtime shape,
// this is fixed-dimension tensor algebra with Einstein summation and a metric.
//
// `tensor<Value, Rank, Dim>` fixes BOTH rank and dimension at compile time and
// stores its components on the stack, so every (Rank, Dim) pair is a separate
// type. The enumeration is smaller than it looks: nothing upstream uses rank 3,
// and rank 4 arises only as the result of outer(rank2, rank2). So ranks 1, 2
// and 4 over dimensions 2, 3 and 4 cover the whole module.
//
// **Contraction was the interesting part to bind.** `contract(bind(A, i, j),
// bind(B, j, k))` takes its index names as compile-time `char` template
// parameters and finds the repeated one with a `static_assert`, so a runtime
// index string cannot reach it -- the same wall as mtl::array::slice. But
// unlike slice, the space is tiny: the repeated index sits in one of two
// positions on each side, so four instantiations cover every rank2 x rank2
// contraction up to relabelling and two cover rank2 x rank1. This dispatches
// among those, which means Python reaches MTL5's own contraction rather than a
// reimplementation of it.
//
// `raise` is a Python keyword, so the metric operations are exposed as
// `raise_index` / `lower_index` for the rank-1 pair, matching the
// `raise_first` / `lower_second` naming upstream already uses for rank 2.

#include "mtl5_types.hpp"

#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mtl/tensor/index.hpp>
#include <mtl/tensor/metric.hpp>
#include <mtl/tensor/properties.hpp>
#include <mtl/tensor/symmetric.hpp>
#include <mtl/tensor/tensor.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nb::literals;

namespace {

namespace mt = mtl::tensor;

template <typename T, std::size_t R, std::size_t D>
using Tn = mt::tensor<T, R, D>;

std::string type_name(std::size_t rank, std::size_t dim, const char* suffix) {
    return "Tensor_r" + std::to_string(rank) + "_d" + std::to_string(dim) + "_" +
           suffix;
}

/// Flat component index from a multi-index, row-major over Rank axes of Dim.
template <std::size_t R, std::size_t D>
std::size_t flat(const std::array<std::size_t, R>& idx) {
    std::size_t f = 0;
    for (std::size_t i = 0; i < R; ++i) f = f * D + idx[i];
    return f;
}

template <typename T, std::size_t R, std::size_t D>
void check_index(const std::array<std::size_t, R>& idx) {
    for (std::size_t i = 0; i < R; ++i)
        if (idx[i] >= D)
            throw nb::index_error(
                ("index " + std::to_string(idx[i]) + " out of range for dimension " +
                 std::to_string(D)).c_str());
}

// ---------------------------------------------------------------------------
// Container
// ---------------------------------------------------------------------------
template <typename T, std::size_t R, std::size_t D>
void register_tensor(nb::module_& m) {
    using Ten = Tn<T, R, D>;
    const std::string name = type_name(R, D, type_suffix<T>());

    nb::class_<Ten>(m, name.c_str())
        .def_prop_ro_static("rank", [](nb::handle) { return R; })
        .def_prop_ro_static("dimension", [](nb::handle) { return D; })
        .def_prop_ro("shape", [](const Ten&) {
            return std::vector<std::size_t>(R, D);
        })
        .def_prop_ro("dtype", [](const Ten&) { return type_suffix<T>(); })
        .def_prop_ro("size", [](const Ten&) { return Ten::num_components; })
        .def("__getitem__", [](const Ten& t, std::array<std::size_t, R> idx) -> T {
            check_index<T, R, D>(idx);
            return t[idx];
        })
        .def("__setitem__", [](Ten& t, std::array<std::size_t, R> idx, T v) {
            check_index<T, R, D>(idx);
            t[idx] = v;
        })
        .def("to_numpy", [](const Ten& t) {
            // A fresh buffer: the tensor's storage is on the stack inside the
            // object, so a view would tie NumPy's lifetime to it for no gain at
            // these sizes (at most Dim^Rank components).
            const std::size_t n = Ten::num_components;
            T* buf = new T[n];
            std::array<std::size_t, R> idx{};
            for (std::size_t k = 0; k < n; ++k) {
                buf[k] = t[idx];
                for (std::size_t d = R; d-- > 0;) {
                    if (++idx[d] < D) break;
                    idx[d] = 0;
                }
            }
            std::size_t shape[R];
            for (std::size_t i = 0; i < R; ++i) shape[i] = D;
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
            return nb::ndarray<nb::numpy, T, nb::ndim<R>>(buf, R, shape, owner);
        }, "Components as a NumPy array of shape (dim,) * rank")
        .def("__repr__", [name](const Ten&) {
            return "mtl5.tensor." + name + "()";
        });

    // Build from a NumPy array whose shape is exactly (D,) * R.
    m.def("_from_numpy", [](nb::ndarray<T, nb::ndim<R>, nb::c_contig,
                            nb::device::cpu> a) {
        // Rank comes from ndim, but every (Rank, Dim) pair registers the same
        // ndarray signature, so nanobind cannot tell them apart by type. Fall
        // through until the extents match one.
        for (std::size_t i = 0; i < R; ++i)
            if (a.shape(i) != D) throw nb::next_overload();
        Ten t;
        const T* src = a.data();
        std::array<std::size_t, R> idx{};
        for (std::size_t k = 0; k < Ten::num_components; ++k) {
            t[idx] = src[k];
            for (std::size_t d = R; d-- > 0;) {
                if (++idx[d] < D) break;
                idx[d] = 0;
            }
        }
        return t;
    }, "a"_a);

}

// ---------------------------------------------------------------------------
// Contraction: dispatch on where the repeated index sits
// ---------------------------------------------------------------------------
/// Position of `c` in `s`, or -1.
int pos_of(const std::string& s, char c) {
    for (std::size_t i = 0; i < s.size(); ++i)
        if (s[i] == c) return static_cast<int>(i);
    return -1;
}

/// The single index appearing in both, rejecting the ambiguous cases up front
/// so the error names the problem rather than a static_assert firing inside.
char repeated_index(const std::string& sa, const std::string& sb) {
    // Check for an index repeated within one tensor first: 'ii' also shows up
    // as a doubled shared index, and "this is a trace" is the more useful
    // message than "more than one index is repeated".
    for (const std::string* s : {&sa, &sb}) {
        for (std::size_t i = 0; i < s->size(); ++i)
            for (std::size_t j = i + 1; j < s->size(); ++j)
                if ((*s)[i] == (*s)[j])
                    throw nb::value_error(
                        "contract: an index repeats within one tensor, which is "
                        "a trace rather than a contraction and is not supported.");
    }
    std::vector<char> shared;
    for (char c : sa)
        if (pos_of(sb, c) >= 0) shared.push_back(c);
    if (shared.empty())
        throw nb::value_error(
            "contract: the two index strings share no index, so there is "
            "nothing to sum over. Use outer() for a tensor product.");
    if (shared.size() > 1)
        throw nb::value_error(
            "contract: more than one index is repeated. MTL5 contracts over a "
            "single index; apply contract twice, or use mtl5.array for general "
            "multi-index summation.");
    return shared.front();
}

template <typename T, std::size_t D>
void register_contract(nb::module_& m) {
    using T2 = Tn<T, 2, D>;
    using T1 = Tn<T, 1, D>;
    // Four representative letter choices. Any rank2 x rank2 contraction is one
    // of these up to relabelling, so dispatching on the repeated index's
    // position reaches MTL5's own contract rather than reimplementing it.
    mt::Index<'i'> I; mt::Index<'j'> J; mt::Index<'k'> K;

    m.def("contract", [I, J, K](const T2& a, const std::string& sa,
                                const T2& b, const std::string& sb) {
        if (sa.size() != 2 || sb.size() != 2)
            throw nb::value_error("contract: index strings must match the ranks");
        const char r = repeated_index(sa, sb);
        const int pa = pos_of(sa, r), pb = pos_of(sb, r);
        if (pa == 1 && pb == 0) return contract(bind(a, I, J), bind(b, J, K));
        if (pa == 1 && pb == 1) return contract(bind(a, I, J), bind(b, K, J));
        if (pa == 0 && pb == 0) return contract(bind(a, J, I), bind(b, J, K));
        return                         contract(bind(a, J, I), bind(b, K, J));
    }, "a"_a, "a_indices"_a, "b"_a, "b_indices"_a,
       "Contract two rank-2 tensors over their shared index, e.g. "
       "contract(A, 'ij', B, 'jk'). The index letters are arbitrary; only which "
       "positions they share matters.");

    m.def("contract", [I, J](const T2& a, const std::string& sa,
                             const T1& b, const std::string& sb) {
        if (sa.size() != 2 || sb.size() != 1)
            throw nb::value_error("contract: index strings must match the ranks");
        const char r = repeated_index(sa, sb);
        const int pa = pos_of(sa, r);
        if (pa == 1) return contract(bind(a, I, J), bind(b, J));
        return              contract(bind(a, J, I), bind(b, J));
    }, "a"_a, "a_indices"_a, "b"_a, "b_indices"_a,
       "Contract a rank-2 tensor with a rank-1 tensor, e.g. "
       "contract(A, 'ij', x, 'j') for A @ x or contract(A, 'ji', x, 'j') for "
       "A.T @ x.");
}

// ---------------------------------------------------------------------------
// Metric, outer products and properties
// ---------------------------------------------------------------------------
template <typename T, std::size_t D>
void register_algebra(nb::module_& m) {
    using T1 = Tn<T, 1, D>;
    using T2 = Tn<T, 2, D>;

    m.def("outer", [](const T1& a, const T1& b) { return mt::outer(a, b); },
          "a"_a, "b"_a, "Outer product of two rank-1 tensors, giving rank 2");
    m.def("outer", [](const T2& a, const T2& b) { return mt::outer(a, b); },
          "a"_a, "b"_a, "Outer product of two rank-2 tensors, giving rank 4");

    m.def("lower_index", [](const T1& v, const T2& g) { return mt::lower(v, g); },
          "v"_a, "g"_a,
          "Lower a rank-1 index with the metric: T_i = g_ij T^j");
    m.def("raise_index", [](const T1& v, const T2& g_inv) {
        return mt::raise(v, g_inv);
    }, "v"_a, "g_inv"_a,
       "Raise a rank-1 index with the inverse metric: T^i = g^ij T_j. Named "
       "raise_index because `raise` is a Python keyword.");

    m.def("lower_first", [](const T2& t, const T2& g) {
        return mt::lower_first(t, g);
    }, "t"_a, "g"_a);
    m.def("lower_second", [](const T2& t, const T2& g) {
        return mt::lower_second(t, g);
    }, "t"_a, "g"_a);
    m.def("raise_first", [](const T2& t, const T2& g_inv) {
        return mt::raise_first(t, g_inv);
    }, "t"_a, "g_inv"_a);
    m.def("raise_second", [](const T2& t, const T2& g_inv) {
        return mt::raise_second(t, g_inv);
    }, "t"_a, "g_inv"_a);

    m.def("is_symmetric", [](const T2& t, double tol) {
        return mt::is_symmetric(t, static_cast<T>(tol));
    }, "t"_a, "tol"_a = 0.0, "True if t(i, j) == t(j, i) within tol");
    m.def("is_antisymmetric", [](const T2& t, double tol) {
        return mt::is_antisymmetric(t, static_cast<T>(tol));
    }, "t"_a, "tol"_a = 0.0,
       "True if t(i, j) == -t(j, i) within tol, which also requires a vanishing "
       "diagonal");

    // Packed symmetric storage: D(D+1)/2 components instead of D^2.
    using Sym = mt::symmetric_tensor<T, D>;
    const std::string sname =
        "SymmetricTensor_d" + std::to_string(D) + "_" + type_suffix<T>();
    nb::class_<Sym>(m, sname.c_str())
        .def(nb::init<>())
        .def_prop_ro("dimension", [](const Sym&) { return D; })
        .def_prop_ro("dtype", [](const Sym&) { return type_suffix<T>(); })
        .def_prop_ro("num_stored", [](const Sym&) { return Sym::num_stored; },
                     "Components actually held: D(D+1)/2, against D^2 logical")
        .def("__getitem__", [](const Sym& s, std::pair<std::size_t, std::size_t> ij) -> T {
            if (ij.first >= D || ij.second >= D) throw nb::index_error();
            return s(ij.first, ij.second);
        })
        .def("__setitem__", [](Sym& s, std::pair<std::size_t, std::size_t> ij, T v) {
            if (ij.first >= D || ij.second >= D) throw nb::index_error();
            s(ij.first, ij.second) = v;
        })
        .def("to_dense", [](const Sym& s) {
            T2 out;
            for (std::size_t i = 0; i < D; ++i)
                for (std::size_t j = 0; j < D; ++j)
                    out(i, j) = s(i, j);
            return out;
        }, "Expand to a full rank-2 tensor")
        .def("__repr__", [sname](const Sym&) {
            return "mtl5.tensor." + sname + "()";
        });
}

template <typename T, std::size_t D>
void register_for_dim(nb::module_& m) {
    register_tensor<T, 1, D>(m);
    register_tensor<T, 2, D>(m);
    register_tensor<T, 4, D>(m);   // only ever produced by outer(rank2, rank2)
    register_contract<T, D>(m);
    register_algebra<T, D>(m);

}

}  // namespace

// ===========================================================================
void register_tensor_algebra(nb::module_& m) {
    nb::module_ t = m.def_submodule(
        "tensor", "Index-notation tensor algebra (mtl/tensor)");

    register_for_dim<float, 2>(t);
    register_for_dim<float, 3>(t);
    register_for_dim<float, 4>(t);
    register_for_dim<double, 2>(t);
    register_for_dim<double, 3>(t);
    register_for_dim<double, 4>(t);

    t.def("zeros", [](std::size_t rank, std::size_t dim, const std::string& dtype)
                    -> nb::object {
        auto pick = [&]<typename T>() -> nb::object {
            if (rank == 1) { if (dim==2) return nb::cast(Tn<T,1,2>(T{0}));
                             if (dim==3) return nb::cast(Tn<T,1,3>(T{0}));
                             if (dim==4) return nb::cast(Tn<T,1,4>(T{0})); }
            if (rank == 2) { if (dim==2) return nb::cast(Tn<T,2,2>(T{0}));
                             if (dim==3) return nb::cast(Tn<T,2,3>(T{0}));
                             if (dim==4) return nb::cast(Tn<T,2,4>(T{0})); }
            if (rank == 4) { if (dim==2) return nb::cast(Tn<T,4,2>(T{0}));
                             if (dim==3) return nb::cast(Tn<T,4,3>(T{0}));
                             if (dim==4) return nb::cast(Tn<T,4,4>(T{0})); }
            throw nb::value_error(
                "zeros: rank must be 1, 2 or 4 and dim must be 2, 3 or 4");
        };
        if (dtype == "f64") return pick.template operator()<double>();
        if (dtype == "f32") return pick.template operator()<float>();
        throw nb::value_error("zeros: dtype must be 'f32' or 'f64'");
    }, "rank"_a, "dim"_a, "dtype"_a = "f64",
       "A zero tensor of the given rank and dimension");

    t.def("euclidean_metric", [](std::size_t dim, const std::string& dtype)
                               -> nb::object {
        auto pick = [&]<typename T>() -> nb::object {
            if (dim == 2) return nb::cast(mt::euclidean_metric<T, 2>());
            if (dim == 3) return nb::cast(mt::euclidean_metric<T, 3>());
            if (dim == 4) return nb::cast(mt::euclidean_metric<T, 4>());
            throw nb::value_error("euclidean_metric: dim must be 2, 3 or 4");
        };
        if (dtype == "f64") return pick.template operator()<double>();
        if (dtype == "f32") return pick.template operator()<float>();
        throw nb::value_error("euclidean_metric: dtype must be 'f32' or 'f64'");
    }, "dim"_a, "dtype"_a = "f64", "The identity metric in `dim` dimensions");

    t.def("minkowski_metric", []() { return mt::minkowski_metric<double>(); },
          "The 4-D Minkowski metric, signature (-, +, +, +)");

    t.def("dimensions", []() { return std::vector<std::size_t>{2, 3, 4}; },
          "The tensor dimensions instantiated");
    t.def("ranks", []() { return std::vector<std::size_t>{1, 2, 4}; },
          "The tensor ranks instantiated. Rank 3 is absent because nothing in "
          "mtl/tensor produces or consumes it; rank 4 exists only as the result "
          "of outer(rank2, rank2).");
}
