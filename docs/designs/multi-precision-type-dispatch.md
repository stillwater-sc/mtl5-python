# Multi-Precision Type Dispatch

## Status

Implemented (v0.1.0) for IEEE types. Universal number types pending custom dtype registration (#5).

## Need

MTL5's value proposition is mixed-precision computation — the same algorithm running at different precisions depending on accuracy and energy requirements. The C++ library is fully templated: `dense_vector<T>`, `dense2D<T>`, and all operations work with any scalar type `T`, including:

**Standard IEEE types:** `float` (fp32), `double` (fp64)

**MTL5 Universal number types:** `posit<16,2>` (posit16), `posit<32,2>` (posit32), `cfloat<8,2>` (fp8), and many more configurable formats spanning fp4 through fp128+.

The Python bindings must expose this full type spectrum without:
- Hardcoding to a single precision (the initial `double`-only implementation)
- Requiring massive code duplication per type
- Sacrificing Python's ergonomic "just pass an array" interface

This is especially critical for Stillwater's KPU hardware accelerators, where posit arithmetic is natively supported and the ability to run a solver at posit16 vs float64 directly impacts performance and energy efficiency.

## What

### User-facing API

The Python API auto-dispatches based on the input array's dtype. Users never need to specify types manually:

```python
import numpy as np
import mtl5

# float64 — dispatches to dense_vector<double>
x64 = mtl5.vector(np.array([1.0, 2.0, 3.0], dtype=np.float64))
# → mtl5.DenseVector_f64

# float32 — dispatches to dense_vector<float>
x32 = mtl5.vector(np.array([1.0, 2.0, 3.0], dtype=np.float32))
# → mtl5.DenseVector_f32

# int32 — dispatches to dense_vector<int32_t>
xi = mtl5.vector(np.array([1, 2, 3], dtype=np.int32))
# → mtl5.DenseVector_i32

# Operations dispatch on input dtype
mtl5.norm(np.array([3.0, 4.0], dtype=np.float32))  # → float32 computation
mtl5.solve(A_f32, b_f32)                            # → float32 LU solve

# Future: Universal types (once custom dtypes are registered)
# x = mtl5.vector(np.array([1.0, 2.0], dtype=mtl5.posit16))
# → mtl5.DenseVector_posit16
```

### Type matrix

Each registered scalar type gets a full set of bindings:

| Component | Generated per type | Example (float) |
|---|---|---|
| Vector class | `DenseVector_{suffix}` | `DenseVector_f32` |
| Matrix class | `DenseMatrix_{suffix}` | `DenseMatrix_f32` |
| `vector()` overload | dtype-dispatched factory | `vector(np.float32 array)` → `DenseVector_f32` |
| `matrix()` overload | dtype-dispatched factory | `matrix(np.float32 array)` → `DenseMatrix_f32` |
| `norm()` overload | dtype-dispatched operation | `norm(np.float32 array)` |
| `dot()` overload | dtype-dispatched operation | `dot(np.float32, np.float32)` |
| `solve()` overload | floating-point types only | `solve(np.float32, np.float32)` |

Convenience aliases `DenseVector` and `DenseMatrix` default to the `_f64` variants.

### Currently registered types

| Suffix | C++ type | vector/matrix | norm/dot | solve |
|---|---|---|---|---|
| `f32` | `float` | yes | yes | yes |
| `f64` | `double` | yes | yes | yes |
| `i32` | `int32_t` | yes | yes | no |
| `i64` | `int64_t` | yes | yes | no |

## How

### Template-based registration

The binding code uses C++ function templates to stamp out identical binding structures for each scalar type. Three compile-time traits map a C++ type to its Python-side metadata:

```cpp
// 1. NumPy dtype code — for nanobind's ndarray type matching
template <typename T> struct numpy_dtype;
template <> struct numpy_dtype<float>  { static constexpr auto value = nb::dtype<float>(); };

// 2. Human-readable suffix — for Python class names
template <typename T> constexpr const char* type_suffix();
template <> constexpr const char* type_suffix<float>() { return "f32"; }

// 3. Registration templates — stamp out all bindings for one type
template <typename T> void register_dense_vector(nb::module_& m);
template <typename T> void register_dense_matrix(nb::module_& m);
template <typename T> void register_vector_overload(nb::module_& m);
// ... etc.
```

### Convenience wrappers

Two top-level functions combine all registrations:

```cpp
// register_all<T>() — vector, matrix, norm, dot
// register_all_with_solve<T>() — same + solve (floating-point only)
```

### nanobind overload dispatch

nanobind natively supports function overloading based on `nb::ndarray` dtype constraints. When multiple overloads of `vector()` are registered with different dtype templates:

```cpp
// These are separate overloads; nanobind picks the right one at call time
m.def("vector", [](nb::ndarray<float,  nb::ndim<1>, ...> a) { ... });
m.def("vector", [](nb::ndarray<double, nb::ndim<1>, ...> a) { ... });
m.def("vector", [](nb::ndarray<int32_t, nb::ndim<1>, ...> a) { ... });
```

nanobind inspects the incoming NumPy array's dtype and dispatches to the matching overload. If no overload matches, it raises a `TypeError` with a clear message listing the supported types.

### Adding a new type

Adding a new scalar type requires exactly three things:

```cpp
// 1. type_suffix specialization
template <> constexpr const char* type_suffix<posit<16,2>>() { return "posit16"; }

// 2. numpy_dtype specialization (once the custom dtype is registered with NumPy)
template <> struct numpy_dtype<posit<16,2>> {
    static constexpr auto value = nb::dtype<posit<16,2>>();
};

// 3. One registration call in NB_MODULE
register_all_with_solve<posit<16,2>>(m);   // posit16
```

This generates: `DenseVector_posit16`, `DenseMatrix_posit16`, and typed overloads for `vector()`, `matrix()`, `norm()`, `dot()`, `solve()`.

### Planned type roadmap

| Priority | Type | Suffix | C++ type | Notes |
|---|---|---|---|---|
| Current | IEEE single | `f32` | `float` | Done |
| Current | IEEE double | `f64` | `double` | Done |
| Current | 32-bit int | `i32` | `int32_t` | Done |
| Current | 64-bit int | `i64` | `int64_t` | Done |
| Next | posit16 | `posit16` | `posit<16,2>` | Requires custom dtype (#5) |
| Next | posit32 | `posit32` | `posit<32,2>` | Requires custom dtype (#5) |
| Future | fp8 | `fp8` | `cfloat<8,2>` | ML inference workloads |
| Future | fp16 | `fp16` | `cfloat<16,5>` or `half` | Standard half-precision |
| Future | fp128 | `fp128` | `cfloat<128,15>` | Extended precision |

### Interaction with KPU dispatch

The type registration is orthogonal to device dispatch. A `DenseVector_posit16` can reside on CPU or KPU. The multi-precision design ensures that when KPU-native posit arithmetic is available, the binding layer already has the right type instantiations in place — the dispatch layer (issue #4) simply routes the operation to the KPU backend instead of the CPU reference implementation.

### Design constraints

**Why separate Python classes per type (not a single generic class)?**

nanobind (and CPython's C API) requires concrete C++ types for class bindings. A `dense_vector<float>` and `dense_vector<double>` are distinct C++ types with different memory layouts. Wrapping them in a type-erased `std::variant` would add runtime dispatch overhead to every element access and complicate the buffer protocol implementation.

The typed-class approach is also the pattern used by PyTorch (`torch.float32`, `torch.float16`), JAX (`jax.numpy.float32`), and ml_dtypes — users are familiar with it.

**Why overloaded functions instead of runtime dtype inspection?**

nanobind's overload resolution is implemented in C++ and runs at function-call entry, before any Python-level dispatch. This is faster than inspecting `array.dtype` in a Python `if/elif` chain or a C++ `switch` on dtype codes. It also produces better error messages when an unsupported dtype is passed.
