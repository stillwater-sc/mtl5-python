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

### Two registration paths

There are two kinds of scalar types, each with a different registration pattern:

#### Native C++ types (float, double, int32_t, int64_t)

These have corresponding NumPy dtypes. nanobind's `nb::ndarray<T, ...>` dispatches based on the incoming array's dtype automatically:

```cpp
register_native_with_solve<float>(m);   // f32
register_native_with_solve<double>(m);  // f64
register_native<int32_t>(m);           // i32
```

Users call `mtl5.vector(np_array)` and the right overload is selected by dtype.

#### Universal number types (fp8, fp16, posit, cfloat, etc.)

These have no NumPy dtype equivalent. They use **named factory functions** that accept float64 arrays and convert:

```cpp
// 1. type_suffix specialization
template <> constexpr const char* type_suffix<fp16>() { return "fp16"; }

// 2. One registration call — generates DenseVector_fp16, DenseMatrix_fp16,
//    vector_fp16(), matrix_fp16(), norm/dot/solve on typed objects
register_universal<fp16>(m, "vector_fp16", "matrix_fp16");
```

Users call `mtl5.vector_fp16(np_array)` to create fp16 vectors. Operations like `norm()`, `dot()`, and `solve()` accept the typed objects directly:

```python
v = mtl5.vector_fp16(np.array([3.0, 4.0]))
print(mtl5.norm(v))      # works — overloaded on DenseVector_fp16
print(mtl5.dot(v, v))    # works — overloaded on DenseVector_fp16

A = mtl5.matrix_fp16(np.eye(3))
b = mtl5.vector_fp16(np.ones(3))
x = mtl5.solve(A, b)     # returns DenseVector_fp16
```

The `to_numpy()` method converts back to float64 for interop with the NumPy ecosystem.

### Adding a new Universal type

Adding a new type from the Stillwater Universal library requires exactly two things:

```cpp
// 1. type_suffix specialization
template <> constexpr const char* type_suffix<posit<16,2>>() { return "posit16"; }

// 2. One registration call in NB_MODULE
register_universal<posit<16,2>>(m, "vector_posit16", "matrix_posit16");
```

This generates: `DenseVector_posit16`, `DenseMatrix_posit16`, `vector_posit16()`, `matrix_posit16()`, and overloaded `norm()`, `dot()`, `solve()` on the typed objects.

### Type roadmap

| Priority | Type | Suffix | C++ type | Status |
|---|---|---|---|---|
| Current | IEEE single | `f32` | `float` | Done — native dtype dispatch |
| Current | IEEE double | `f64` | `double` | Done — native dtype dispatch |
| Current | 32-bit int | `i32` | `int32_t` | Done — native dtype dispatch |
| Current | 64-bit int | `i64` | `int64_t` | Done — native dtype dispatch |
| Current | IEEE fp8 | `fp8` | `sw::universal::fp8` (`cfloat<8,2>`) | Done — named factory |
| Current | IEEE fp16 | `fp16` | `sw::universal::fp16` (`cfloat<16,5>`) | Done — named factory |
| Next | bfloat16 | `bf16` | `sw::universal::bfloat_t` (`cfloat<16,8>`) | Named factory |
| Next | fp8e4m3 | `fp8e4m3` | `sw::universal::fp8e4m3` (`cfloat<8,4>`) | Named factory (NVIDIA) |
| Next | fp8e5m2 | `fp8e5m2` | `sw::universal::fp8e5m2` (`cfloat<8,5>`) | Named factory (ARM/AMD) |
| Next | posit16 | `posit16` | `posit<16,2>` | Named factory |
| Next | posit32 | `posit32` | `posit<32,2>` | Named factory |
| Future | fp128 | `fp128` | `sw::universal::fp128` (`cfloat<128,15>`) | Named factory |

### Interaction with KPU dispatch

The type registration is orthogonal to device dispatch. A `DenseVector_posit16` can reside on CPU or KPU. The multi-precision design ensures that when KPU-native posit arithmetic is available, the binding layer already has the right type instantiations in place — the dispatch layer (issue #4) simply routes the operation to the KPU backend instead of the CPU reference implementation.

### Design constraints

**Why separate Python classes per type (not a single generic class)?**

nanobind (and CPython's C API) requires concrete C++ types for class bindings. A `dense_vector<float>` and `dense_vector<double>` are distinct C++ types with different memory layouts. Wrapping them in a type-erased `std::variant` would add runtime dispatch overhead to every element access and complicate the buffer protocol implementation.

The typed-class approach is also the pattern used by PyTorch (`torch.float32`, `torch.float16`), JAX (`jax.numpy.float32`), and ml_dtypes — users are familiar with it.

**Why overloaded functions instead of runtime dtype inspection?**

nanobind's overload resolution is implemented in C++ and runs at function-call entry, before any Python-level dispatch. This is faster than inspecting `array.dtype` in a Python `if/elif` chain or a C++ `switch` on dtype codes. It also produces better error messages when an unsupported dtype is passed.

**Why named factories for Universal types (`vector_fp16`) instead of overloading `vector()`?**

nanobind's `nb::ndarray<T, ...>` can only dispatch on types that have a NumPy dtype code (float32, float64, int32, etc.). Universal types like `cfloat<16,5>` have no NumPy dtype until custom dtype registration is implemented (issue #5). Named factories make the type explicit in the call and avoid ambiguity. Once custom dtypes are registered, `vector()` overloads can be added alongside the named factories.

### Dependencies

The Universal number types come from the [Stillwater Universal](https://github.com/stillwater-sc/universal) library, a separate header-only C++ library. It is fetched via CMake FetchContent alongside MTL5. The `config/TestMatrixDataDirConfig.hpp.in` stub is required because Universal's CMakeLists.txt references `CMAKE_SOURCE_DIR` for test matrix data paths, which resolves to the parent project when used via FetchContent.
