# Custom Dtype Registration Feasibility Analysis

## Status

Analysis document. Phase 1 (named factories + pandas ExtensionDtype) implemented in resolution of issue #5. Phase 2 (NumPy dtype) and Phase 3 (PyTorch tensor wrappers) deferred.

## The problem

Stillwater Universal provides custom number types (posit, fixpnt, lns, cfloat, dd, qd, …) that satisfy MTL5's `Scalar` concept and work seamlessly with `dense_vector<T>` / `dense2D<T>`. The Python bindings need to expose these to data scientists in a way that feels native — ideally as first-class dtypes in NumPy, PyTorch, and pandas.

The aspirational user experience:

```python
import numpy as np
import torch
import pandas as pd
import mtl5

# NumPy
a = np.array([1.0, 2.0, 3.0], dtype=mtl5.posit16)
b = a * 2.0   # element-wise math in posit arithmetic
np.sum(a)     # reductions

# PyTorch
t = torch.tensor([1.0, 2.0, 3.0], dtype=mtl5.torch_posit16)
loss = (t * w).sum()
loss.backward()  # autograd through posit operations

# pandas
s = pd.Series([1.0, 2.0, 3.0], dtype=mtl5.PandasPosit16Dtype())
s.groupby(...).mean()
```

This document analyzes how feasible each of these is in 2026.

## Framework-by-framework analysis

### NumPy custom dtypes

NumPy has had two custom dtype APIs:

**Legacy API (NumPy < 2.0)**: `PyArray_RegisterDataType`. The reference implementation is the [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes) project, which provides `bfloat16`, `float8_e4m3fn`, `float8_e5m2`, etc. Each dtype requires implementing dozens of C-level type slots:
- Cast functions (to/from every other dtype)
- Comparison functions (`<`, `==`, sort)
- UFunc loops for `+`, `-`, `*`, `/`, `**`, transcendentals
- Buffer protocol descriptors
- String formatting (`__repr__`, `__str__`)
- Pickling support
- NaN and infinity handling

The `ml_dtypes` codebase is **~5000 lines of C/C++ per dtype**, plus ~2000 lines of shared infrastructure. It is maintained by the JAX team primarily for ML use cases.

**New API (NumPy 2.0+)**: A redesigned dtype system that's more flexible but also more complex. NumPy 2.0 was released in 2024 and the new DType API is still considered experimental. Even with the new API, you cannot escape implementing all the type slots — you're just using a slightly cleaner C ABI.

**Pragmatic conclusion**: True NumPy dtype registration for posit/fixpnt/lns is feasible but expensive. To register all 4 posit configurations (8/16/32/64) plus 2 fixpnt and 2 lns variants would require ~40,000 lines of C++ across the ml_dtypes pattern, plus significant infrastructure to bridge with MTL5/Universal templates.

What works as a stopgap: **named factory functions** (`mtl5.vector_posit16(np_array)`) accept a NumPy float64 array, convert to posit internally, and return an `mtl5.DenseVector_posit16` object. This is what Phase 1 implements. The underlying storage is a contiguous C++ array of `posit<16,2>` objects, accessed via the MTL5 type. Conversion back to NumPy goes through float64.

### PyTorch custom dtypes

PyTorch's dtype system is **fundamentally closed**. The `torch.dtype` enum is a hardcoded list of types compiled into the PyTorch C++ binary:
- `torch.float16`, `torch.bfloat16`, `torch.float32`, `torch.float64`
- `torch.float8_e4m3fn`, `torch.float8_e5m2` (added in PyTorch 2.1)
- Integer types: `int8`, `uint8`, `int16`, `int32`, `int64`
- `torch.complex64`, `torch.complex128`
- `torch.bool`

There is no public API to register a new dtype. PyTorch's autograd, dispatcher, JIT, and CUDA kernels all assume the closed dtype set. The float8 variants were added by patching the core PyTorch source — they were not implemented as an extension.

**Available workarounds**:
1. **Storage pattern**: Store `posit<16,2>` values as `uint16` in a regular `torch.uint16` tensor. Provide conversion functions `mtl5.posit16_to_torch(arr)` and `mtl5.torch_to_posit16(tensor)`. Element-wise operations require unwrapping to MTL5, computing, and re-wrapping. Autograd does not work because PyTorch sees only uint16.
2. **Tensor subclass** (`torch.Tensor.__torch_function__`): A Python-level wrapper class that intercepts operations and dispatches to MTL5. This works for some operations but breaks integration with the rest of PyTorch (TorchScript, FX tracing, distributed, etc.).
3. **Wait for `__torch_dispatch__` maturation**: PyTorch is slowly building infrastructure for true extension types via the `__torch_dispatch__` protocol, but it's not yet stable enough for production use.

**Pragmatic conclusion**: True PyTorch dtype integration is **not currently possible** without forking PyTorch. The best we can offer is a tensor wrapper (Phase 3, future) that converts at boundaries. Autograd through posit arithmetic is out of reach unless PyTorch upstream adds extension dtype support.

### pandas ExtensionDtype

pandas provides a clean Python API for custom dtypes via `pandas.api.extensions.ExtensionDtype` and `pandas.api.extensions.ExtensionArray`. This is the **most feasible** integration of the three.

A pandas extension type requires:
1. **`ExtensionDtype` subclass**: Defines the type name, kind, and the backing array class.
2. **`ExtensionArray` subclass**: Implements the storage and operations:
   - `__getitem__` / `__setitem__`
   - `__len__`
   - `dtype` property
   - `nbytes`
   - `isna()`
   - `take()`, `copy()`, `_concat_same_type()`
   - Arithmetic via `_from_sequence()` and operator dispatch

The reference for this pattern is the `pandas.array(['a', 'b'], dtype='string')` implementation and the `pandas-stubs` extension types in the wider ecosystem. A pandas extension dtype for posit can use an MTL5 `DenseVector_posit16` as backing storage and delegate operations to MTL5.

**Pragmatic conclusion**: pandas extension dtype is doable in pure Python with the existing MTL5 bindings. ~200-300 lines per dtype, no C++ work required. Phase 1 includes this for posit16 as a reference implementation.

### JAX custom types

JAX uses NumPy dtypes under the hood, so any successful NumPy dtype registration would also work in JAX. JAX additionally supports:
- The `jax.dtypes` registry that bridges to NumPy
- Custom JVP/VJP rules for differentiation
- XLA lowering for hardware acceleration

If/when full NumPy dtype registration is implemented (Phase 2), JAX support comes mostly for free for non-traced code. Tracing through `jit` would require additional XLA primitives, which is a separate effort.

### scikit-learn

scikit-learn does not have a dtype concept of its own — it operates on NumPy arrays. Any data type that satisfies the NumPy array protocol works (with caveats around what operations the estimator uses internally). Once Phase 2 is done, sklearn estimators that use only basic arithmetic (LinearRegression, Ridge, simple distance metrics) should work with posit arrays. Estimators that call into LAPACK (most decompositions) will not, because LAPACK only knows IEEE float types.

## Phased delivery plan

### Phase 1 — Named factories + pandas ExtensionDtype (this PR)

**Scope**:
- Add posit8, posit16, posit32, posit64 via the existing `register_universal<T>` pattern
- Add fixpnt8, fixpnt16
- Add lns16, lns32
- pandas ExtensionDtype + ExtensionArray for posit16 as a reference implementation
- Tests for arithmetic, conversion, basic ops, pandas Series operations

**User experience**:
```python
v = mtl5.vector_posit16(np.array([1.0, 2.0, 3.0]))
mtl5.norm(v)
mtl5.dot(v, v)
mtl5.solve(A, b)  # all with posit16 arithmetic

s = pd.Series([1.0, 2.0, 3.0], dtype=mtl5.PandasPosit16Dtype())
s.sum()
```

**What you cannot do in Phase 1**:
- `np.array([1, 2, 3], dtype=mtl5.posit16)` — no NumPy dtype yet
- `torch.tensor([1, 2, 3], dtype=mtl5.posit16)` — never possible without PyTorch upstream
- `np.sum(posit_array)` — the array isn't a NumPy array, it's an MTL5 wrapper

### Phase 2 — NumPy DType registration (future epic)

**Scope**:
- Implement the ml_dtypes pattern for posit16 first (proof of concept)
- ~5000 lines of C++ per dtype
- Cast tables, ufunc loops, comparison, sort, formatting, pickling
- After posit16 is solid, factor out shared infrastructure for posit8/32/64 and fixpnt
- Integrate with the MTL5 binding layer so `mtl5.vector(posit_array)` works zero-copy

**Estimated effort**: 4-8 weeks per dtype family for first implementation, 1-2 weeks per additional configuration once infrastructure is in place.

**Should be its own epic** with sub-issues for each dtype family.

### Phase 3 — PyTorch tensor wrappers (future, low priority)

**Scope**:
- Tensor wrapper class storing posit bytes as `torch.uint16` / `torch.uint8`
- Conversion methods at boundaries (numpy ↔ wrapper ↔ MTL5)
- No autograd, no JIT, no distributed support
- Document that this is a workaround, not a true integration

**Estimated effort**: 1-2 weeks for the wrapper, ongoing maintenance burden because PyTorch internals shift across versions.

**Recommended priority**: low, until PyTorch's `__torch_dispatch__` matures or upstream adds extension dtype support.

## Key insight: KPU changes the calculus

The current analysis assumes posit arithmetic happens via software emulation on CPU. **When Stillwater KPU hardware ships, the calculus changes significantly**:

- KPU has native posit and fixpnt arithmetic units — operations execute directly in hardware, not through C++ emulation
- The MTL5 binding layer can dispatch posit operations to KPU through the existing backend hierarchy
- Performance and energy efficiency become competitive with IEEE float operations on CPU
- The "should I bother registering this with NumPy?" question becomes "yes, because it's actually fast"

This argues for investing in Phase 2 (NumPy dtype registration) once KPU hardware is generally available. Until then, Phase 1's named factories are sufficient for users to evaluate posit arithmetic on their workloads.

## Decision

**Approve Phase 1 for issue #5**. Open follow-up issues for Phase 2 (NumPy dtype) and Phase 3 (PyTorch wrapper) to be scheduled when KPU hardware availability justifies the investment.
