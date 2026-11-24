# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mtl5-python** is a nanobind-based Python module providing zero-copy interop between Python's scientific/ML ecosystem and MTL5 (the Matrix Template Library, version 5, a C++ library by Stillwater Supercomputing, Inc.).

Target ecosystem: NumPy, SciPy, JAX, PyTorch, pandas, scikit-learn.

Key capabilities:
- Zero-copy array interop via `nb::ndarray` (NumPy, PyTorch, JAX)
- 25 element types: native IEEE/complex/integer plus Universal number systems (posit, fixpnt, lns, cfloat, takum, float cascades). See "Element types" below
- Dense/sparse linear algebra solvers exposed to Python with hardware accelerator dispatch
- SciPy sparse matrix interop and LinearOperator protocol

## Build and Test Commands

```bash
# Install (builds C++ extension via scikit-build-core + CMake)
pip install .

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_vector.py

# Run a single test
pytest tests/test_vector.py::TestNorm::test_l2_norm -v

# Lint
pip install ruff
ruff check .
ruff format --check .

# Documentation site (Sphinx + MyST-NB; executes the notebooks)
pip install -e ".[docs]"
sphinx-build -E -W --keep-going -b html docs docs/_build/html
```

## Architecture

- **nanobind** (not pybind11) for C++ → Python bindings — smaller binaries, faster compilation, first-class `nb::ndarray` multi-framework support
- **MTL5 C++ headers** (C++20, header-only) fetched via CMake FetchContent from stillwater-sc/mtl5
- **scikit-build-core** as PEP 517 backend for `pip install` support
- CMake option `MTL5_ENABLE_PYTHON=ON` (default ON) to build the module
- The nanobind extension module is `mtl5/_core.so`, re-exported via `mtl5/__init__.py`

### Key source files

- `python/src/mtl5_module.cpp` — nanobind `NB_MODULE` entry point; the bindings are split across the other `python/src/mtl5_*.cpp` files (dense ops/factorizations, sparse, Krylov, multigrid, mixed precision, tensor, I/O, ...)
- `python/src/mtl5_types.hpp` — element-type aliases and `type_suffix<T>()`, the authoritative list of bound element types
- `python/CMakeLists.txt` — nanobind module build target, links against MTL5 headers
- `CMakeLists.txt` — top-level CMake, fetches MTL5 and nanobind via FetchContent
- `mtl5/__init__.py` — Python package, re-exports from `_core`
- `mtl5/_universal_interop.py` — bit-exact copy to/from `universal_dtypes` NumPy arrays
- `mtl5/pandas_ext.py` — pandas extension dtype
- `docs/` — Sphinx source for the documentation site (`conf.py`, `index.md`, `getting-started.md`, `guide/`, `api/`), deployed to GitHub Pages by `.github/workflows/docs.yml`. `docs/designs/`, `docs/sessions/` and the other loose notes there are internal and excluded from the site. The site's notebooks are copied from `notebooks/` at build time.
- `applications/` — Python example scripts, plus C++ experiments built only with `-DMTL5_BUILD_APPLICATIONS=ON` (default OFF, so `pip install` never compiles them)

### Element types

Every element type has its own `DenseVector_<suffix>` / `DenseMatrix_<suffix>` class and `vector_<suffix>` / `matrix_<suffix>` factory. The source of truth is `type_suffix<T>()` in `python/src/mtl5_types.hpp`.

| Suffix | C++ type | Notes |
|---|---|---|
| `f32`, `f64` | `float`, `double` | zero-copy |
| `c64`, `c128` | `std::complex<float>`, `std::complex<double>` | zero-copy; NumPy naming (total width) |
| `i8`, `i16`, `i32`, `i64`, `u8`, `u32` | fixed-width integers | zero-copy; narrow ints are for widening dot / integer GEMM, `u32` is the `u8×u8` quad-MAC result |
| `fp8` | `sw::universal::fp8` | `cfloat<8,2>` (e2m5), **not** OCP e5m2 |
| `fp16` | `sw::universal::fp16` | IEEE half via cfloat |
| `cfloat32` | `sw::universal::fp32` | emulated binary32, used as the control against hardware `f32` |
| `posit8/16/32/64` | `posit<N,2>` | es=2 for every width |
| `fixpnt8`, `fixpnt16` | `fixpnt<8,4,Saturate>`, `fixpnt<16,8,Saturate>` | saturating |
| `lns16`, `lns32` | `lns<16,8>`, `lns<32,16>` | logarithmic number system |
| `takum32` | `takum<32,3,uint32_t>` | tapered logarithmic |
| `dd_cascade`, `td_cascade`, `qd_cascade` | Universal float cascades | about 32/48/64 decimal digits |

Coverage differs by entry point:
- `mtl5.dtypes()` / `convert()` cover the 17 non-complex, non-integer types. Integer types are deliberately excluded, because converting reals to i8 is a quantization scheme, not a cast.
- `universal_dtypes` interop (`_universal_interop.SUPPORTED`) covers 13 types: posit*, fixpnt*, lns*, takum32, fp16, and the cascades. `fp8` and `cfloat32` are not covered.
- pandas registers only `posit16` as an extension dtype.
- No custom dtypes are registered with PyTorch or JAX. Those frameworks get zero-copy only for the native types.

### MTL5 C++ types used in bindings

| Python | C++ | Header |
|---|---|---|
| `mtl5.DenseVector` (= `DenseVector_f64`) | `mtl::vec::dense_vector<double>` | `<mtl/vec/dense_vector.hpp>` |
| `mtl5.DenseMatrix` (= `DenseMatrix_f64`) | `mtl::mat::dense2D<double>` | `<mtl/mat/dense2D.hpp>` |
| `mtl5.norm()` | `mtl::one_norm`, `mtl::two_norm`, `mtl::infinity_norm` | `<mtl/operation/norms.hpp>` |
| `mtl5.dot()` | `mtl::dot` | `<mtl/operation/dot.hpp>` |
| `mtl5.solve()` | `mtl::lu_factor` + `mtl::lu_solve` | `<mtl/operation/lu.hpp>` |

## Repository

- Remote: `git@github.sw:stillwater-sc/mtl5-python`
- Branch: `main`
