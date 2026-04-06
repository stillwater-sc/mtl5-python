# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mtl5-python** is a nanobind-based Python module providing zero-copy interop between Python's scientific/ML ecosystem and MTL5 (Mathematics Template Library v5, a C++ library by Stillwater Supercomputing, Inc.).

Target ecosystem: NumPy, SciPy, JAX, PyTorch, pandas, scikit-learn.

Key capabilities:
- Zero-copy array interop via `nb::ndarray` (NumPy, PyTorch, JAX)
- Custom Universal number type dtypes (posit16, posit32) registered across NumPy, PyTorch, and pandas
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
```

## Architecture

- **nanobind** (not pybind11) for C++ → Python bindings — smaller binaries, faster compilation, first-class `nb::ndarray` multi-framework support
- **MTL5 C++ headers** (C++20, header-only) fetched via CMake FetchContent from stillwater-sc/mtl5
- **scikit-build-core** as PEP 517 backend for `pip install` support
- CMake option `MTL5_ENABLE_PYTHON=ON` (default ON) to build the module
- The nanobind extension module is `mtl5/_core.so`, re-exported via `mtl5/__init__.py`

### Key source files

- `python/src/mtl5_module.cpp` — nanobind `NB_MODULE` entry point, all C++ bindings
- `python/CMakeLists.txt` — nanobind module build target, links against MTL5 headers
- `CMakeLists.txt` — top-level CMake, fetches MTL5 and nanobind via FetchContent
- `mtl5/__init__.py` — Python package, re-exports from `_core`

### MTL5 C++ types used in bindings

| Python | C++ | Header |
|---|---|---|
| `mtl5.DenseVector` | `mtl::vec::dense_vector<double>` | `<mtl/vec/dense_vector.hpp>` |
| `mtl5.DenseMatrix` | `mtl::mat::dense2D<double>` | `<mtl/mat/dense2D.hpp>` |
| `mtl5.norm()` | `mtl::one_norm`, `mtl::two_norm`, `mtl::infinity_norm` | `<mtl/operation/norms.hpp>` |
| `mtl5.dot()` | `mtl::dot` | `<mtl/operation/dot.hpp>` |
| `mtl5.solve()` | `mtl::lu_factor` + `mtl::lu_solve` | `<mtl/operation/lu.hpp>` |

## Repository

- Remote: `git@github.sw:stillwater-sc/mtl5-python`
- Branch: `main`
