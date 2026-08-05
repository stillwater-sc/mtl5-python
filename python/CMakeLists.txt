# Find Python — nanobind requires the target name "Python" (not "Python3")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind: try find_package first, then FetchContent
find_package(nanobind QUIET)
if(NOT nanobind_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nanobind)
endif()

nanobind_add_module(_core
    src/mtl5_module.cpp
    src/mtl5_mixed_precision.cpp
    src/mtl5_sparse_direct.cpp
    src/mtl5_dense_factor.cpp
    src/mtl5_dense_ops.cpp
    src/mtl5_generators.cpp
    src/mtl5_io.cpp
    src/mtl5_complex.cpp
    src/mtl5_sparse_formats.cpp
    src/mtl5_ndarray.cpp
    src/mtl5_krylov.cpp
    src/mtl5_multigrid.cpp)

# The MTL5 + Universal composition headers we own (mtl/math/quire_accumulator.hpp
# bridges Universal's quire into MTL5's accumulator_traits). Kept under an
# <mtl/...> path so the specialization sits next to the generic trait it extends,
# matching the mp-* repos that mirror the same header.
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Inject the package version from pyproject.toml into the C++ module via a
# compile definition. scikit-build-core sets SKBUILD_PROJECT_VERSION; a plain
# `cmake -S . -B build` has no package metadata at all, so it falls back to the
# "0.0.0-dev" sentinel.
#
# PROJECT_VERSION here is mtl5_python's, which is deliberately unset. Do NOT use
# CMAKE_PROJECT_VERSION: FetchContent brings MTL5's own project(MTL5 VERSION
# x.y.z) into the build, and that value leaks through, stamping MTL5's version
# onto the Python package (a standalone build reported 5.7.0 for a 5.2.0
# package). Tests treat the sentinel as "dev build, version check not
# applicable".
if(DEFINED SKBUILD_PROJECT_VERSION AND NOT "${SKBUILD_PROJECT_VERSION}" STREQUAL "")
    set(_MTL5PY_VERSION "${SKBUILD_PROJECT_VERSION}")
elseif(DEFINED PROJECT_VERSION AND NOT "${PROJECT_VERSION}" STREQUAL "")
    set(_MTL5PY_VERSION "${PROJECT_VERSION}")
else()
    set(_MTL5PY_VERSION "0.0.0-dev")
endif()
target_compile_definitions(_core PRIVATE MTL5PY_VERSION="${_MTL5PY_VERSION}")

# MTL5 exports MTL5_NATIVE_FAST_GEMM through a $<BUILD_INTERFACE:...> guard, so
# it propagates when MTL5 is consumed via FetchContent but NOT when it is found
# as an installed package. The macro only gates header code — no library, no ISA
# assumption — so defining it directly here makes both paths behave the same.
# MTL5_HAS_HIGHWAY is deliberately not mirrored: it requires linking hwy.
if(MTL5_NATIVE_FAST_GEMM)
    target_compile_definitions(_core PRIVATE MTL5_NATIVE_FAST_GEMM)
endif()

# Use namespaced target — works for both find_package and FetchContent
# (FetchContent creates both 'mtl5' and 'MTL5::mtl5' alias)
target_link_libraries(_core PRIVATE MTL5::mtl5 universal)

# Install the extension module into the mtl5 package directory
install(TARGETS _core DESTINATION mtl5)
