cmake_minimum_required(VERSION 3.22)
project(mtl5_python CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: report correct __cplusplus value so that C++20 headers like <bit>
# are properly detected by Universal's bit_cast.hpp
if(MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

option(MTL5_ENABLE_PYTHON "Build Python bindings" ON)

# C++ numerical experiments under applications/. Deliberately OFF by default:
# scikit-build-core drives this file on every `pip install`, so an ON default
# would compile a demo binary into every user's install for no benefit to them.
# Turn it on explicitly:  cmake -DMTL5_BUILD_APPLICATIONS=ON
option(MTL5_BUILD_APPLICATIONS "Build the C++ applications under applications/" OFF)

# ---------------------------------------------------------------------------
# MTL5 acceleration options
#
# Every MTL5_WITH_*/MTL5_NATIVE_* option defaults to OFF upstream, which
# compiles out MTL5's entire native performance path — mtl::mult falls back to
# the generic scalar kernel and get_backend() can only ever report "reference".
# We declare the same option names here so the cache entry exists before MTL5's
# own option() runs, which lets us pick wheel-appropriate defaults while still
# honouring an explicit -D on the command line.
#
# MTL5_NATIVE_FAST_GEMM and MTL5_WITH_HIGHWAY are ON together, and that pairing
# is not optional. The blocked GEMM pays a packing cost that only pays off once
# the micro-kernel is actually vectorised, and mtl::simd falls back to a
# scalar batch (lane count 1) unless Highway is present. Measured here,
# double-precision matmul, GCC -O3, single thread, no -march=native:
#
#     config                       n=200      n=600     n=1000
#     both OFF (status quo)      3.51 GF/s  2.67 GF/s  2.15 GF/s
#     fast GEMM alone            0.79 GF/s  0.79 GF/s  0.79 GF/s   <- 3.4x SLOWER
#     fast GEMM + Highway       12.63 GF/s 14.40 GF/s 14.66 GF/s   <- 5.5-6.8x faster
#
# Highway is fetched at a pinned version, is Apache-2.0/BSD-3-Clause, and uses
# STATIC dispatch — one ISA chosen by the compiler flags. With no -march that
# is the x86-64 baseline, which is what the numbers above measure, so the
# result is still a portable binary.
#
#   MTL5_NATIVE_FAST_GEMM  ON  — blocked GEMM / SIMD GEMV. if-constexpr gated to
#                                dense contiguous float/double, so int/posit/
#                                fixpnt/lns still take the generic path.
#   MTL5_WITH_HIGHWAY      ON  — vectorises that kernel. See the table above for
#                                what happens if you turn this off on its own.
#   MTL5_WITH_BLAS         OFF — needs a BLAS on the build machine.
#   MTL5_WITH_LAPACK       OFF — needs LAPACK; enables geev/syev/gesdd dispatch,
#                                which the Phase 3 eigen/SVD bindings will want.
#   MTL5_NATIVE_ARCH       OFF — -march=native lets Highway target AVX2/AVX-512
#                                instead of the baseline, but the binary then
#                                only runs on machines like the build host.
#                                Worth enabling for a local build, never for a
#                                distributed wheel.
#
# Override at install time, e.g.:
#   pip install . -C cmake.define.MTL5_WITH_BLAS=ON \
#                 -C cmake.define.MTL5_WITH_LAPACK=ON
# ---------------------------------------------------------------------------
option(MTL5_NATIVE_FAST_GEMM "Route mtl::mult through MTL5's blocked GEMM / SIMD GEMV" ON)
option(MTL5_WITH_HIGHWAY     "Use Google Highway as the mtl::simd backend"             ON)
option(MTL5_WITH_BLAS        "Link an external BLAS for dense acceleration"            OFF)
option(MTL5_WITH_LAPACK      "Link LAPACK (enables geev/syev/gesdd dispatch)"          OFF)
option(MTL5_NATIVE_ARCH      "Tune for the host CPU (-march=native; non-portable)"     OFF)

if(MTL5_NATIVE_FAST_GEMM AND NOT MTL5_WITH_HIGHWAY)
    message(WARNING
        "MTL5_NATIVE_FAST_GEMM=ON with MTL5_WITH_HIGHWAY=OFF is slower than "
        "turning both off: the blocked GEMM pays its packing cost but mtl::simd "
        "falls back to a scalar batch, so the micro-kernel never vectorises "
        "(measured ~3.4x slower than the generic kernel). Enable "
        "MTL5_WITH_HIGHWAY, or set MTL5_NATIVE_FAST_GEMM=OFF as well.")
endif()

# --- MTL5 (header-only) ---
#
# Pinned to a release tag, not a branch. FetchContent runs at `pip install`
# time, so a floating `main` would make every user's build -- and every wheel
# CI ships -- a different library. The pin is what makes a released wheel
# reproducible, and it is the thing to bump when tracking MTL5 upstream:
# mtl5-python's minor version follows the MTL5 tag named here.
#
# v5.10.0 is a floor, not a preference. Three things the bindings now need,
# each from a different release, so the floor is the newest of them:
#   v5.9.0  -- mtl5_mixed_precision.cpp needs the two_norm<Accumulator, Result> /
#              frobenius_norm<Accumulator, Result> forms (v5.8.0 has no Result
#              parameter), and mtl5_complex.cpp needs the Hermitian
#              factorizations cholesky_h_factor / cholesky_h_solve /
#              ldlt_h_factor / ldlt_h_solve and CHOLESKY_NOT_HERMITIAN.
#   v5.9.1  -- mtl5_sparse_direct.cpp calls sparse_lu_refactor_in_place and
#              supernodal_lu_refactor_in_place (MTL5 #310). The value-returning
#              sparse_lu_refactor / supernodal_lu_refactor still exist and still
#              compile, so dropping below this floor would silently reintroduce
#              the copy the in-place forms were written to remove -- a 23%
#              regression on the refactor path, with no diagnostic.
#   v5.10.0 -- mtl5_module.cpp calls mtl::util::build_isa_list() (MTL5 #443),
#              which does not exist before this tag.
#
# The version argument to find_package is what applies that floor to the OTHER
# build path. A bare find_package(MTL5 QUIET) accepts *any* system-installed
# MTL5 and skips the FetchContent block entirely, so a developer with 5.7.x
# installed would silently build against it and get ~140 template errors deep
# inside norms.hpp instead of a version message. MTL5 ships its ConfigVersion
# with COMPATIBILITY SameMajorVersion, under which 5.10.0 rejects an installed
# 5.9.x and accepts 5.10.0 or 5.11.0 -- exactly the floor we want. (It also
# rejects a 6.x install, which is correct: an MTL5 major bump is the manual
# intervention case in the version policy, not something to absorb silently.)
find_package(MTL5 5.10.0 QUIET)
if(NOT MTL5_FOUND)
    include(FetchContent)
    # Suppress MTL5's own tests, examples, and install targets
    set(MTL5_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MTL5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MTL5_BUILD_REGRESSION_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        mtl5
        GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
        GIT_TAG        v5.10.0
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(mtl5)
endif()

# --- Universal number library (header-only) ---
# Provides cfloat<nbits,es>, posit<nbits,es>, and other custom number types.
# Pinned for the same reason as MTL5 above: this is fetched at `pip install`
# time, and the quire sizing it supplies backs the mixed-precision norms.
# Version-floored for the same reason too -- see the find_package note above.
find_package(universal 4.7.9 QUIET)
if(NOT universal_FOUND)
    include(FetchContent)
    set(BUILD_ALL_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_DEMONSTRATION OFF CACHE BOOL "" FORCE)
    set(BUILD_NUMBERS OFF CACHE BOOL "" FORCE)
    set(BUILD_NUMERICS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_REGRESSION_SUITES OFF CACHE BOOL "" FORCE)
    set(BUILD_CMD_LINE_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_EDUCATION OFF CACHE BOOL "" FORCE)
    set(BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
    set(BUILD_PLAYGROUND OFF CACHE BOOL "" FORCE)
    set(BUILD_VALIDATION_HW OFF CACHE BOOL "" FORCE)
    set(BUILD_C_API_LIB OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        universal
        GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
        GIT_TAG        v4.7.9
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(universal)
endif()

if(MTL5_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

if(MTL5_BUILD_APPLICATIONS)
    add_subdirectory(applications)
endif()
