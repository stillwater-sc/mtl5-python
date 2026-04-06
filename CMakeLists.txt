cmake_minimum_required(VERSION 3.22)
project(mtl5_python CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MTL5_ENABLE_PYTHON "Build Python bindings" ON)

# --- MTL5 (header-only) ---
find_package(MTL5 QUIET)
if(NOT MTL5_FOUND)
    include(FetchContent)
    # Suppress MTL5's own tests, examples, and install targets
    set(MTL5_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MTL5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(MTL5_BUILD_REGRESSION_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        mtl5
        GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(mtl5)
endif()

# --- Universal number library (header-only) ---
# Provides cfloat<nbits,es>, posit<nbits,es>, and other custom number types
find_package(universal QUIET)
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
        GIT_TAG        main
        GIT_SHALLOW    TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(universal)
endif()

if(MTL5_ENABLE_PYTHON)
    add_subdirectory(python)
endif()
