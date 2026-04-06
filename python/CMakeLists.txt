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

nanobind_add_module(_core src/mtl5_module.cpp)

# Use namespaced target — works for both find_package and FetchContent
# (FetchContent creates both 'mtl5' and 'MTL5::mtl5' alias)
target_link_libraries(_core PRIVATE MTL5::mtl5)

# Install the extension module into the mtl5 package directory
install(TARGETS _core DESTINATION mtl5)
