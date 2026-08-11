# C++ numerical experiments.
#
# These are demonstrations, not part of the installed Python package: nothing
# here is installed, and the whole directory is gated behind
# MTL5_BUILD_APPLICATIONS (OFF by default) so that `pip install` never compiles
# them. See the option in the top-level CMakeLists.txt.
#
# The Python applications alongside these (precision_showcase.py and friends)
# need no build step -- they import the compiled mtl5 module.

# Every C++ application links the same two header-only libraries the Python
# extension does. MTL5::mtl5 is the namespaced target that works for both the
# find_package and FetchContent paths; `universal` carries the include
# directory for <universal/number/...>.
function(mtl5_add_application name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE MTL5::mtl5 universal)
    # Not part of `all`: MTL5_BUILD_APPLICATIONS=ON should make these buildable,
    # and CI builds them explicitly by target, but a developer who flips the
    # option on while iterating on the bindings should not pay for them on every
    # incremental build.
    set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL FALSE)
endfunction()

# Mixed-precision UKF stability comparison: Cholesky vs LDL^T vs Bunch-Kaufman
# across number systems. See issue #18.
mtl5_add_application(ukf_mixed_precision)
