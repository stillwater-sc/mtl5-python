// mtl5-python -- Matrix Market I/O and spy visualization (MTL5 #125/#197/#252).
//
// Two things worth knowing before using these.
//
// 1. Unlike scipy.io.mmread, the *function* picks the container, not the file.
//    Both mm_read and mm_read_dense parse the banner and accept either
//    `coordinate` or `array` format; mm_read always gives you CSR and
//    mm_read_dense always gives you a dense matrix. So reading a dense .mtx
//    into CSR (or a sparse one into dense) is a deliberate call, not an error.
//
// 2. `.gz` paths are read transparently only when MTL5 is built with zlib,
//    which is off by default here. `mtl5.build_info()["zlib"]` reports it, and
//    without it a `.gz` path raises rather than silently mis-parsing.
//
// spy writes a PNG through MTL5's own from-scratch writer — no image library,
// no matplotlib. That matters for a headless run that wants a sparsity picture
// without pulling in a plotting stack.

#include "mtl5_types.hpp"

#include <nanobind/stl/string.h>

#include <mtl/io/matrix_market.hpp>
#include <mtl/io/spy.hpp>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>

using namespace nb::literals;

namespace {

/// MTL5 throws std::runtime_error for a missing/malformed file; nanobind maps
/// that to RuntimeError. Reading a path that does not exist is an OSError in
/// Python terms, so check first and raise the type a caller would except on.
void require_readable(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
        throw nb::value_error(("no such file: " + path).c_str());
    if (std::filesystem::is_directory(path, ec))
        throw nb::value_error((path + " is a directory").c_str());
}

void require_writable_parent(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (parent.empty()) return;
    std::error_code ec;
    if (!std::filesystem::exists(parent, ec))
        throw nb::value_error(
            ("no such directory: " + parent.string()).c_str());
}

template <typename T>
void register_io_for(nb::module_& io) {
    using MV   = MatrixView<T>;
    using SMat = mtl::mat::compressed2D<T>;

    io.def("mm_write", [](const std::string& path, const MV& A,
                          const std::string& comment) {
        require_writable_parent(path);
        nogil guard;
        mtl::io::mm_write(path, A.mat, comment);
    }, "path"_a, "A"_a, "comment"_a = "",
       "Write a dense matrix in Matrix Market `array` format");

    io.def("mm_write_sparse", [](const std::string& path, const SMat& A,
                                 const std::string& comment) {
        require_writable_parent(path);
        nogil guard;
        mtl::io::mm_write_sparse(path, A, comment);
    }, "path"_a, "A"_a, "comment"_a = "",
       "Write a sparse matrix in Matrix Market `coordinate` format");

    // -- spy: dense and sparse overloads for each element type ---------------
    auto opts = [](std::size_t max_pixels, bool log_scale) {
        if (max_pixels == 0)
            throw nb::value_error("max_pixels must be >= 1");
        mtl::io::spy_options o;
        o.max_pixels = max_pixels;
        o.log_scale = log_scale;
        return o;
    };

    io.def("spy", [opts](const MV& A, const std::string& path, std::size_t max_pixels) {
        require_writable_parent(path);
        auto o = opts(max_pixels, false);
        nogil guard;
        mtl::io::spy(A.mat, path, o);
    }, "A"_a, "path"_a, "max_pixels"_a = 1024,
       "Binary non-zero pattern as a grayscale PNG (MATLAB `spy`)");

    io.def("spy", [opts](const SMat& A, const std::string& path, std::size_t max_pixels) {
        require_writable_parent(path);
        auto o = opts(max_pixels, false);
        nogil guard;
        mtl::io::spy(A, path, o);
    }, "A"_a, "path"_a, "max_pixels"_a = 1024);

    io.def("spy_magnitude", [opts](const MV& A, const std::string& path,
                                   std::size_t max_pixels, bool log_scale) {
        require_writable_parent(path);
        auto o = opts(max_pixels, log_scale);
        nogil guard;
        mtl::io::spy_magnitude(A.mat, path, o);
    }, "A"_a, "path"_a, "max_pixels"_a = 1024, "log_scale"_a = false,
       "Colour each non-zero by |value| — shows where the mass is, not just "
       "where the pattern is");

    io.def("spy_magnitude", [opts](const SMat& A, const std::string& path,
                                   std::size_t max_pixels, bool log_scale) {
        require_writable_parent(path);
        auto o = opts(max_pixels, log_scale);
        nogil guard;
        mtl::io::spy_magnitude(A, path, o);
    }, "A"_a, "path"_a, "max_pixels"_a = 1024, "log_scale"_a = false);

    io.def("spy_density", [opts](const MV& A, const std::string& path,
                                 std::size_t max_pixels, bool log_scale) {
        require_writable_parent(path);
        auto o = opts(max_pixels, log_scale);
        nogil guard;
        mtl::io::spy_density(A.mat, path, o);
    }, "A"_a, "path"_a, "max_pixels"_a = 1024, "log_scale"_a = false,
       "Colour each pixel by how many non-zeros fell into it — the useful view "
       "once the matrix is larger than the image");

    io.def("spy_density", [opts](const SMat& A, const std::string& path,
                                 std::size_t max_pixels, bool log_scale) {
        require_writable_parent(path);
        auto o = opts(max_pixels, log_scale);
        nogil guard;
        mtl::io::spy_density(A, path, o);
    }, "A"_a, "path"_a, "max_pixels"_a = 1024, "log_scale"_a = false);
}

}  // namespace

// ===========================================================================
void register_io(nb::module_& m) {
    nb::module_ io = m.def_submodule(
        "io", "Matrix Market read/write and PNG sparsity visualization");

    // Readers are float64 only: Matrix Market carries no precision information
    // beyond real/integer, so a narrower element type would be a silent choice
    // on the caller's behalf. Use mtl5.convert() afterwards to say so out loud.
    io.def("mm_read", [](const std::string& path) {
        require_readable(path);
        nogil guard;
        return mtl::io::mm_read<double>(path);
    }, "path"_a,
       "Read a Matrix Market file into a sparse (CSR) matrix. Accepts both "
       "`coordinate` and `array` files — unlike scipy, the function chooses the "
       "container, not the file.");

    io.def("mm_read_dense", [](const std::string& path) {
        require_readable(path);
        mtl::mat::dense2D<double> A;
        {
            nogil guard;
            A = mtl::io::mm_read_dense<double>(path);
        }
        return MatrixView<double>(std::move(A));
    }, "path"_a,
       "Read a Matrix Market file into a dense matrix. Accepts both "
       "`coordinate` and `array` files.");

    register_io_for<float>(io);
    register_io_for<double>(io);
}
