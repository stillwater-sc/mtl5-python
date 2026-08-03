"""Matrix Market I/O and spy PNG output.

The round trips are checked for exact equality, not closeness: Matrix Market is
a text format written at 17 significant digits, so a float64 value must come
back bit-identical. Anything less means the writer is losing precision.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import mtl5

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_header(path):
    """(width, height, channels) from a PNG's IHDR."""
    data = path.read_bytes()
    assert data[:8] == PNG_SIGNATURE, "not a PNG"
    width, height = struct.unpack(">II", data[16:24])
    colour_type = data[25]
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour_type]
    return width, height, channels


class TestMatrixMarketRoundTrip:
    def test_dense_round_trip_is_exact(self, tmp_path):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((7, 5))
        path = tmp_path / "dense.mtx"
        mtl5.io.mm_write(path, mtl5.matrix(A), "test")
        back = mtl5.io.mm_read_dense(path).to_numpy()
        np.testing.assert_array_equal(back, A)

    def test_sparse_round_trip_is_exact(self, tmp_path):
        pytest.importorskip("scipy.sparse")
        import mtl5.sparse as ms

        L = mtl5.generators.laplacian_2d(6, 6)
        path = tmp_path / "sparse.mtx"
        mtl5.io.mm_write_sparse(path, L, "laplacian")
        back = mtl5.io.mm_read(path)
        assert back.shape == L.shape
        assert back.nnz == L.nnz
        np.testing.assert_array_equal(ms.to_scipy(back).toarray(), ms.to_scipy(L).toarray())

    def test_banner_names_the_format(self, tmp_path):
        dense_path = tmp_path / "d.mtx"
        mtl5.io.mm_write(dense_path, np.eye(3), "")
        assert "array" in dense_path.read_text().splitlines()[0]

        sparse_path = tmp_path / "s.mtx"
        mtl5.io.mm_write_sparse(sparse_path, mtl5.generators.laplacian_1d(4), "")
        assert "coordinate" in sparse_path.read_text().splitlines()[0]

    def test_comment_is_written(self, tmp_path):
        path = tmp_path / "c.mtx"
        mtl5.io.mm_write(path, np.eye(2), "hello from the test")
        assert "hello from the test" in path.read_text()


class TestContainerNotFormat:
    """Unlike scipy.io.mmread, the *function* picks the container. Both readers
    accept both file formats, so this is a documented capability rather than an
    accident — pin it."""

    def test_array_file_can_be_read_into_csr(self, tmp_path):
        pytest.importorskip("scipy.sparse")
        import mtl5.sparse as ms

        A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        path = tmp_path / "a.mtx"
        mtl5.io.mm_write(path, A, "")
        csr = mtl5.io.mm_read(path)
        assert csr.shape == (2, 3)
        # Structural nonzeros only — the explicit zeros are dropped.
        assert csr.nnz == 3
        np.testing.assert_array_equal(ms.to_scipy(csr).toarray(), A)

    def test_coordinate_file_can_be_read_into_dense(self, tmp_path):
        pytest.importorskip("scipy.sparse")
        import mtl5.sparse as ms

        L = mtl5.generators.laplacian_1d(5)
        path = tmp_path / "c.mtx"
        mtl5.io.mm_write_sparse(path, L, "")
        dense = mtl5.io.mm_read_dense(path).to_numpy()
        np.testing.assert_array_equal(dense, ms.to_scipy(L).toarray())


class TestNumpyCoercion:
    def test_accepts_numpy_arrays(self, tmp_path):
        A = np.arange(6.0).reshape(2, 3)
        mtl5.io.mm_write(tmp_path / "n.mtx", A)
        mtl5.io.spy(A, tmp_path / "n.png")
        assert (tmp_path / "n.png").exists()

    def test_accepts_float32(self, tmp_path):
        A = np.eye(4, dtype=np.float32)
        mtl5.io.mm_write(tmp_path / "f.mtx", A)
        np.testing.assert_array_equal(mtl5.io.mm_read_dense(tmp_path / "f.mtx").to_numpy(), A)

    def test_rejects_other_dtypes(self, tmp_path):
        with pytest.raises(TypeError, match="float32 or float64"):
            mtl5.io.spy(np.eye(3, dtype=np.int64), tmp_path / "x.png")


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="no such file"):
            mtl5.io.mm_read(tmp_path / "absent.mtx")

    def test_directory_instead_of_file(self, tmp_path):
        with pytest.raises(ValueError, match="is a directory"):
            mtl5.io.mm_read_dense(tmp_path)

    def test_missing_output_directory(self, tmp_path):
        with pytest.raises(ValueError, match="no such directory"):
            mtl5.io.mm_write(tmp_path / "nope" / "x.mtx", np.eye(2))

    def test_malformed_banner(self, tmp_path):
        path = tmp_path / "bad.mtx"
        path.write_text("%%MatrixMarket matrix nonsense real general\n1 1 1\n1 1 1.0\n")
        with pytest.raises(RuntimeError, match="unrecognized"):
            mtl5.io.mm_read(path)

    def test_unsupported_field(self, tmp_path):
        """Complex Matrix Market files must be refused, not mis-parsed."""
        path = tmp_path / "cx.mtx"
        path.write_text("%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1.0 0.0\n")
        with pytest.raises(RuntimeError, match="unsupported.*field|field"):
            mtl5.io.mm_read(path)

    def test_zero_max_pixels(self, tmp_path):
        with pytest.raises(ValueError, match="max_pixels"):
            mtl5.io.spy(np.eye(4), tmp_path / "z.png", max_pixels=0)


class TestSpy:
    def test_writes_a_valid_grayscale_png(self, tmp_path):
        path = tmp_path / "p.png"
        mtl5.io.spy(mtl5.generators.laplacian_2d(8, 8), path)
        width, height, channels = png_header(path)
        assert (width, height) == (64, 64)
        assert channels == 1, "spy is a binary pattern, so grayscale"

    def test_magnitude_and_density_are_rgb(self, tmp_path):
        L = mtl5.generators.laplacian_2d(8, 8)
        for name, fn in (("m", mtl5.io.spy_magnitude), ("d", mtl5.io.spy_density)):
            path = tmp_path / f"{name}.png"
            fn(L, path)
            assert png_header(path)[2] == 3, f"{name} should be colour"

    def test_max_pixels_downsamples(self, tmp_path):
        L = mtl5.generators.laplacian_2d(20, 20)  # 400x400
        path = tmp_path / "small.png"
        mtl5.io.spy_density(L, path, max_pixels=64)
        width, height, _ = png_header(path)
        assert max(width, height) <= 64

    def test_dense_input(self, tmp_path):
        rng = np.random.default_rng(1)
        A = np.tril(rng.standard_normal((30, 30)))
        path = tmp_path / "dense.png"
        mtl5.io.spy_magnitude(A, path, log_scale=True)
        assert png_header(path)[:2] == (30, 30)

    def test_png_is_uncompressed(self, tmp_path):
        """MTL5's writer emits DEFLATE stored blocks so it needs no image
        library. Size is therefore ~w*h*channels — documented in mtl5/io.py
        because it surprises people on a large matrix."""
        path = tmp_path / "u.png"
        mtl5.io.spy(mtl5.generators.laplacian_2d(10, 10), path)
        width, height, channels = png_header(path)
        raw = height * (1 + width * channels)  # +1 filter byte per scanline
        assert path.stat().st_size == pytest.approx(raw, rel=0.02)


class TestBuildInfo:
    def test_zlib_flag_is_reported(self):
        """`.gz` support depends on it, so it has to be queryable."""
        assert isinstance(mtl5.build_info()["zlib"], bool)
