"""Basic smoke tests — verify the module loads and exposes expected symbols."""

import mtl5


def test_import():
    assert hasattr(mtl5, "__version__")


def test_version_string():
    assert isinstance(mtl5.__version__, str)
    # Version must be a valid semver (major.minor.patch)
    parts = mtl5.__version__.split(".")
    assert len(parts) >= 3, f"Expected semver, got '{mtl5.__version__}'"
    assert all(p.isdigit() for p in parts[:3])


def test_version_sync():
    """Python package version must match the C++ extension module version."""
    import mtl5._core as _core

    assert mtl5.__version__ == _core.__version__, (
        f"Version mismatch: mtl5.__version__={mtl5.__version__!r} "
        f"vs _core.__version__={_core.__version__!r}"
    )


def test_public_api():
    for name in ["vector", "vector_copy", "matrix", "matrix_copy", "norm", "dot", "solve"]:
        assert hasattr(mtl5, name), f"mtl5.{name} not found"


def test_typed_vector_classes():
    for suffix in ["f32", "f64", "fp8", "fp16", "i32", "i64"]:
        name = f"DenseVector_{suffix}"
        assert hasattr(mtl5, name), f"mtl5.{name} not found"


def test_typed_matrix_classes():
    for suffix in ["f32", "f64", "fp8", "fp16", "i32", "i64"]:
        name = f"DenseMatrix_{suffix}"
        assert hasattr(mtl5, name), f"mtl5.{name} not found"


def test_universal_factories():
    for suffix in ["fp8", "fp16"]:
        assert hasattr(mtl5, f"vector_{suffix}")
        assert hasattr(mtl5, f"matrix_{suffix}")


def test_default_aliases():
    assert mtl5.DenseVector is mtl5.DenseVector_f64
    assert mtl5.DenseMatrix is mtl5.DenseMatrix_f64


def test_device_api():
    assert hasattr(mtl5, "devices")
    devs = mtl5.devices()
    assert "cpu" in devs
