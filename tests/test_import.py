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
    import pytest

    if _core.__version__ == "0.0.0-dev":
        pytest.skip("dev build: extension built without scikit-build package metadata")

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


def test_all_matches_the_public_surface():
    """`__all__` must list every public name, and list nothing that is missing.

    This has drifted twice: `generators`/`arange`/`linspace`/`logspace`/
    `geomspace` shipped without being added, so `from mtl5 import *` and any
    tooling that reads `__all__` silently missed them. A hand-maintained list
    beside a growing module needs a check, not discipline.
    """
    exported = set(mtl5.__all__)

    public = {
        name
        for name in vars(mtl5)
        if not name.startswith("_")
        and name not in {"pandas_ext", "sparse"}  # submodules, imported for effect
    }

    missing = sorted(public - exported)
    assert not missing, f"public but absent from __all__: {missing}"

    dangling = sorted(n for n in exported if not hasattr(mtl5, n))
    assert not dangling, f"listed in __all__ but not defined: {dangling}"


def test_all_has_no_duplicates():
    assert len(mtl5.__all__) == len(set(mtl5.__all__)), "duplicate entries in __all__"
