"""Basic smoke tests — verify the module loads and exposes expected symbols."""

import mtl5


def test_import():
    assert hasattr(mtl5, "__version__")


def test_version_string():
    assert isinstance(mtl5.__version__, str)
    assert mtl5.__version__ == "0.1.0"


def test_public_api():
    for name in ["vector", "matrix", "norm", "dot", "solve", "DenseVector", "DenseMatrix"]:
        assert hasattr(mtl5, name), f"mtl5.{name} not found"
