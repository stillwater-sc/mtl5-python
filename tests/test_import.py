"""Basic smoke tests — verify the module loads and exposes expected symbols."""

import mtl5


def test_import():
    assert hasattr(mtl5, "__version__")


def test_version_string():
    assert isinstance(mtl5.__version__, str)
    assert mtl5.__version__ == "0.1.0"


def test_public_api():
    for name in ["vector", "matrix", "norm", "dot", "solve"]:
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
        assert hasattr(mtl5, f"vector_{suffix}"), f"mtl5.vector_{suffix} not found"
        assert hasattr(mtl5, f"matrix_{suffix}"), f"mtl5.matrix_{suffix} not found"


def test_default_aliases():
    assert mtl5.DenseVector is mtl5.DenseVector_f64
    assert mtl5.DenseMatrix is mtl5.DenseMatrix_f64
