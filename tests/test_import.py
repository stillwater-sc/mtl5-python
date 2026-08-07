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


def test_stale_core_guard_is_actionable():
    """A compiled _core older than the Python layer must fail with a clear
    rebuild message, not the cryptic 'cannot import name X from mtl5._core'.

    Reproduces the editable-install hazard (updated the source, forgot to
    rebuild the extension) in a subprocess: inject a fake mtl5._core that is
    missing the newer `tensor` submodule, then import the source package.
    """
    import pathlib
    import subprocess
    import sys
    import textwrap

    repo = pathlib.Path(__file__).resolve().parents[1]
    script = textwrap.dedent(f"""
        import sys, types
        core = types.ModuleType("mtl5._core")
        for _n in ("view", "mg", "mixed", "array", "backends", "build_info"):
            setattr(core, _n, object())          # everything EXCEPT `tensor`
        core.__version__ = "0.0.0"
        core.__file__ = "fake/_core.pyd"
        sys.modules["mtl5._core"] = core
        sys.path.insert(0, {str(repo)!r})        # load the source mtl5 package
        try:
            import mtl5
        except ImportError as e:
            msg = str(e)
            assert "missing: tensor" in msg, msg
            assert "pip install" in msg, msg     # points at the fix
            print("GUARD_OK")
        else:
            raise SystemExit("stale-core guard did not fire")
    """)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout!r}\nstderr={r.stderr!r}"
    assert "GUARD_OK" in r.stdout


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
