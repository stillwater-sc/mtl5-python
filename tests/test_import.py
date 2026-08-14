"""Basic smoke tests — verify the module loads and exposes expected symbols."""

import subprocess
import sys
import tempfile
import textwrap

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

    Covers the SUBMODULE half of the hazard with a synthetic _core. The class
    half — the one that actually reached a user — is covered against the real
    extension in test_a_stale_core_produces_the_actionable_message below.
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
            assert "out of sync" in msg, msg
            assert "tensor" in msg, msg          # names what is missing
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


# ---------------------------------------------------------------------------
# The stale-extension guard
#
# mtl5/__init__.py reframes an ImportError from _core into an actionable
# "rebuild your extension" message. It is worth testing because it has already
# failed once in the field: the guard used to probe a fixed tuple of submodules,
# so when #69-#73 added CLASSES to _core it passed happily and a user with an
# editable install got the bare "cannot import name 'DenseMatrix_cfloat32'".
#
# Both directions matter — reframing too little leaves the original confusion,
# reframing too much mislabels an unrelated failure as a stale build. Each runs
# in a subprocess because it has to manipulate sys.modules before mtl5 is
# imported, which cannot be undone within a live interpreter.
# ---------------------------------------------------------------------------


def _run(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter, from OUTSIDE the repository.

    cwd matters and is not incidental. `python -c` prepends the working
    directory to sys.path, and the subprocess inherits pytest's cwd — the repo
    root in CI. From there the source `mtl5/` package shadows the installed one
    and has no compiled `_core`, so the snippet dies before reaching what it
    meant to test. That produced an empty stdout and an assertion complaining
    about a missing substring, which said nothing about the cause.

    Running from a temp directory makes the subprocess resolve `mtl5` the same
    way an ordinary user would.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tempfile.gettempdir(),
    )


def test_a_stale_core_produces_the_actionable_message():
    """Load the real extension, hide one class a newer .py layer imports, and
    confirm the guard explains the rebuild instead of leaking the raw error.

    The extension is loaded through ExtensionFileLoader rather than
    importlib.util.find_spec: find_spec imports the PARENT package first, which
    would run mtl5/__init__.py to completion before anything could be hidden.
    """
    result = _run("""
        import glob, importlib.machinery, importlib.util, os, sys

        # Locate the extension WITHOUT importing it. find_spec on a top-level
        # package does not execute its __init__.py, so _core is never
        # initialised here — which matters: `import mtl5._core` followed by
        # exec_module on the same file would initialise a single-phase
        # extension twice in one process and abort the interpreter. (It did,
        # silently, with empty stdout.)
        pkg = importlib.util.find_spec("mtl5")
        pkgdir = list(pkg.submodule_search_locations)[0]
        cands = [
            p
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            for p in glob.glob(os.path.join(pkgdir, "_core" + suffix))
        ]
        assert cands, f"no _core extension found in {pkgdir}"

        loader = importlib.machinery.ExtensionFileLoader("mtl5._core", cands[0])
        spec = importlib.util.spec_from_loader("mtl5._core", loader)
        stale = importlib.util.module_from_spec(spec)
        loader.exec_module(stale)
        delattr(stale, "DenseMatrix_cfloat32")   # pretend it predates #70
        sys.modules["mtl5._core"] = stale

        try:
            import mtl5
        except ImportError as exc:
            print(exc)
        else:
            print("NO ERROR")
    """)
    out = result.stdout
    # Surface stderr on failure: the first version of this test died in the
    # subprocess and asserted against an empty string, which said nothing about
    # why.
    ctx = f"\\nstdout={out!r}\\nstderr={result.stderr!r}"
    assert "NO ERROR" not in out, "the guard did not fire on a stale extension" + ctx
    assert "out of sync" in out, ctx
    assert "DenseMatrix_cfloat32" in out, "the missing symbol must be named" + ctx
    assert "pip install -e ." in out, "the message must say how to fix it" + ctx


def test_an_unrelated_import_error_is_not_mislabelled():
    """A failure that is not about _core must surface as itself.

    Reframing everything would be worse than reframing nothing: it would send
    someone with a missing dependency off rebuilding a C++ extension.
    """
    result = _run("""
        import sys

        # Fail the mtl5.tensor import itself, from the import machinery, so the
        # error is raised INSIDE the try that wraps the _core imports and the
        # guard has to decide what to do with it. Stubbing sys.modules instead
        # would not do: `from mtl5 import tensor` binds an already-cached
        # module without ever touching it, so a stub that raises on attribute
        # access never fires and the import quietly succeeds.
        class Boom:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mtl5.tensor":
                    raise ImportError("some unrelated dependency is not installed")
                return None

        sys.meta_path.insert(0, Boom())

        try:
            import mtl5
        except ImportError as exc:
            print("RAISED:", exc)
        else:
            print("NO ERROR")
    """)
    out = result.stdout
    ctx = f"\\nstdout={out!r}\\nstderr={result.stderr!r}"

    assert "NO ERROR" not in out, "the unrelated failure never happened" + ctx
    assert "some unrelated dependency is not installed" in out, (
        "the original error must survive" + ctx
    )
    assert "out of sync" not in out, (
        "an unrelated ImportError was mislabelled as a stale extension" + ctx
    )
    assert "pip install -e ." not in out, (
        "must not tell someone with a missing dependency to rebuild C++" + ctx
    )
