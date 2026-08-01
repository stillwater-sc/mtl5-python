"""Build introspection, backend reporting, threading, and GIL-release behaviour.

The threading tests run in fresh subprocesses on purpose: MTL5 sizes its thread
pool once, on first use, so anything that has already run a kernel in this
interpreter has fixed the pool and would make the assertions order-dependent.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import mtl5

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_fresh(code: str, env_extra: dict[str, str] | None = None):
    """Execute `code` in a clean interpreter that has never touched MTL5."""
    env = dict(os.environ)
    env.pop("MTL5_NUM_THREADS", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )


class TestBuildInfo:
    def test_keys_and_types(self):
        info = mtl5.build_info()
        assert isinstance(info, dict)
        for key in ["blas", "lapack", "native_fast_gemm", "highway_simd", "kpu"]:
            assert key in info, f"build_info() missing '{key}'"
            assert isinstance(info[key], bool)

    def test_kpu_not_built(self):
        # No KPU hardware support exists yet; if this ever flips, the backend
        # dispatch tests below need revisiting.
        assert mtl5.build_info()["kpu"] is False

    def test_agrees_with_backend_list(self):
        info = mtl5.build_info()
        backends = mtl5.backends()
        assert ("blas" in backends) == info["blas"]
        assert ("native" in backends) == info["native_fast_gemm"]
        assert "reference" in backends, "reference is always the fallback"


class TestBackendReporting:
    def test_get_backend_is_first_in_dispatch_order(self):
        assert mtl5.get_backend() == mtl5.backends()[0]

    def test_get_backend_is_a_real_backend(self):
        assert mtl5.get_backend() in {"kpu", "blas", "native", "reference"}

    def test_set_backend_accepts_compiled_in(self):
        for name in mtl5.backends():
            mtl5.set_backend(name)  # must not raise

    def test_set_backend_accepts_cpu_alias(self):
        mtl5.set_backend("cpu")

    def test_set_backend_unknown(self):
        with pytest.raises(RuntimeError, match="Unknown backend"):
            mtl5.set_backend("nonexistent")

    def test_set_backend_rejects_uncompiled(self):
        """A backend that is not in this build must fail loudly, not no-op."""
        missing = [b for b in ("blas", "kpu") if b not in mtl5.backends()]
        if not missing:
            pytest.skip("every candidate backend is compiled into this build")
        with pytest.raises(RuntimeError, match="not available in this build"):
            mtl5.set_backend(missing[0])


class TestThreading:
    def test_get_num_threads_positive(self):
        assert mtl5.get_num_threads() >= 1

    def test_default_is_serial(self):
        r = run_fresh("import mtl5; print(mtl5.get_num_threads())")
        assert r.returncode == 0, r.stderr
        assert int(r.stdout.strip()) == 1, "MTL5 must stay serial by default"

    def test_env_var_honoured(self):
        if os.cpu_count() is None or os.cpu_count() < 2:
            pytest.skip("needs >= 2 logical cores")
        r = run_fresh(
            "import mtl5; print(mtl5.get_num_threads())",
            env_extra={"MTL5_NUM_THREADS": "2"},
        )
        assert r.returncode == 0, r.stderr
        assert int(r.stdout.strip()) == 2

    def test_set_before_first_use(self):
        if os.cpu_count() is None or os.cpu_count() < 2:
            pytest.skip("needs >= 2 logical cores")
        r = run_fresh("import mtl5;n = mtl5.set_num_threads(2);print(n, mtl5.get_num_threads())")
        assert r.returncode == 0, r.stderr
        assert r.stdout.split() == ["2", "2"]

    def test_clamped_to_hardware_concurrency(self):
        r = run_fresh("import os, mtl5;print(mtl5.set_num_threads(10_000), os.cpu_count())")
        assert r.returncode == 0, r.stderr
        got, cpus = (int(x) for x in r.stdout.split())
        assert got == cpus

    def test_rejects_zero(self):
        r = run_fresh(
            "import mtl5\n"
            "try:\n"
            "    mtl5.set_num_threads(0)\n"
            "except ValueError:\n"
            "    print('rejected')\n"
        )
        assert r.stdout.strip() == "rejected", r.stderr

    def test_too_late_raises_instead_of_silently_ignoring(self):
        """The pool is sized once; asking to resize it must be an error."""
        if os.cpu_count() is None or os.cpu_count() < 2:
            pytest.skip("needs >= 2 logical cores")
        r = run_fresh(
            "import numpy as np, mtl5\n"
            "mtl5.norm(np.ones(4))          # materializes the pool at size 1\n"
            "try:\n"
            "    mtl5.set_num_threads(2)\n"
            "except RuntimeError as e:\n"
            "    print('raised' if 'already sized' in str(e) else 'wrong-message')\n"
            "else:\n"
            "    print('silently-ignored')\n"
        )
        assert r.stdout.strip() == "raised", (r.stdout, r.stderr)


class TestGilRelease:
    """The kernels must not hold the interpreter for their whole duration."""

    def _progress_during(self, work) -> int:
        started = threading.Event()
        done = threading.Event()
        error = []

        def worker():
            started.set()
            try:
                work()
            except Exception as exc:  # pragma: no cover - surfaced via assert
                error.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=worker)
        t.start()
        assert started.wait(30), "worker never started"
        spins = 0
        while not done.is_set() and spins < 50_000_000:
            spins += 1
        t.join(120)
        assert not error, error
        return spins

    def test_matmul_releases_gil(self):
        n = 600
        rng = np.random.default_rng(0)
        A = rng.standard_normal((n, n))

        spins = self._progress_during(lambda: [mtl5.matmul(A, A) for _ in range(3)])
        # Holding the GIL for the whole call would leave the main thread at ~0.
        assert spins > 1000, f"main thread made almost no progress ({spins} spins)"

    def test_solve_releases_gil(self):
        n = 400
        rng = np.random.default_rng(1)
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        b = rng.standard_normal(n)

        spins = self._progress_during(lambda: [mtl5.solve(A, b) for _ in range(5)])
        assert spins > 1000, f"main thread made almost no progress ({spins} spins)"
