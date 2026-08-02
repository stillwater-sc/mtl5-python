# mtl5-python

Python bindings for [MTL5](https://github.com/stillwater-sc/mtl5) — NumPy/SciPy/JAX/PyTorch interop with hardware accelerator dispatch.

Built with [nanobind](https://github.com/wjakob/nanobind) for minimal overhead and zero-copy array interop.

## Install

```bash
pip install .
```

Requires Python 3.10+ and a C++20 compiler (GCC 12+, Clang 15+, MSVC 2022).

## Quick start

```python
import numpy as np
import mtl5

# Vectors and norms
v = mtl5.vector(np.array([3.0, 4.0]))
print(mtl5.norm(np.array([3.0, 4.0])))  # 5.0

# Dot product
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print(mtl5.dot(a, b))  # 32.0

# Solve Ax = b
A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([5.0, 7.0])
x = mtl5.solve(A, b)
print(x)  # [1.6, 1.8]
```

## Performance

### Threading

MTL5's kernels are parallel but **serial by default**. The thread pool is sized
once, on the first MTL5 call, and cannot be resized afterwards — so configure it
before you do any work:

```python
import mtl5

mtl5.set_num_threads(8)  # must precede the first MTL5 operation
print(mtl5.get_num_threads())
```

Equivalently, set `MTL5_NUM_THREADS` in the environment before importing:

```bash
MTL5_NUM_THREADS=8 python my_script.py
```

`set_num_threads()` raises `RuntimeError` rather than silently doing nothing if
the pool has already been sized. The count is clamped to hardware concurrency.

All non-trivial kernels release the GIL, so MTL5 work overlaps with other Python
threads instead of blocking the interpreter.

Double-precision `matmul`, 1000×1000, on a 20-core box:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| GFLOP/s | 14.4 | 27.2 | 48.0 | 72.7 |

### Build options

The wheel builds MTL5's blocked GEMM vectorised through
[Google Highway](https://github.com/google/highway) by default. Everything else
is opt-in, since it needs a library on the build machine or makes the binary
non-portable:

```bash
pip install . -C cmake.define.MTL5_WITH_BLAS=ON      # external BLAS
pip install . -C cmake.define.MTL5_WITH_LAPACK=ON    # geev/syev/gesdd dispatch
pip install . -C cmake.define.MTL5_NATIVE_ARCH=ON    # -march=native, local builds only
```

`MTL5_NATIVE_ARCH=ON` lets Highway target AVX2/AVX-512 rather than the x86-64
baseline — a large additional speedup, but the resulting binary only runs on
machines like the one that built it.

Check what a given install actually has:

```python
>>> mtl5.build_info()
{'blas': False, 'lapack': False, 'native_fast_gemm': True,
 'highway_simd': True, 'kpu': False}
>>> mtl5.get_backend()
'native'
```

`set_backend()` validates against this build rather than silently accepting a
backend that was never compiled in; backend selection itself is compile-time.

## Development

```bash
pip install -e ".[dev]"
pytest -v
```
