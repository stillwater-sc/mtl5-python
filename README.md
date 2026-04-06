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

## Development

```bash
pip install -e ".[dev]"
pytest -v
```
