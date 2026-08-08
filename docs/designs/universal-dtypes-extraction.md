# `universal_dtypes` — a standalone NumPy-dtype package for Universal number types

## Status

Design proposal. Supersedes the "Phase 2 in mtl5-python" framing of issue #14 by
moving the custom-dtype work into a dedicated package. No code yet. Builds on
`docs/designs/custom-dtype-feasibility.md` (the framework-by-framework analysis)
and revisits **where** that work should live, not **whether** it is feasible.

## Motivation

Stillwater Universal provides custom number types (posit, fixpnt, lns, cfloat).
Registering them as first-class NumPy dtypes is a *number-representation*
concern, and it depends on **Universal**, not on MTL5 (the linear-algebra
library). Today that dependency is inverted:

- `mtl5/pandas_ext.py` (the Phase 1 pandas `ExtensionDtype` for posit16) stores
  a `float64` NumPy array and quantizes values by round-tripping through
  **temporary** `mtl5.vector_posit16` objects — i.e. a scalar number-system dtype
  reaching **up** into a matrix library for its quantization. (It does not hold a
  `DenseVector_posit16`; the point stands either way — the dependency runs the
  wrong direction.)

Three problems follow from housing dtypes in mtl5-python:

1. **Inverted layering.** A `posit16` array is meaningful with zero matrix code.
   The correct dependency graph is `universal_dtypes → Universal` and
   `mtl5-python → {universal_dtypes, mtl5}`.
2. **Global dtype identity.** NumPy dtype registration is process-global. If
   mtl5-python registers `posit16` and any other library also does, the result
   is two incompatible dtype objects (or an outright collision). The only way to
   have **one** canonical `posit16` that every framework shares is a single
   package — exactly the role `ml_dtypes` plays for `bfloat16` across JAX/TF.
3. **Reuse without the matrix library.** Someone doing pure-NumPy or pandas work
   with posits should not have to install MTL5.

The precedent is direct and load-bearing: [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes)
is a small, framework-agnostic package that JAX and TensorFlow depend on for ML
number formats. `universal_dtypes` is the same idea for Universal's formats, and
a natural **sister repo of Universal**.

## Proposed design

Create `universal_dtypes` as a standalone package and sister repo of Universal:

- **Repo:** `github.com/stillwater-sc/universal_dtypes`
- **Depends on:** Universal (header-only, via CMake FetchContent), NumPy. **Not
  MTL5.**
- **Provides:** true NumPy custom dtypes for the Universal number systems,
  following the `ml_dtypes` C++ pattern (cast tables, ufunc loops, comparison,
  sort, formatting, pickling), plus an optional pandas adapter.
- **Consumed by:** mtl5-python (for interop and the linear-algebra layer), and
  any other library or user wanting Universal dtypes in NumPy. Interop is
  *conversion* today — the MTL5 factories accept contiguous `float64` and copy
  into native vectors, so a `universal_dtypes` array is cast to `float64`
  (1-D for `mtl5.vector`, 2-D for `mtl5.matrix`) before the copy; non-finite
  inputs map to the element type's exceptional value (posits have a single NaR,
  not separate ±inf/NaN). True **zero-copy** (a `universal_dtypes` array and an
  MTL5 container sharing one buffer) is a future goal that requires a defined
  memory contract; see Open decisions.

## Naming

Three names are in play and they are not the same string:

| Name | Value | Separator | Rule |
|---|---|---|---|
| Python import | `import universal_dtypes` | **underscore** | forced — hyphens are illegal in identifiers |
| PyPI project | `universal-dtypes` | **hyphen** (canonical) | PEP 503 normalizes `_`/`.`/`-` → `-`; `pip install universal_dtypes` still resolves |
| GitHub repo | `stillwater-sc/universal_dtypes` | **underscore** | free choice; underscore matches the import name |

**Decision: use `universal_dtypes` (underscore) for the repo and import name, and
let PyPI canonicalize the project to `universal-dtypes`.** This mirrors the
reference project exactly — `ml_dtypes`'s repo/import are `ml_dtypes` while its
PyPI name is `ml-dtypes`. It also keeps install-name and import-name visually
identical (`pip install universal_dtypes` → `import universal_dtypes`), avoiding
the "install X, import Y" gotcha of the older scikit-learn/`sklearn` style.

Both `universal-dtypes` and `universal_dtypes` were unclaimed on PyPI (both
returned 404 on 2026-08-08) — reserve the name early, since registry ownership
can change and a distribution name is immutable once first published.

*Naming note:* the earlier working title was `mp_dtypes` (mixed-precision
dtypes), tied to the mpdsp product line. `universal_dtypes` was chosen instead
because Universal's types are *alternative number systems*, not solely a
mixed-precision concern, and a Universal-branded name attracts non-Stillwater
adopters the way `ml_dtypes` does — the goal is to be *the* canonical posit dtype
package, not a product-specific one.

## Scope and the `universal_dtypes` ↔ `mtl5` boundary

The line must be drawn explicitly or the two packages will contend over where an
operation like "sum of posits" belongs.

**`universal_dtypes` owns** (element-level, no linear algebra):
- The NumPy dtype objects and their registration.
- Casts to/from `float32`/`float64`/`int32`/`int64`.
- **Element-wise** ufunc loops (`+ - * / **`, `abs`, `sqrt`, `exp`, `log`,
  trig) computed in true Universal arithmetic — **not** upcast-to-float32 the way
  `ml_dtypes` handles some formats. This is the point of posits.
- Reductions (`sum`, `mean`, `min`, `max`), comparison, sort, `repr`/`str`,
  pickling.
- Optional pandas `ExtensionDtype`/`ExtensionArray` (thin, no MTL5).

**`mtl5` keeps** (linear algebra + accumulation strategy):
- Dense/sparse linear algebra over these element types.
- **The quire / accumulator mixed-precision policy** (`mtl5.mixed`,
  `accumulator=` on `dot`/`norm`/`matmul`, iterative refinement). Fused,
  exact accumulation is an *algorithm* choice that belongs with the solvers, not
  with the scalar dtype.

The clean phrasing: **`universal_dtypes` is the element type and its scalar/
element-wise arithmetic; `mtl5` is what you do with arrays of them, including how
you accumulate.**

**Reductions cross this line and must be documented as deliberately different.**
`np.sum(posit_array)` / `np.mean(...)` accumulate **naively in the element type**
— each partial sum rounds to a posit — which is *not* the same value as the
**quire**-accumulated reductions in `mtl5` (exact until the final round). Note
`mtl5` exposes the accumulator policy on `dot`, `norm`, `frobenius_norm`,
`matvec` and `matmul` (via `mtl5.mixed`); there is **no** free `mtl5.sum`, so the
accuracy-preserving analogue of `np.sum` is `mtl5.dot(x, ones)` /
`mtl5.norm`-style calls with `accumulator=`. This split is by design:
`universal_dtypes` gives ordinary NumPy semantics for the dtype, and exactness
lives in `mtl5`. The two answers differ on ill-conditioned data, so both packages'
docs should state it and point users needing exactness at the `mtl5` accumulator
API.

The precise `universal_dtypes` reduction contract — result dtype, whether
`dtype=` overrides accumulation precision, empty-input behavior, and final-
rounding rules — is **deliberately left to implementation** (see Open decisions);
pinning it in this proposal would invent decisions no consumer has driven yet.

## Framework support

Following the feasibility analysis (`custom-dtype-feasibility.md`):

- **NumPy — core.** Expensive but the right and only real target; the whole
  package exists for this.
- **pandas — optional `[pandas]` extra.** A thin `ExtensionDtype`/
  `ExtensionArray` over the NumPy dtype, pure Python, no MTL5. (Migrated from
  today's `mtl5/pandas_ext.py`, re-based off the NumPy dtype instead of
  `DenseVector_posit16`.)
- **PyTorch — explicitly out of scope.** PyTorch's dtype enum is closed; the
  only options are storing values as `uint16` or a fragile tensor subclass. At
  most, `universal_dtypes` may offer `posit16_to_torch`/`torch_to_posit16`
  storage helpers. It will **not** promise a `torch.dtype`.

## Relationship to issue #14

Issue #14 ("NumPy custom DType registration for Universal types, Phase 2")
**is** the implementation work; this design changes only its *home* (a new repo)
and its *dependency* (Universal, not MTL5). Everything the issue estimated still
holds and must not be undersold:

- ~5000 lines of C/C++ per dtype family, plus shared infrastructure.
- ~4–8 weeks for the first family (posit16 as proof of concept), 1–2 weeks per
  additional configuration once the infrastructure exists.
- NumPy's DType API (legacy `PyArray_RegisterDataType` vs. the NEP 42+ DType API)
  is a real, ongoing maintenance treadmill — `ml_dtypes` has a team behind it.

"Extract" is therefore a misnomer: almost none of this exists yet (Phase 1 is a
pandas-only, pure-Python posit16 dtype). `universal_dtypes` is mostly the
greenfield #14 build, done in the right place — plus a rewrite of the pandas
array to drop its MTL5 dependency.

## Migration & compatibility (mtl5-python)

mtl5-python 5.7.x already exposes `mtl5.Posit16Dtype` / `mtl5.Posit16Array`
(pandas). The transition:

1. mtl5-python adds a dependency on `universal_dtypes`. Because the pandas types
   live in the **optional** `[pandas]` extra, mtl5-python must depend on
   `universal-dtypes[pandas]` (not bare `universal-dtypes`) — otherwise the
   re-exported `Posit16Dtype`/`Posit16Array` would be missing. (Equivalently,
   gate the re-export on pandas being importable, matching how `mtl5.__init__`
   already guards its pandas surface.)
2. mtl5-python re-exports `Posit16Dtype`/`Posit16Array` (and future dtypes) from
   `universal_dtypes` for a deprecation window, so existing imports keep working.
3. **Preserve behavior, not just import paths.** Re-exporting keeps
   `mtl5.Posit16Dtype` importable, but the new implementation must match — or
   deliberately, and in a documented breaking release, change — the observable
   semantics of today's `Posit16Array`: scalar indexing returns a Python
   `float`, `to_numpy()` yields `float64`, and assignment quantizes through
   posit16. Pin these with tests carried over from the current pandas suite
   before deleting `mtl5/pandas_ext.py`.
4. `mtl5/pandas_ext.py`'s independent implementation is removed once the
   re-export and the behavioral tests are in place.
5. The `mtl5.vector_posit16(...)` factories and the `mixed`/accumulator surface
   stay in mtl5-python.

This is a minor-version-worthy reorganization under the project's versioning
policy (minor tracks the upstream library; a dependency/layout change of this
size is not a mere patch).

## Build & dependencies

`universal_dtypes` mirrors mtl5-python's build shape: scikit-build-core +
nanobind (or the NumPy C API directly, per the `ml_dtypes` approach), fetching
Universal headers via CMake FetchContent, wheels via cibuildwheel, and PyPI
publishing via Trusted Publishing (OIDC) — the same keyless pipeline this repo
now uses. It versions alongside Universal.

## Cost and scheduling

Issue #14 was deliberately gated behind *"KPU hardware GA **or** concrete
downstream demand."* Repackaging does not change that calculus. The decision to
start `universal_dtypes` should be pulled by a real consumer:

- **mpdsp (mixed-precision DSP)** is the most likely trigger. If it needs NumPy
  posit/cfloat arrays, that is precisely the "concrete downstream demand" #14
  named, and `universal_dtypes` is its natural foundation.

Absent such a pull, the layering argument still stands, but the ~40k-line,
maintenance-heavy reality argues for waiting rather than building speculatively.

## Open decisions

1. **Binding tech:** follow `ml_dtypes` (NumPy C API directly) or use nanobind as
   mtl5-python does? `ml_dtypes` predates a nanobind-based dtype story; the NumPy
   DType API is C-level regardless, so the choice is mostly about the surrounding
   scaffolding.
2. **Legacy vs. NEP 42 DType API:** which to target first, given NumPy 2.x. This
   dominates the maintenance profile and should be decided before the posit16
   proof of concept.
3. **First family:** posit16 (per #14) — confirm it is also the format mpdsp
   needs first.
4. **Zero-copy contract (deferred):** to move from conversion to true zero-copy
   between a `universal_dtypes` array and an MTL5 container, define the shared
   memory layout (element width/encoding), strides, buffer ownership and
   lifetime, and mutability — and whether MTL5 factories gain a
   borrow-a-buffer entry point. Until that exists, interop is a copy.
5. **Reduction/ufunc contract (deferred):** for the NumPy ufuncs and reductions,
   settle the result dtype, whether `dtype=` overrides accumulation precision,
   empty-input semantics, and where the final rounding falls. Decide at
   implementation with the first consumer, not speculatively here.

## Recommendation

Endorse the split. When a consumer (likely mpdsp) pulls it:

1. Reserve `universal_dtypes`/`universal-dtypes` on PyPI now (cheap, immutable
   later).
2. Stand up `stillwater-sc/universal_dtypes` — NumPy core + optional pandas
   extra, depending on Universal, **not** MTL5, **no** torch promise.
3. Implement issue #14's `ml_dtypes` pattern there (posit16 first).
4. Make mtl5-python depend on it and re-export for a compat window; keep the
   quire/accumulator story in mtl5.
5. Settle the two open API decisions (binding tech, legacy vs. NEP 42) before the
   proof of concept.
