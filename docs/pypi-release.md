# Releasing `mtl5` to PyPI

A step-by-step, repeatable procedure for publishing the `mtl5` package to
[PyPI](https://pypi.org/). This document reflects the release machinery that
already lives in the repo and describes the one piece that is deliberately not
wired up yet: the final upload to PyPI.

> **Status:** As of this writing there is **no PyPI release** of `mtl5`. The
> automation below already tags versions, generates GitHub Releases, and builds
> multi-platform wheels — but the artifacts stop at GitHub. This document closes
> that last gap.

---

## 1. What already exists

| Piece | File | What it does today |
|---|---|---|
| Version policy | `pyproject.toml` `[tool.semantic_release]` | `mtl5` tracks MTL5's `major.minor`; semantic-release manages only the **patch** component from conventional commits. Minor/major bumps are manual. |
| Tag + GitHub Release | `.github/workflows/release.yml` | On push to `main`, runs `semantic-release version`, commits the bump, pushes a `v{version}` tag, and creates a GitHub Release with generated notes. |
| Wheel + sdist build | `.github/workflows/wheels.yml` | On **Release published**, `cibuildwheel` builds cp310/311/312 wheels for Linux/macOS/Windows (no musllinux) and an sdist. **Artifacts are uploaded to the workflow run only — not to PyPI.** |
| CI gate | `.github/workflows/ci.yml` | Lint + build/test matrix + ecosystem + zlib lanes on every push/PR. |

The two intentional stubs to be aware of:

- `pyproject.toml` → `[tool.semantic_release.publish]` → `upload_to_pypi = false`
- `release.yml` declares `permissions: id-token: write` with a `# trusted
  publishing (future PyPI)` comment, but no publish job consumes it yet.

**The only missing step is a job that takes the built distributions and
uploads them to PyPI.** Everything else is a matter of one-time account setup.

---

## 2. One-time setup (do this once, ever)

### 2.1 Reserve the project name

1. Create a PyPI account (and a [TestPyPI](https://test.pypi.org/) account —
   they are separate) for the release owner, ideally under an
   `stillwater-sc`-owned email with 2FA enabled.
2. Confirm the name `mtl5` is available: <https://pypi.org/project/mtl5/>.
   If it is taken, decide on a fallback distribution name (e.g. `mtl5-python`)
   and update `[project].name` in `pyproject.toml` — this is the name users
   `pip install`, and it is **immutable once first published**, so choose
   deliberately.
3. Add a second owner (a co-maintainer or a team account) so releases are not
   bottlenecked on one person.

### 2.2 Authentication: Trusted Publishing (OIDC)

**Decision (mpdsp deployment): use Trusted Publishing — no API tokens.**
Managing PyPI API tokens was judged too error-prone; OIDC keeps the publish
path keyless.

> **This repo is on github.com.** The `origin` reads `git@github.sw:...`, but
> `github.sw` is **only an SSH-config alias** (`~/.ssh/config`: `Host github.sw`
> → `HostName github.com`, `IdentityFile ~/.ssh/sw_github`) that selects the
> Stillwater SSH identity. It is **not** a GitHub Enterprise host. So Actions
> run on github.com and PyPI's OIDC issuer
> (`https://token.actions.githubusercontent.com`) trusts them directly —
> **no mirror, no special routing.** (Aside: PyPI has no SSH auth path, so the
> SSH key is irrelevant to publishing either way — OIDC is what authenticates.)

Setup:

1. On PyPI: **project → Publishing → Add a new trusted publisher (GitHub)**.
   - Owner: `stillwater-sc`
   - Repository: `mtl5-python`
   - Workflow filename: `wheels.yml`
   - Environment name: `pypi`
2. Do the same on **TestPyPI** for the dry run (§4) — trusted publishers are
   per-index, so register one on each.
3. No secret is stored. The `publish-pypi` job declares `permissions:
   id-token: write` and authenticates via the minted OIDC token.

> **First-publish chicken-and-egg:** a project-scoped trusted publisher can only
> be added *after* the project exists. For the very first upload, use PyPI's
> **pending publisher** flow (**Account → Publishing → Add a pending publisher**),
> which pre-authorizes owner/repo/workflow/environment for a project name that
> does not exist yet; PyPI creates the project on first successful publish.

---

## 3. The publish job (already wired up)

`.github/workflows/wheels.yml` contains the `publish-pypi` job below. It runs
after the wheel/sdist jobs, gathers every artifact, and uploads via Trusted
Publishing:

```yaml
  publish-pypi:
    needs: [build-wheels, build-sdist]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    environment: pypi
    permissions:
      id-token: write          # mint the OIDC token PyPI verifies; no password
    steps:
      - uses: actions/download-artifact@v5
        with:
          path: dist
          merge-multiple: true   # flatten wheels-*/ and sdist/ into dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist
```

Notes:
- Runs on github.com (the `github.sw` origin is just an SSH alias for it, §2.2),
  so PyPI's OIDC trusts these runners directly — no mirror or special routing.
- `merge-multiple: true` collapses the `wheels-<os>` and `sdist` artifacts into
  a single `dist/` directory, which is what the publish action expects.
- Gating on `github.event_name == 'release'` keeps the `workflow_dispatch`
  path (used for build smoke-tests) from ever publishing.
- No `password:` — auth is the OIDC token, enabled by `id-token: write` plus the
  trusted publisher registered on PyPI (§2.2).
- Leaving `upload_to_pypi = false` in `pyproject.toml` is fine — semantic-release
  is not the publisher here; the dedicated job is. Do **not** also enable
  semantic-release publishing, or you will get double uploads.

---

## 4. Dry run on TestPyPI (do this before the first real release)

Prove the whole path end-to-end against TestPyPI so the first real upload is
boring. This is a **standing, repeatable lane** — no temporary edits. The
`publish-testpypi` job in `wheels.yml` runs only on `workflow_dispatch`,
publishes to TestPyPI via the trusted publisher registered in §2.2, and carries
`skip-existing: true` so re-runs at an unchanged version don't fail.

1. Make sure `wheels.yml` is committed and pushed to `main` (a workflow must be
   on GitHub, and on the default branch, to be dispatchable).
2. Trigger the build+TestPyPI-publish run:
   ```bash
   gh workflow run wheels.yml --ref main
   gh run watch                       # or: gh run list --workflow=wheels.yml
   ```
   (Or in the UI: **Actions → Build wheels → Run workflow → main**.)
   The `build-wheels`/`build-sdist` jobs run, then `publish-testpypi`. The
   release-only `publish-pypi` job is skipped — a dispatch cannot reach
   production PyPI.
   > If the `pypi` environment has required reviewers, the publish step pauses
   > for approval — approve it in the run page to proceed.
3. Confirm the release appears at `https://test.pypi.org/project/mtl5/`.
4. Install it into a clean environment on each target OS/Python and smoke-test:
   ```bash
   python -m venv /tmp/mtl5-test && source /tmp/mtl5-test/bin/activate
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ mtl5
   python -c "import mtl5; print(mtl5.__version__); print(mtl5.build_info())"
   pytest --pyargs mtl5   # if tests are packaged; otherwise run the repo tests
   ```
   The `--extra-index-url` is needed so `numpy` and friends resolve from real
   PyPI while `mtl5` comes from TestPyPI.

> **Iterating on TestPyPI:** TestPyPI versions are immutable just like PyPI's.
> `skip-existing: true` means a re-run at the *same* version is a no-op (the
> old files stay). To actually test new artifacts, bump to a throwaway dev
> version first — e.g. set `version = "5.7.0.dev1"` in `pyproject.toml` on a
> scratch branch, dispatch, then `.dev2`, etc. Never let a `.devN` reach `main`.

---

## 5. Pre-release checklist (every release)

Before letting a release go out, confirm on `main`:

- [ ] CI is green on the commit being released (lint, build/test matrix,
      ecosystem, zlib lanes).
- [ ] `CHANGELOG.md` has the intended entries; the `<!-- version list -->`
      insertion flag is intact (semantic-release inserts the new section there).
- [ ] The version in `pyproject.toml` is correct for the intended bump:
      - A `feat:`/`fix:`/`perf:`/`refactor:` commit → semantic-release bumps the
        **patch** automatically.
      - A new MTL5 `major.minor` (e.g. bumping to `5.8.0`) → **edit
        `pyproject.toml` by hand first**, commit it, then let semantic-release
        resume patch management from there. See the policy comment in
        `pyproject.toml`.
- [ ] `README.md` renders correctly as the PyPI long description (it is the
      `readme`). Check for anything that breaks on PyPI's stricter renderer —
      relative image links and internal-only URLs will not resolve.
- [ ] `LICENSE` (MIT) and `[project].license` agree. *(Note: `license` is
      currently `{text = "MIT"}`; modern setuptools/PyPI metadata prefers the
      SPDX form `license = "MIT"`. Harmless today, worth tidying.)*
- [ ] The distribution name in `[project].name` is the one you intend to ship
      **permanently** — it cannot be changed after first publish.

---

## 6. Cutting a release (the repeatable happy path)

Once §2 and §3 are done, every subsequent release is just:

1. **Merge conventional commits to `main`.** A `feat:`/`fix:` (etc.) commit is
   what drives a patch bump. For a major/minor bump aligned with MTL5 upstream,
   land a commit that manually sets `[project].version` in `pyproject.toml`.
2. **`release.yml` runs automatically** on the push to `main`:
   - `semantic-release version` computes the bump, updates `pyproject.toml`,
     and inserts the changelog section.
   - A `chore(release): v{version}` commit is pushed with a `v{version}` tag.
   - A GitHub Release is created with generated notes.
   - *(If no release-worthy commits are present, nothing happens — this is
     expected.)*
3. **`wheels.yml` runs automatically** on **Release published**:
   - Builds wheels + sdist across the matrix.
   - The new `publish-pypi` job uploads them to PyPI.
4. **Verify** the release landed:
   ```bash
   pip index versions mtl5        # or visit https://pypi.org/project/mtl5/
   pip install mtl5==<version>
   python -c "import mtl5; print(mtl5.__version__)"
   ```

There is no manual `twine upload` in the happy path — CI owns the upload so
releases are reproducible and auditable.

---

## 7. Manual fallback (break-glass)

If CI is unavailable and a release must go out by hand:

```bash
# From a clean checkout at the tagged commit:
python -m pip install --upgrade build twine
python -m build                      # builds sdist + a local wheel
#   NOTE: `python -m build` produces a wheel for THIS machine only.
#   A hand-built release is not multi-platform — prefer fixing CI.
python -m twine check dist/*
python -m twine upload dist/*        # prompts for token; use __token__ / pypi-...
```

Use this only as a last resort; a manual wheel ships one platform/Python combo,
whereas `wheels.yml` ships the full matrix. Prefer re-running the workflow.

---

## 8. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `403 Forbidden` / OIDC auth fails | The trusted publisher on PyPI does not match the run (owner, repo, workflow, or environment name), or `id-token: write` is missing on the job | Verify the PyPI trusted publisher's owner/repo/`workflow=wheels.yml`/`environment=pypi` exactly match the running workflow. |
| `403` on the very first publish | No project yet, so no project-scoped trusted publisher exists | Register a **pending publisher** on PyPI first (§2.2); it creates the project on first publish. |
| `400 File already exists` | That exact version was already uploaded (PyPI is immutable — versions cannot be re-uploaded or deleted-and-reused) | Bump to a new version; never reuse a version number. |
| Long description rejected / renders wrong | `README.md` uses markup PyPI's renderer dislikes, or relative links | Run `twine check dist/*` locally; fix links to be absolute. |
| No release cut after merging | No release-worthy conventional commits since the last tag | Expected; land a `feat:`/`fix:` commit, or bump the version manually. |
| Double upload | Both semantic-release publishing and the `publish-pypi` job enabled | Keep `upload_to_pypi = false`; let only `wheels.yml` publish. |

---

## 9. Summary of first-time actions

1. Reserve `mtl5` on PyPI + TestPyPI; enable 2FA; add a second owner. *(§2.1)*
2. Register the GitHub **trusted publisher** (a *pending* publisher for the
   first release) on both PyPI and TestPyPI: owner `stillwater-sc`, repo
   `mtl5-python`, `workflow=wheels.yml`, `environment=pypi`. *(§2.2)*
3. Confirm the `publish-pypi` job is present in `wheels.yml` — it already is. *(§3)*
4. Create the `pypi` environment in the repo settings. *(§3)*
5. Dry-run against TestPyPI and install-test on every target. *(§4)*
6. Cut the first real release via the normal merge-to-`main` flow. *(§6)*

After that, releasing is: merge conventional commits → CI does the rest.
