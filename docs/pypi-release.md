# Releasing `mtl5` to PyPI

A step-by-step, repeatable procedure for publishing the `mtl5` package to
[PyPI](https://pypi.org/). It reflects the consolidated release pipeline that is
now live in the repo.

> **Status:** fully wired and in production. `mtl5` 5.7.0 → 5.7.1 → 5.7.2 are
> published on [PyPI](https://pypi.org/project/mtl5/), cut through this exact
> automation — one workflow (`wheels.yml`), keyless OIDC Trusted Publishing, no
> tokens. §1 is the machinery, §6 is the day-to-day happy path, and the one-time
> account/publisher setup (§2) is already done for `mtl5`.

---

## 1. What already exists

| Piece | File | What it does today |
|---|---|---|
| Version policy | `pyproject.toml` `[tool.semantic_release]` | `mtl5` tracks MTL5's `major.minor`; semantic-release manages only the **patch** component from conventional commits. Minor/major bumps are manual. |
| Release + build + publish | `.github/workflows/wheels.yml` | **One workflow, three triggers.** `push: main` → `semantic-release` bumps/tags/creates the GitHub Release, then (if it released) builds wheels + sdist and publishes to PyPI. `release: published` → a human-cut release builds + publishes to PyPI. `workflow_dispatch` → dry-run to TestPyPI. |
| CI gate | `.github/workflows/ci.yml` | Lint + build/test matrix + ecosystem + zlib lanes on every push/PR. |

> **Why everything is in one workflow (not `release.yml` + a reusable `wheels.yml`).**
> PyPI Trusted Publishing **does not support reusable workflows** — the OIDC
> `workflow` claim becomes ambiguous and the upload is rejected
> ([PyPI docs](https://docs.pypi.org/trusted-publishers/troubleshooting/#reusable-workflows-on-github)).
> And a GitHub Release created by CI's `GITHUB_TOKEN` does not trigger other
> workflows (anti-recursion), so a separate publish workflow would never fire on
> an automated release. Consolidating into `wheels.yml` keeps the OIDC claim on a
> single **direct** workflow, so one trusted publisher covers every path with no
> tokens. (`upload_to_pypi = false` stays in `pyproject.toml` — semantic-release
> is not the publisher; the dedicated `publish-pypi` job is.)

---

## 2. One-time setup (do this once, ever)

> **Already complete for `mtl5`** — the name is claimed, the PyPI + TestPyPI
> trusted publishers exist, and the `pypi` environment is configured. This
> section is kept as the record and the template for a new sister package.

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
2. Do the same `wheels.yml` publisher on **TestPyPI** too, for the dry run
   (§4) — trusted publishers are per-index, so register on each.
3. No secret is stored. The publish jobs declare `permissions: id-token: write`
   and authenticate via the minted OIDC token.

**One publisher covers everything.** All publishing runs from `wheels.yml` — the
automated (`push`), manual (`release`), and dry-run (`workflow_dispatch`) paths
are all jobs in that one workflow, so the OIDC `workflow` claim is always
`wheels.yml`. Do **not** add a `release.yml` publisher: there is no `release.yml`,
and reusable-workflow claims are unsupported by PyPI anyway (see §1).

> **Adding a publisher to an *existing* project** goes through the project's own
> **Publishing** page (pypi.org → Your projects → `mtl5` → Manage → Publishing),
> not the account-level **pending publisher** flow — pending publishers are only
> for project names that do not exist yet, so PyPI rejects one for a name that is
> already taken.
>
> **First-publish chicken-and-egg:** a project-scoped trusted publisher can only
> be added *after* the project exists. For the very first upload, use PyPI's
> **pending publisher** flow (**Account → Publishing → Add a pending publisher**),
> which pre-authorizes owner/repo/workflow/environment for a project name that
> does not exist yet; PyPI creates the project on first successful publish.

---

## 3. The publish job (already wired up)

`.github/workflows/wheels.yml` contains the `publish-pypi` job below. It runs
after the wheel/sdist jobs, gathers every artifact, and uploads via Trusted
Publishing. It fires on a human-cut Release **or** on a push where the `release`
job cut one — but never on the `workflow_dispatch` dry-run (that goes to
TestPyPI):

```yaml
  publish-pypi:
    needs: [release, build-wheels, build-sdist]
    if: >-
      !cancelled()
      && needs.build-wheels.result == 'success'
      && needs.build-sdist.result == 'success'
      && (github.event_name == 'release'
          || (github.event_name == 'push' && needs.release.outputs.released == 'true'))
    runs-on: ubuntu-latest
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
- The gate publishes on `release` (manual) and on `push` when a release was cut,
  and excludes `workflow_dispatch` — so the dry-run only ever reaches TestPyPI.
- No `password:` — auth is the OIDC token, enabled by `id-token: write` plus the
  single `wheels.yml` trusted publisher on PyPI (§2.2).
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

> **✅ Validated 2026-08-07.** This lane has been exercised end-to-end. A
> `workflow_dispatch` run of `wheels.yml` built the full matrix (cp310/311/312
> × Linux `manylinux_2_28_x86_64` / macOS `arm64` / Windows `amd64`, 64-bit
> only per §3), and `publish-testpypi` uploaded `mtl5 5.7.0` (9 wheels + sdist)
> to <https://test.pypi.org/project/mtl5/> — keyless, via the pending trusted
> publisher, which TestPyPI promoted to the real project on first publish. A
> clean-venv install from TestPyPI then imported the compiled `_core`, reported
> the expected `build_info()` (`native_fast_gemm`/`highway_simd` on), and passed
> `norm`/`dot` smoke checks. The production PyPI path differs in three ways:
> trigger (`release` vs `workflow_dispatch`), target index, and duplicate
> handling — `publish-testpypi` sets `skip-existing: true` so re-runs are no-ops,
> whereas `publish-pypi` deliberately omits it, so a duplicate version fails
> loudly rather than being silently skipped.

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
   # Pin the EXACT version you dispatched. --extra-index-url also exposes
   # production PyPI (needed for numpy et al.), so an *unpinned* `mtl5` would
   # resolve to the highest version across BOTH indexes — i.e. production, not
   # the artifact under test. Dry-run a dev version (see the note below) so only
   # TestPyPI can satisfy the pin.
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ "mtl5==5.8.0.dev1"
   python -c "import mtl5; print(mtl5.__version__); print(mtl5.build_info())"
   pytest --pyargs mtl5   # if tests are packaged; otherwise run the repo tests
   ```
   The `--extra-index-url` lets `numpy` and friends resolve from real PyPI while
   the pinned `mtl5` version — which exists only on TestPyPI — comes from
   TestPyPI. Without the pin, pip picks the highest version across both indexes,
   which is now production (5.7.2 > any TestPyPI dry-run at 5.7.x).

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
      - A new MTL5 `major.minor` (e.g. bumping to `5.9.0`) → **edit
        `pyproject.toml` by hand first**, commit it under a non-releasing type,
        tag by hand (see §6 step 1), then let semantic-release resume patch
        management from there. See the policy comment in `pyproject.toml`.
- [ ] The `GIT_TAG` for `mtl5` in `CMakeLists.txt` is the MTL5 release this
      version claims to track, and it is a **tag, not a branch**. FetchContent
      runs at `pip install` time, so a branch there means the published wheel
      and any from-source install are built against a moving target. The
      `universal` pin next to it is subject to the same rule. Verify the pinned
      combination actually compiles before tagging — a pin that predates an API
      the bindings call fails at build time, not at import.
- [ ] `README.md` is correct **as the PyPI long description** (it is the
      `readme`) — for a reader who found the package on PyPI, not a source
      checkout. In particular:
      - The Install section leads with `pip install mtl5` (installing the
        published wheel), **not** `pip install .` (source-only — fails in an
        arbitrary directory and misleads PyPI users). Source-build steps belong
        under a clearly-labeled "From source" subsection.
      - It **renders** on PyPI's stricter markup renderer — relative image links
        and internal-only URLs will not resolve.
      Note: PyPI does not let you edit a published version's description (it is
      baked into the uploaded wheel/sdist metadata), so a README fix only reaches
      the project page on the **next** release.
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

   > **A minor/major bump must not ride a release-worthy commit.**
   > semantic-release reads the *current* version from the last git tag, not
   > from `pyproject.toml` — it only ever writes that file. So merging a
   > hand-set `version = "5.9.0"` together with a `feat:`/`fix:`/`perf:`/
   > `refactor:` commit makes the `release` job compute 5.7.2 → **5.7.3** from
   > the tag and rewrite your 5.9.0 back down. Land the bump under a type that
   > is *not* in `patch_tags` (`build:`, `chore:`, `docs:`, `ci:`) so the
   > `release` job no-ops, then create the tag by hand — **pinned to the merge
   > commit you verified**:
   >
   > ```bash
   > SHA=$(git rev-parse origin/main)   # the commit CI went green on
   > gh release create v5.9.0 --target "$SHA" --title v5.9.0 --notes-file <notes>
   > ```
   >
   > `--target` is not optional here. When the tag does not already exist, `gh`
   > creates it from the tip of the default branch *at that moment* — so an
   > unrelated merge landing between your CI check and your `gh release create`
   > would get tagged and published instead. (If you tagged and pushed by hand
   > instead, use `--verify-tag` so a typo'd tag name fails rather than
   > silently creating a new one.)
   >
   > That fires the `release: published` trigger, which builds at the tag and
   > publishes — the same path used for 5.7.0–5.7.2. Patch releases need none
   > of this; step 2 handles them.

2. **`wheels.yml` runs automatically** on the push to `main`, doing everything
   in one workflow run:
   - The `release` job runs `semantic-release version` — computes the bump,
     updates `pyproject.toml` and the changelog, commits `chore(release):
     v{version}`, tags `v{version}`, pushes, and creates the GitHub Release. It
     outputs whether it released and the new tag.
   - If it released, the `build-wheels`/`build-sdist` jobs (checking out that
     tag) and then `publish-pypi` run **in the same run**, uploading to PyPI via
     OIDC (`workflow=wheels.yml`, the existing trusted publisher).
   - *(If no release-worthy commits are present, the `release` job is a no-op and
     the build/publish jobs are skipped — expected.)*

   > **Why one workflow, same run.** A GitHub Release created by CI's
   > `GITHUB_TOKEN` does **not** trigger other workflows (anti-recursion), so a
   > separate publish workflow would never fire on an automated release; and
   > reusable workflows can't be used because PyPI Trusted Publishing doesn't
   > support them (§1). Keeping build+publish as native jobs in `wheels.yml`
   > sidesteps both — no PAT/App token. The `release: published` trigger still
   > serves the **manual** `gh release create` path (how 5.7.0–5.7.2 were cut).
3. **Verify** the release landed:
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
| Spurious `1.0.0` on a pre-1.0 / tagless repo | `semantic-release` with `allow_zero_version` unset force-escapes `0.x` to `1.0.0` on the first run — even for non-release commits — and can publish it | Set `allow_zero_version = true`; bootstrap the first release from an **annotated** baseline tag. `mtl5` (5.x, tagged) is not affected — this bit the `universal_dtypes` sister package and is why its `wheels.yml` also guards against a tagless auto-release. |

---

## 9. First-time setup — completed

All done for `mtl5`; kept as the record and the template for a new package:

- [x] `mtl5` reserved on PyPI + TestPyPI; 2FA on. *(§2.1)*
- [x] GitHub **trusted publisher** on both PyPI and TestPyPI: owner
      `stillwater-sc`, repo `mtl5-python`, `workflow=wheels.yml`,
      `environment=pypi`. *(§2.2)*
- [x] `publish-pypi` job present in `wheels.yml`. *(§3)*
- [x] `pypi` environment configured in repo settings. *(§3)*
- [x] TestPyPI dry-run + install-test validated. *(§4)*
- [x] First real release cut — 5.7.0, then 5.7.1 and 5.7.2. *(§6)*

Day-to-day, releasing is now just: **merge conventional commits → the pipeline
does the rest** (§6). For a new sister package, walk §2 → §4 → §6 in order.
