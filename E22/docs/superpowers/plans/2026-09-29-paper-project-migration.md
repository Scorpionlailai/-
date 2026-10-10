# Paper Workspace Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish independently versioned E22, E23, SAITS and S4M projects under `/home/xyw/paper`, with verified E22-family results and shared data, without changing historical evidence or the old workspace.

**Architecture:** A reviewed selection manifest identifies only retained E22-family and selected P4 baseline artifacts. A safe copy tool checks source and destination hashes; a migration catalog maps untouched historical paths to new files, and a strict resolver makes new runs independent of the old checkout. The operator executes bulk copy, Git creation, and final cutover commands; the agent prepares and verifies them.

**Tech Stack:** Python 3.10, `pathlib`, `hashlib`, JSON/JSONL, pytest, local Git `/home/xyw/miniforge3/envs/malr/bin/git`, existing `malr`/`malr-saits`/`malr-s4m` Conda environments.

**Spec:** `docs/superpowers/specs/2026-09-29-paper-project-migration-design.md`

## Global Constraints

- Source E22 checkout: `/home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout`; old upstream baselines: `/home/xyw/missing_aware_latent_recovery/external/SAITS` and `/home/xyw/missing_aware_latent_recovery/external/S4M`. Never mutate or delete these source trees.
- Target root: `/home/xyw/paper`; its existing `E22/` is empty at plan time. Stop on a pre-existing conflicting destination; do not overwrite it.
- E22 result datasets: ETTh1, Weather, ETTh2, ETTm1, ETTm2, Exchange. Include only selected E22-family runs/scenarios and their Stage 1 lineage, fixed masks, Clean/Missing caches, manifests, evaluations and necessary logs; E1–E21, failed jobs and abandoned baseline v1 attempts remain only in the old tree.
- `paper/data` is the one raw/downloaded/processed input copy. E22/E23 use a relative `data -> ../data` link. E23 receives checked E22 source and the existing E23 spec/plan, but no E22 artifacts.
- Historical checkpoints, masks, cached baselines, run manifests and evaluation records are copied byte-for-byte. Never rewrite them to change absolute paths; a catalog and relocation-aware loader resolve them with SHA-256 checks.
- SAITS/S4M preserve recorded upstream commit identity. S4M is locally dirty and detached: clone its upstream `HEAD` and migrate only the reviewed P4 overlay/bridge, never blindly transfer every modified upstream file. Explicitly verify its protected-file semantics.
- Source, config, docs, code and small catalogs belong in each project's Git repository; large data, masks, checkpoints and logs stay on disk but out of Git. Existing Conda/CUDA environments are reused.
- **Operator boundary:** The human executes bulk copy, `git clone`/`git init`/initial repository commits, and final path switch. The agent may run read-only inventory/verification and short tests, and write preparation code/docs; it must hand key commands to the human and wait for their output before dependent steps.
- All preparation code and tests are written under the new `/home/xyw/paper/E22` target, not the old checkout. Until its Git repository is initialized, save those files without committing; thereafter commit them only in the new E22 repository. The old checkout is read-only during implementation.
- Any long-running copy or verification must be detached via `scripts/run_background.py` or equivalent with unique job directory and visible progress. Never rerun cached Clean/Missing or old selected tests.

## Review Focus

1. A destination file already exists with different bytes: refuse overwrite, preserve both trees, and name the conflict (Task 2).
2. A source changes between inventory and copy, or a copy is interrupted: reject changed source and resume only files whose destination hash matches (Task 2).
3. A historical manifest points at an old path that still exists: resolver must load the **new** checked copy, not silently use the old file (Task 4).
4. S4M's detached, dirty checkout or an unrecognized upstream revision: refuse an unofficial baseline snapshot rather than claim official identity (Task 5).
5. An E22 scenario has no unambiguous selected run, or DRR is absent: stop for selection review and label absent metrics rather than rank by test score or fabricate results (Tasks 1, 6).

---

## File map

| File | Responsibility |
| --- | --- |
| `scripts/paper_migration/__init__.py` | Package boundary for migration tooling |
| `scripts/paper_migration/selection.py` | Explicit selected-run map and dependency closure checks |
| `scripts/paper_migration/inventory.py` | Read-only source inventory, SHA-256, source revisions and entry schema |
| `scripts/paper_migration/copy_verified.py` | Operator-run copy engine with progress, no-overwrite and resume semantics |
| `scripts/paper_migration/verify.py` | Read-only destination, Git, manifest and metric acceptance checks |
| `src/malr/migration.py` | Strict original-path-to-new-path resolver used only by relocated E22/E23 copies |
| `src/malr/cli.py` | Apply resolver to existing manifest/checkpoint loading when catalog is configured |
| `integrations/forecast_compare/protocol.py` | Keep P4 identity checks working with relocated paths |
| `migration/paper_selection.json` | Reviewed dataset/scenario/run identity and source inclusion map |
| `migration/paper_catalog.jsonl` | Versioned original/destination path, size and SHA-256 records |
| `docs/experiments/paper_migration_acceptance.md` | Final source/target evidence and command handoff record |

File-map paths are relative to the new `/home/xyw/paper/E22` unless a task says otherwise. The new repositories each receive only their appropriate parts. Tests for migration tooling live under `tests/migration/`. Do not turn this into a general-purpose backup utility.

### Task 1: Lock selected evidence and source identity

**Files:** Create `scripts/paper_migration/__init__.py`, `selection.py`, `inventory.py`, `migration/paper_selection.json`; test `tests/migration/test_inventory.py`.

**Interfaces:** `load_selection(path: Path) -> Selection`; `build_inventory(selection: Selection, *, source_checkout: Path, old_project: Path) -> list[MigrationEntry]`; each entry contains `source`, `destination_relative`, `size`, `sha256`, `owner`, `dataset`, `scenario`, `run_id`, `role`. Inventory must not write or modify a source file.

- [ ] **Step 1:** Read the existing ledgers and selected manifests; list exact selected E22-family runs per ETTh1/Weather/ETTh2/ETTm1/ETTm2/Exchange scenario, their Stage 1/2 lineage and Clean/Missing/fixed-mask dependencies. List SAITS v2 and S4M selected P4 outputs and upstream commit IDs. Require operator review if a selection is ambiguous; never choose by test MSE/MAE.
- [ ] **Step 2:** Write `test_inventory_rejects_ambiguous_or_missing_lineage` and `test_inventory_excludes_old_trials_and_source_mutation` using a tiny synthetic tree. Assert no `E1–E21`/failed/v1 result is selected and that inventory returns hashes without writing to the source.
- [ ] **Step 3:** From `/home/xyw/paper/E22`, run `PYTHONPATH=. /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/migration/test_inventory.py -q`; expect import failure.
- [ ] **Step 4:** Implement selection and inventory; record a source-tree digest derived from inventoried file hashes, E22 source paths including dirty/untracked modules, SAITS upstream `660b87f19c1277065f314f24134f646229e89ca9` and S4M upstream `a718823addd3606e763dfc261174e0135b2535f4`. Include all dependency files needed to load selected results, not only summary spreadsheets.
- [ ] **Step 5:** Rerun the focused tests; expect pass. Present the concrete selection table and source counts to the operator before any bulk copy. Keep Task 1 files only in the new target; the old checkout stays read-only.

### Task 2: Safe operator-run copy and resumable integrity check

**Files:** Create `scripts/paper_migration/copy_verified.py`; test `tests/migration/test_copy_verified.py`.

**Interfaces:** `copy_entries(entries: Sequence[MigrationEntry], *, target_root: Path, owners: frozenset[str], apply: bool=False, progress: Callable[[int,int,str],None]|None=None) -> CopyReport`; default is dry-run; `--apply` is an operator-only CLI mode. Owners are `data`, `E22`, `E23`, `SAITS`, `S4M`. It copies to a temporary sibling, fsyncs and atomically renames only into a nonexistent destination; matching existing bytes are skipped.

- [ ] **Step 1:** Write `test_copy_refuses_conflicting_destination`, `test_copy_refuses_changed_source`, `test_copy_resumes_matching_files`, and `test_copy_rejects_path_escape`; assert source tree and mismatching destination remain unchanged.
- [ ] **Step 2:** Run focused tests; expect missing module failure.
- [ ] **Step 3:** Implement owner filtering, path containment, source pre/post hash checks, no-overwrite, matching-file resume, per-file/byte progress and a final copy report. After each successful operator copy, atomically update `migration/paper_catalog.jsonl` from all verified copied entries. Never follow a source or destination symlink outside the reviewed roots.
- [ ] **Step 4:** Rerun focused tests; expect pass. Agent runs dry-run only and gives the operator the exact `--apply --owners data,E22` command in a unique detached job with `tail -f` and `status.json` paths. Wait for operator-run completion before moving on. Do not populate SAITS/S4M directories before their Git clones.
- [ ] **Step 5:** Verify destination entries against the inventory. Keep preparation files in the new target until the operator initializes its repository.

### Task 3: Freeze E22 source/data layout and seed E23

**Files:** Add `.gitignore`, `migration/README.md` and relative `data` link in new E22/E23 copies; add `tests/migration/test_project_layout.py` in the new E22 target.

**Interfaces:** `verify_project_layout(paper_root: Path) -> LayoutReport`; checks E22/E23 source imports, `data` link target, lack of copied `.git` file, E23 artifact emptiness, and E22 model/config file hashes from the catalog.

- [ ] **Step 1:** Write `test_layout_rejects_windows_gitdir_and_old_absolute_data_link` and `test_e23_has_source_but_no_e22_artifacts` against synthetic trees.
- [ ] **Step 2:** Run focused tests; expect failure on missing checker.
- [ ] **Step 3:** Implement the checker and a reviewed E22 source/E23 seed entry set. E23 receives the E23 design and plan documents; E22 receives current E22 architecture, result ledgers, source/config/tests/background runner. Exclude `.codex_tmp`, caches, old trial outputs and `.git` from both.
- [ ] **Step 4:** Rerun focused tests; expect pass. Agent gives operator the exact `--apply --owners E23`, relative-link and new `git init`/initial-commit commands, with `git status` and `git ls-files` prechecks proving large artifacts remain untracked. Operator runs those commands; agent verifies output.
- [ ] **Step 5:** Run short imports/config loading in `malr` from both new paths. Once the operator initializes the E22/E23 repositories, commit only reviewed source/catalog paths in the new repositories through the agreed operator Git workflow.

### Task 4: Relocate immutable E22 records without old-path fallback

**Files:** Create `src/malr/migration.py`; minimally modify `src/malr/cli.py` and `integrations/forecast_compare/protocol.py` in new E22/E23 copies; test `tests/migration/test_relocated_artifacts.py`.

**Interfaces:** `MigrationCatalog.load(path: Path) -> MigrationCatalog`; `MigrationCatalog.resolve(original: str|Path, *, expected_sha256: str|None=None) -> Path`. When catalog mode is active, only a mapped destination under `/home/xyw/paper` may be used; an unmapped/mismatched artifact raises `ValueError` even if the old path still exists. Legacy mode without a catalog retains original behavior.

- [ ] **Step 1:** Write `test_resolver_chooses_new_copy_even_if_old_exists`, `test_resolver_rejects_hash_or_path_mismatch`, and `test_original_manifest_hash_is_unchanged`. Add integration test loading the migrated E22 selected Stage 1/Stage 2 checkpoint chain using the catalog.
- [ ] **Step 2:** Run focused tests; expect missing resolver failure.
- [ ] **Step 3:** Implement catalog parsing and strict resolution, then thread it through the CLI's manifest/checkpoint and P4 protocol path reads. Historical JSON stays byte-for-byte identical. New commands declare the catalog explicitly; never transparently fall back to the old project.
- [ ] **Step 4:** Rerun focused tests; expect pass. Run one short selected-checkpoint load and fixed-mask read from new E22 while a test hook rejects any attempted read under the old checkout.
- [ ] **Step 5:** Commit only relocation code/tests in the new E22 Git repo through the agreed operator Git workflow; copy the same portability code to E23 before its first model commit.

### Task 5: Materialize SAITS and S4M as identifiable sibling baselines

**Files:** Place reviewed bridge modules under new `SAITS/bridge/` and `S4M/bridge/`; add each project's `migration/README.md` and source catalog entries; test `tests/migration/test_baseline_layout.py`.

**Interfaces:** `verify_baseline(name: Literal['SAITS','S4M'], *, project: Path, catalog: MigrationCatalog) -> BaselineReport`; returns upstream commit, bridge hash, selected evaluation hash and portability status.

- [ ] **Step 1:** Write `test_baseline_rejects_wrong_upstream_commit` and `test_s4m_overlay_does_not_change_protected_semantics`. A dirty upstream clone or missing local P4 overlay must fail preflight, not be labeled official.
- [ ] **Step 2:** Run focused tests; expect failure.
- [ ] **Step 3:** Implement verifier. SAITS bridge takes `integrations/forecast_compare/saits_model.py` and `saits_run.py`; S4M bridge takes `s4m_p4.py`, the `integrations/s4m` provenance/overlay files, and the approved `s4m/malr_fair` overlay. Keep old v2 SAITS and selected official P4 S4M manifests, checkpoints, evaluations and comparison report lineage; exclude failed v1 attempts.
- [ ] **Step 4:** Rerun focused tests; expect pass. Hand operator exact local-clone commands into empty SAITS/S4M directories, then `--apply --owners SAITS,S4M` reviewed bridge/result copy commands; the agent does not run them. Verify upstream commits and protected-file hashes after operator reports completion.
- [ ] **Step 5:** Run short import/preflight checks in `malr-saits` and `malr-s4m` using the frozen E22 P4 contract and new paths; do not rerun training or selected tests. New baseline Git additions/commits are operator-owned.

### Task 6: Destination-wide verification and result catalog

**Files:** Create `scripts/paper_migration/verify.py`, `docs/experiments/paper_migration_acceptance.md`; test `tests/migration/test_verify.py`.

**Interfaces:** `verify_migration(*, paper_root: Path, inventory: Path, selection: Path) -> AcceptanceReport`; reports per-project Git identity, file count/size/hash, selected run/checkpoint chain, fixed masks, cached Clean/Missing, and archived MSE/MAE/DRR-MSE/DRR-MAE without recomputing or ranking them.

- [ ] **Step 1:** Write `test_verify_fails_on_missing_checkpoint_or_mask`, `test_verify_labels_unreported_drr_absent`, and `test_verify_blocks_partial_copy`. A mismatch must make overall acceptance false.
- [ ] **Step 2:** Run focused tests; expect failure.
- [ ] **Step 3:** Implement read-only verification and a machine-readable report. Compare reported figures to copied JSON/XLSX/CSV ledger entries with documented metric spaces; never infer absent DRR or use a new test score to select a run.
- [ ] **Step 4:** Rerun focused tests and the relevant existing relocation/P4 smoke tests; expect pass. For a full or long test suite, use a new detached job and confirm exit code `0` in `status.json`.
- [ ] **Step 5:** Produce the human-readable acceptance document with exact source/target counts, hashes, upstream revisions, excluded material and all remaining failures. Do not claim migration complete if any gate fails.

### Task 7: Operator cutover and E23 handoff

**Files:** Update the new E22/E23/SAITS/S4M READMEs with portable commands and operator handoff; keep old READMEs untouched.

**Interfaces:** `paper_root=/home/xyw/paper`; explicit commands for checking `git -C` in all four projects, resolving selected artifacts, loading one selected checkpoint, and opening E23 as the next workspace.

- [ ] **Step 1:** Review Task 6 acceptance evidence with the operator. If any selected run, hash or import is missing, stop; leave old checkout active.
- [ ] **Step 2:** Give operator the final path-switch commands and a `pwd`/`git status`/smoke-command postcheck. Operator runs them, not the agent.
- [ ] **Step 3:** Confirm the new E23 working directory and repeat a short read-only smoke. Record the cutover time and operator confirmation in the acceptance document. Leave the old project on disk as rollback.
- [ ] **Step 4:** Present the E23 implementation plan for separate execution approval; migration approval does not authorize starting E23 model changes.

## Execution handoff and safety

After operator initialization, use focused commits in the new projects and stage explicit paths; do not stage or commit the dirty old checkout. The user's material-copy and Git-initialization commands remain human-run even if a task's code/tests are delegated. No task may silently run an operator command to unblock itself. A resumed copy may skip only byte-identical destinations. Before any claimed success, check fresh acceptance output and the destination Git state, not merely a successful copy process exit.
