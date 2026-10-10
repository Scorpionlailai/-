# E23 Multi-Expert Explicit Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate an explicit-repair E23 encoder that beats the selected SAITS ETTh1 P4 run on both prediction errors and controlled training/inference speed.

**Architecture:** New E23 modules live beside, not inside, E22. H, V and T each return a fixed-shape evidence record from the same masked patch query; a shared router mixes only their latent residuals with an always-available `none` path, and one decoder creates an observation-clamped sequence. A separate locked-P4 runner handles two-stage training, state pairing and selected testing.

**Tech Stack:** Python 3.10, PyTorch in `/home/xyw/miniforge3/envs/malr`, NumPy, pytest, existing MALR P4 protocol/checkpoint helpers; no new CUDA/PyTorch dependency.

**Spec:** `docs/superpowers/specs/2026-09-29-e23-multi-expert-explicit-recovery-design.md`

## Global Constraints

- E22, SAITS, S4M, their artifacts, and earlier E23 designs are read-only. New artifacts live under `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_multi_expert_v1/`.
- Initial comparison uses the existing ETTh1 P4 source/fixed masks, historical/prediction lengths `96/96`, patch `16/8`, batch `128`, seed `2021`, and separate `100`-epoch stages. Resolve the E22 fixed masks through `../E22/migration/paper_catalog.jsonl` in catalog mode; E23 intentionally has no local result artifacts. Never regenerate Clean/Missing or fixed masks.
- Keep the locked P4 protocol separate from `configs/e23_multi_expert_p4.yaml`; manifests hash both. Model defaults: `C=7`, `N=11`, latent width `D=16`; H top-K `4` and threshold `0.7`; V `K=min(4,C-1)` and offsets `{-1,0,+1}`; T local radius `2` and global boundaries `[2,4,8,16]`; residual scales `0.1`; Stage 1 retrieval weight `0.1` and specialist exposure `20%`. Per damaged target, H/V/T consume at most `4/12/10` candidates before eligibility filtering.
- Stage 1 loss is hidden-coordinate Smooth L1 plus the existing H key paired/contrastive representation objective only. Stage 2 loss is future-forecast MSE only. No candidate counterfactual gain, route loss, teacher replay, or load-balancing loss.
- Stage 2 and evaluation freeze both H memory and its matching query-key encoder, while H messages, V/T, the router, repair core, mask adapter and predictor can train in Stage 2.
- Sources are observed input coordinates or published **training** prototypes only; never generated values, future targets, test labels, padding or an updated validation/test bank. Keep the original observation mask at the prediction interface.
- First-run accuracy gates after validation selection: test MSE `<0.4405843816917229` (SAITS) and MAE `<0.4332482160624463` (E22). Controlled two-stage training time and inference throughput must also beat SAITS on the same device and measurement boundary; archival `163.4+188.6 s` SAITS timing is context, not the speed verdict.
- Controlled E22/SAITS training replays are for timing only: write new jobs/checkpoints under E23's new benchmark artifact root, run no additional selected baseline tests, and leave both baseline repositories and archived artifacts unchanged.
- Long training/evaluation/full-suite jobs use `scripts/run_background.py` with a new job directory and visible progress. Short focused tests may run foreground. Every task commits only its listed files in a functioning repository.
- **Execution preflight:** work from the independent `/home/xyw/paper/E23` Git repository on `paper/e23-migration` (or an isolated worktree made from its reviewed HEAD). Check `git status`, the relative `data -> ../data` link, and the sibling E22 catalog/masks before Task 1. The old checkout with a broken Windows `.git` pointer is rollback only; it is not the execution workspace.

## Review Focus

1. E23 has no local P4 result masks: load the cataloged E22 fixed masks, or reject a missing/changed cataloged mask before a batch (Task 1).
2. Empty H bank, one variable/patch or all experts unavailable: V/T return unavailable where needed, router returns `none=1`, repairs remain finite, and observations stay bitwise unchanged (Tasks 2–4, 7).
3. H group with an ineligible future member: retrieval keeps group-center and causal-member semantics; the future member never becomes a candidate (Task 5).
4. Overlapping final patch or padding: produce one valid output per coordinate with no padding gradient; the original mask survives repatching (Task 7).
5. Corrupt Stage 1 memory/key, E23 config, source CSV or fixed-mask hash: Stage 2/evaluation rejects it before a batch (Tasks 9, 10).

---

## File map

| File | Single responsibility |
| --- | --- |
| `configs/e23_multi_expert_p4.yaml` | E23-only model, optimizer, expert and routing defaults; no P4 data override |
| `integrations/e23/config.py` | Strict E23 settings parser and P4/catalog identity check |
| `src/malr/models/e23/contracts.py` | Expert input/output and router/repair output records, shapes and validity checks |
| `src/malr/models/e23/router.py` | Common type-conditioned confidence scoring, `none` and residual mixing |
| `src/malr/models/e23/variable.py` | Directed prior, sparse preselection and V messages |
| `src/malr/models/e23/temporal.py` | Once-per-source encoding, local anchors and full-lookback banded T retrieval |
| `src/malr/prototypes/tensor_index.py` | Read-only GPU/chunked view of published H snapshots, matching causal query semantics |
| `src/malr/models/e23/historical.py` | Existing H key encoder plus H evidence/quality projection; snapshot refresh only |
| `src/malr/models/e23/model.py` | Base patch encoding, parallel experts, one decoder, explicit repair and forecast adapter |
| `integrations/e23/profile_reference.py` | Read-only E22/SAITS segmented reference profiling |
| `integrations/e23/train.py` | Stage 1/2 loops, retrieval EMA/memory lifecycle, paired checkpoints and manifests |
| `integrations/e23/run.py` | Locked-P4 commands, selected evaluation, DRR/provenance output and progress |
| `docs/experiments/E23_multi_expert_P4_trial.md` | Controlled E23 trial results and comparison, created only after completed runs |

Add `__init__.py` in new packages with the task first creating that package. Keep P4-specific protocol imports in `integrations/e23`, not in generic `src/malr/models/e23`.

### Task 1: Lock E23 settings and measure the reference path

**Files:** Create `configs/e23_multi_expert_p4.yaml`, `integrations/e23/__init__.py`, `integrations/e23/config.py`, `integrations/e23/profile_reference.py`; test `tests/integration/test_e23_config.py` and `tests/integration/test_e23_profile_reference.py`.

**Interfaces:** `load_e23_settings(path: Path, *, protocol: P4Protocol) -> E23Settings` returns a frozen record with `config_sha256`, `d_model=16`, `historical_topk=4`, `historical_threshold=0.7`, `variable_topk=4`, `variable_offsets=(-1,0,1)`, `temporal_local_radius=2`, `temporal_boundaries=(2,4,8,16)`, `residual_scale=0.1`, `retrieval_loss_weight=0.1`, `specialist_exposure=0.2`, `batch_size=128`, `seed=2021`, `stage1_epochs=stage2_epochs=100`, `stage1_lr=0.001`, `stage2_lr=0.0001`, `weight_decay=0.0`. Reject unknown data, mask or target overrides. `load_e23_p4_protocol(*, catalog: MigrationCatalog, protocol_config: Path, mask_artifacts: Path) -> P4Protocol` calls the existing catalog-aware P4 loader against E23's seed config and E22's selected mask root. `profile_selected_reference(*, protocol: P4Protocol, selected_manifest: Path, device: torch.device, warmup_batches: int=5, batches: int=10) -> dict[str, float]` accepts only identity-checked E22/SAITS manifests and reports CPU/CUDA, full-pipeline/model-core elapsed time, segment times and peak memory without mutating online state.

- [ ] **Step 1:** Write `test_e23_config_separates_model_from_p4`: default settings include `D=16`, H/V/T bounds `4/12/10`, Adam learning rates `0.001/0.0001`, batch `128`, seed `2021`, epochs `100/100`; unknown data/mask overrides raise.
- [ ] **Step 2:** Write `test_e23_p4_loads_cataloged_masks`: E23 has no local mask artifact, yet the catalog resolves the E22 validation/test masks with pinned hashes; a tampered temporary catalog/mask copy raises before yielding a batch without changing the archived E22 mask.
- [ ] **Step 3:** Write `test_profile_rejects_wrong_mask_and_preserves_memory`: a tampered selected manifest raises before model execution; a valid synthetic model leaves memory hash unchanged and returns nonnegative timing, recorded warmup/batch counts and distinct cold-start/steady-state fields.
- [ ] **Step 4:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/integration/test_e23_config.py tests/integration/test_e23_profile_reference.py -q`; expect FAIL because the E23 config/profiler modules are absent.
- [ ] **Step 5:** Implement strict E23 settings and catalog-aware P4 loading in `config.py`; create the E23 YAML with model/training settings only and no data split, mask or forecast-target override.
- [ ] **Step 6:** Implement the read-only profiler; use CPU timers/CUDA events, synchronize at measurement boundaries, and include source/mask/selection hashes, cold-start time, warmup, full-pipeline and model-core segments.
- [ ] **Step 7:** Rerun both focused tests; expect PASS. Schedule formal GPU measurement only when idle.
- [ ] **Step 8:** Commit the four new files and both tests.

### Task 2: Common evidence contract and residual router

**Files:** Create `src/malr/models/e23/__init__.py`, `contracts.py`, `router.py`; test `tests/unit/test_e23_router.py`.

**Interfaces:** `ExpertInputs(base_latents, patch_values, patch_mask, patch_validity, rho, patch_starts, query_mean, query_std, damaged)`; `ExpertEvidence(message, quality, available, provenance)`; `RouterOutput(latents, weights, deltas)`; `ExpertRouter.forward(inputs: ExpertInputs, evidence: Mapping[str, ExpertEvidence]) -> RouterOutput`. Evidence keys are exactly `historical`, `variable`, `temporal`; weights order is `none,H,V,T`, shape `[B,C,N,4]`.

- [ ] **Step 1:** Write `test_router_masks_unavailable_experts_and_preserves_base`: with all three unavailable, assert `weights[...,0]==1`, `latents==base_latents` and finite gradients; with one available, assert weights sum to one. Reject NaN scores and wrong shapes; when `none` dominates, keep the expert's `available=True` while its weight approaches zero.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/unit/test_e23_router.py -q`; expect FAIL on missing imports.
- [ ] **Step 3:** Implement type adapters, shared confidence scorer, separate `none` score, unavailable masking, `scale_e=0.1` residual projections and the exact `z0 + Σπ_e Δ_e` update. Router input features include observed ratio, valid missing ratio and relative target position only.
- [ ] **Step 4:** Rerun the focused tests; expect PASS, including gradient through an enabled expert and the base path.
- [ ] **Step 5:** Commit only Task 2 files.

### Task 3: Sparse directed V expert

**Files:** Create `src/malr/models/e23/variable.py`; test `tests/unit/test_e23_variable.py`.

**Interfaces:** `SparseVariableExpert.forward(inputs: ExpertInputs, *, epoch: int, explore: bool) -> ExpertEvidence`; trainable independent source/target embeddings produce `A[source,target]`. Default `K=min(4,C-1)`, source-patch offsets `(-1,0,1)`, one rotating exploration slot per target only when `explore=True`.

- [ ] **Step 1:** Write `test_v_constructs_only_selected_local_edges`: for P4 `C=7`, assert candidate count per target `<=12`, source never equals target, direction is `source→target`, and changing an unselected variable cannot change the message. Add `C=1` and no-visible-source unavailable cases; verify V quality uses prior, visible overlap and lag only.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/unit/test_e23_variable.py -q`; expect FAIL.
- [ ] **Step 3:** Implement Top-K/exploration indices **before** gathering source values or running the message MLP. Apply observed-source eligibility and quality features from visible coordinates only; use stable tie-breaking and deterministic epoch/target rotation.
- [ ] **Step 4:** Rerun focused tests; expect PASS and finite gradient to selected relation embeddings/messages.
- [ ] **Step 5:** Commit only the V file and tests.

### Task 4: Full-lookback T expert without repeated source encoding

**Files:** Create `src/malr/models/e23/temporal.py`; test `tests/unit/test_e23_temporal.py`.

**Interfaces:** `MultiScaleTemporalExpert.forward(inputs: ExpertInputs) -> ExpertEvidence`; local nearest observed left/right within radius 2; global one source per direction and distance band `(2,4],(4,8],(8,16],(16,∞)`, ranked over the full available history by batched query/key scores.

- [ ] **Step 1:** Write `test_t_uses_remote_source_and_encodes_each_source_once`: construct a long window whose only useful source is beyond patch lag 16; assert it is selected, local/global stay inside the one T output, candidate count is `<=10` per target, and a forward hook sees one source-encoder call. For P4 `N=11`, assert the `>16` band is empty. Add `N=1` and no-visible-source unavailable cases.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/unit/test_e23_temporal.py -q`; expect FAIL.
- [ ] **Step 3:** Implement pre-encoding `[B,C,N,D]`, batched ranking, band quota, masks/quality and one final T message. Do not feed restored values back as sources or create target-expanded source MLP input.
- [ ] **Step 4:** Rerun focused tests; expect PASS for distance direction, partial visibility, all-empty fallback and finite gradients.
- [ ] **Step 5:** Commit only the T file and tests.

### Task 5: Tensorized H retrieval with old causal semantics

**Files:** Create `src/malr/prototypes/tensor_index.py`; test `tests/unit/test_e23_tensor_index.py`.

**Interfaces:** `TensorPrototypeIndex.from_snapshot(snapshot: OnlinePrototypeSnapshot, *, device: torch.device) -> TensorPrototypeIndex`; `query(keys: Tensor, starts: Tensor, damaged: Tensor, *, topk: int=4) -> TensorRetrieval`. `TensorRetrieval` includes candidate raw patches, key similarity, start, group support and valid mask, all on the query device.

- [ ] **Step 1:** Write `test_tensor_index_matches_online_bank_query_batch`: on fixed snapshots compare candidate identities/order with `OnlinePrototypeMemory.query_batch`, including score ties, empty groups and a group containing a member newer than the query. Add an empty-bank test.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/unit/test_e23_tensor_index.py -q`; expect FAIL.
- [ ] **Step 3:** Materialize immutable group centers/members once per published epoch; use the legacy algorithm: choose the best **eligible group by full group center**, then rank only that group's causal members by key cosine and earliest start. Chunk queries/members for a fixed memory ceiling; never call `.cpu().numpy()` in `query`.
- [ ] **Step 4:** Rerun focused tests; expect PASS on CPU and, when available, CUDA with the same identities and tolerance for floating-point score differences.
- [ ] **Step 5:** Commit only the index file and tests.

### Task 6: H expert and retrieval quality

**Files:** Create `src/malr/models/e23/historical.py`; test `tests/unit/test_e23_historical.py`.

**Interfaces:** `HistoricalExpert.refresh_snapshot(snapshot: OnlinePrototypeSnapshot) -> None`; `HistoricalExpert.forward(inputs: ExpertInputs) -> ExpertEvidence`. Keep the existing `MaskAwareRetrievalEncoder`/`RetrievalEncoderEMA` architecture; stage trainers own its updates. The expert reads the tensor index and outputs H message plus similarity, visible-coordinate error, top-two margin, visible fraction and support quality.

- [ ] **Step 1:** Write `test_h_rejects_invisible_or_future_evidence`: assert no-bank/all-unobserved query is unavailable; a partial query compares prototype values only on actually observed coordinates; the result contains no future candidate, H returns at most four candidates, and its key encoder has the inherited architecture. Missing margin/support features are finite when fewer than two candidates exist.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/unit/test_e23_historical.py -q`; expect FAIL.
- [ ] **Step 3:** Implement raw-to-query normalization and the H evidence adapter without changing the key encoder topology. Refresh the tensor index only after memory publication or checkpoint load.
- [ ] **Step 4:** Rerun focused tests; expect PASS, finite messages and no per-batch host transfer on the query path.
- [ ] **Step 5:** Commit only the H file and tests.

### Task 7: Compose one explicit-repair encoder and mask-aware forecast adapter

**Files:** Create `src/malr/models/e23/model.py`; test `tests/integration/test_e23_model.py`.

**Interfaces:** `MultiExpertRepairEncoder.forward(values: Tensor, observed_mask: Tensor, query_mean: Tensor, query_std: Tensor, patch_starts: Tensor, *, valid_mask: Tensor | None=None, epoch: int=0, explore_variable: bool=False) -> E23RepairOutput`; `E23RepairOutput(explicit: ExplicitRepairBatch, router: RouterOutput, evidence: Mapping[str, ExpertEvidence])`; `E23ForecastModel.forward(...) -> tuple[Tensor, E23RepairOutput]` uses `MaskAwareForecastModel` with the E22-structure `PatchTSTCore`.

- [ ] **Step 1:** Write `test_e23_repairs_only_missing_and_preserves_mask`: assert observed standardized sequence values are bitwise unchanged, padded coordinates contribute neither output nor gradient, overlapping patches yield one output value, the original mask survives repatching, all-expert-empty routes to base, and `forecast_target` is not accepted by the forward interface.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/integration/test_e23_model.py -q`; expect FAIL.
- [ ] **Step 3:** Implement mask-aware base encoding `Linear(2P,D)`, parallel H/V/T calls, Task 2 router, one `Linear(D,P)` decoder and existing `build_explicit_repair`; re-patch its normalized repaired sequence and original observation mask for the forecast adapter. Do not call the E22 teacher or old candidate-gain code.
- [ ] **Step 4:** Rerun focused tests; expect PASS for shape, provenance and gradients into every enabled expert, router, decoder and predictor.
- [ ] **Step 5:** Commit only the model file and tests.

### Task 8: Stage 1 repair and paired online state

**Files:** Create `integrations/e23/train.py`; test `tests/integration/test_e23_stage1.py`.

**Interfaces:** `train_repair(protocol: P4Protocol, settings: E23Settings, *, output: Path, device: torch.device, epochs: int=100) -> Path`; returns a manifest path selected by lowest fixed-validation repair MSE. The checkpoint contains the model, H bank snapshot, matching frozen/EMA key states and epoch. Its manifest contains source CSV, validation/test mask, protocol config, E23 config, checkpoint and state hashes.

- [ ] **Step 1:** Write `test_stage1_loss_and_snapshot_pairing`: assert `loss == smooth_l1_hidden + 0.1*retrieval_loss`, a two-epoch smoke run publishes H snapshots sequentially, and the selected checkpoint restores matching model/bank/key hashes plus E23/protocol config identities.
- [ ] **Step 2:** Write `test_specialist_exposure_is_reproducible`: fixed seed gives the same 20% damaged-target subset, one available expert plus `none`, and no forced unavailable expert.
- [ ] **Step 3:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/integration/test_e23_stage1.py -q`; expect FAIL.
- [ ] **Step 4:** Implement the P4 adapter using `iter_p4_batches`, existing `OnlinePrototypeMemory` and `RetrievalEncoderEMA`. Stage only fully observed valid training patches at `ref.input_start + patch_index*8`; freeze the H query-key view at epoch start and re-key/publish the bank at epoch end.
- [ ] **Step 5:** Implement hidden-coordinate Smooth L1 plus the existing H representation formula `masked_pair_loss + 0.5*masked_contrastive_loss/log(max(query_count,2))`; apply fixed-seed specialist exposure within the same forward and no candidate-gain loss.
- [ ] **Step 6:** Implement fixed-validation repair MSE selection and atomic model+bank+EMA checkpoint/manifest with both config hashes and code revision; print batch, epoch and validation progress.
- [ ] **Step 7:** Rerun the focused tests; expect PASS and a loadable paired checkpoint.
- [ ] **Step 8:** Commit only `integrations/e23/train.py` and Stage 1 tests.

### Task 9: Stage 2 forecast-only handoff

**Files:** Modify `integrations/e23/train.py`; test `tests/integration/test_e23_stage2.py`.

**Interfaces:** `train_forecast(protocol: P4Protocol, settings: E23Settings, *, stage1_manifest: Path, output: Path, device: torch.device, epochs: int=100) -> Path`; returns a manifest path selected by lowest fixed-validation forecast MSE. It consumes Task 8's paired checkpoint, creates a new optimizer and records reset optimizer/RNG lineage.

- [ ] **Step 1:** Write `test_stage2_freezes_retrieval_but_trains_messages`: a changed E23 config, protocol hash, H bank or matching key hash is rejected before the first batch; after two synthetic epochs H bank/key hashes are unchanged, H message/route parameters receive prediction gradients, and `loss == forecast_mse` with no other term.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/integration/test_e23_stage2.py -q`; expect FAIL.
- [ ] **Step 3:** Implement Stage 1 identity checks and state handoff; freeze the published H bank/key encoder but not H messages, V/T, router, decoder or predictor. Use the same-structure mask-aware predictor, a new optimizer, independent epoch numbering and no fixed repaired-data input.
- [ ] **Step 4:** Implement fixed-validation forecast MSE selection, atomic checkpoint/manifest and per-batch/per-epoch progress; record Stage 1 checkpoint/hash, E23/protocol config hashes and reset optimizer/RNG state. Rehash read-only H state after Stage 2.
- [ ] **Step 5:** Rerun focused tests; expect PASS and strict equality between Stage 2 total loss and forecast MSE.
- [ ] **Step 6:** Commit only the Stage 2 changes and tests.

### Task 10: Selected evaluation, DRR, CLI and performance gate

**Files:** Create `integrations/e23/run.py`; test `tests/integration/test_e23_run.py`; update `integrations/e23/profile_reference.py` only for new E23 timing segments.

**Interfaces:** CLI subcommands `train-repair`, `train-forecast`, `evaluate`; common arguments `--protocol-config`, `--e23-config`, `--migration-catalog`, `--mask-artifacts`, `--output`, `--device`; forecast adds `--stage1-manifest`, evaluate adds `--selection`. `evaluate_selected(protocol: P4Protocol, settings: E23Settings, *, selection: Path, output: Path, device: torch.device) -> dict[str, object]` writes once and returns MSE, MAE, both shared-reference DRRs, expert diagnostics and both config identities.

- [ ] **Step 1:** Write `test_selected_test_rejects_mismatch_and_reports_both_drrs`: source/mask/E23-config/checkpoint hash mismatch and existing output path raise before a test batch; a valid synthetic selected run reports all 2688 windows, MSE/MAE, DRR-MSE/DRR-MAE, `none/H/V/T` availability/weight rates and unchanged H state hash. Test zero Missing−Clean gap produces undefined DRR, not infinity.
- [ ] **Step 2:** Run `PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python -m pytest tests/integration/test_e23_run.py -q`; expect FAIL.
- [ ] **Step 3:** Implement the locked-P4 CLI, identity-checked manifests, read-only selected evaluation and existing shared-reference DRR formula. Add per-batch test progress and segmented E23 profiling; reject overwrite and never use test error to choose a checkpoint.
- [ ] **Step 4:** Rerun focused tests plus E23 config/explicit-repair tests; run existing P4 protocol tests from E22 where the cataloged fixed masks live. Expect PASS; do not copy E22 artifacts into E23 merely to satisfy a test path.
- [ ] **Step 5:** Run `pytest tests/unit tests/integration tests/forecast_compare/test_protocol.py::test_hidden_history_does_not_change_student_or_query_stats -q` detached with `scripts/run_background.py`; inspect `status.json`/`console.log` for exit code `0`. Run the existing artifact-dependent P4 comparison tests from E22 as a separate regression job; the copied E23 seed tests that hard-code local E22 artifacts are not an E23 gate because E23 intentionally contains no E22 results.
- [ ] **Step 6:** Commit only Task 10 files and tests.

### Task 11: Controlled P4 trial and evidence report

**Files:** Create `docs/experiments/E23_multi_expert_P4_trial.md`; do not edit prior result files.

**Interfaces:** Consume Tasks 1 and 10 timing/evaluation records and the selected Stage 1/2 manifests. Produce a reproducible comparison with E22 and SAITS, including the two cached Clean/Missing denominators; do not create a new test-selection path.

- [ ] **Step 1:** On an idle GPU, run a one-batch E23 forward/backward smoke and identity-checked E22/SAITS/E23 segmented profiling with the same batch `128`, warmup `5`, timed batches `10`, device and window order. Verify finite gradients, no H-state mutation, segment sums, data-pipeline/model-core timing, cold-start time and peak memory; record GPU, CUDA and PyTorch versions.
- [ ] **Step 2:** If H/V/T profiling exposes a bottleneck, optimize only that execution path and rerun its candidate/gradient equivalence tests before formal training; record the before/after profile. Never change P4 masks, window shape or epoch count to pass this gate.
- [ ] **Step 3:** Launch Stage 1 via `scripts/run_background.py` in a new `e23_multi_expert_v1/jobs/stage1_*` directory; verify PID, epoch/batch log progress and `status.json` exit code `0` before using its selected checkpoint.
- [ ] **Step 4:** Launch Stage 2 separately from that selected Stage 1 manifest; verify both config/manifest hashes, frozen H state, epoch/batch log progress and exit code `0`.
- [ ] **Step 5:** Lock Stage 2 by validation MSE, then launch exactly one selected test job; verify manifest/checkpoint/mask/config hashes, 2688 test windows, both DRRs and exit code `0`. Do not tune against this test result.
- [ ] **Step 6:** Replay E22 and SAITS `100+100` epochs for timing only into fresh E23 benchmark artifact directories, with no selected baseline tests, and compare their training wall time with E23 including validation, snapshot/checkpoint work. Compare steady inference windows/s on identical fixed inputs without computing labels or test metrics; include input preparation, repair and forecast and report cold start separately. Use the same GPU/batch/warmup/timing boundaries; archived `3812.7/352.1 s` are context only.
- [ ] **Step 7:** Write the report with forecast MSE/MAE, DRR-MSE/DRR-MAE, repair diagnostic, stage times, full-pipeline/model-core throughput, cold start, peak memory, gate/availability rates, environment versions and artifact hashes. State whether MSE `<0.4405843816917229`, MAE `<0.4332482160624463`, and controlled training/inference speed against both SAITS and E22 pass. Check figures against generated JSON, then commit only the report.

## Final review and report

Before Task 11, review the whole code diff against the spec and audit the original-mask contract and artifact hashes. Do not equate old historical wall times with controlled speedup. If E23 misses either accuracy metric or either controlled SAITS speed criterion, report that plainly and open a separate design iteration rather than rewriting this one. A successful single-seed P4 result remains exploratory until independently replicated.
