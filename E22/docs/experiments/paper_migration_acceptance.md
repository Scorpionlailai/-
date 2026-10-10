# Paper workspace migration acceptance — 2026-09-29

## Scope and audit record

The workspace is `/home/xyw/paper`: independent E22, E23, SAITS and S4M Git
repositories plus one shared `data` directory. The old
`/home/xyw/missing_aware_latent_recovery` tree remains the rollback source.
The reviewed selection is `../../migration/paper_selection.json` and the
machine-readable acceptance record is `../../migration/acceptance_report.json`.
The latter is generated after commits and intentionally stays outside Git so
it can record the current Git HEADs.

The fresh source inventory has **957 files, 417,653,320 bytes** and digest
`62d7ac52bca804ae1ebff484149032569deb29ee028f45f1ff0c6ba6c770855e`.
All 957 destinations are in the migration catalog. **951 retain the exact
selected source bytes**. Six runtime and bridge source files intentionally
changed for strict relocated loading and the runnable S4M P4 bridge:

- `E22/src/malr/cli.py` and `E22/integrations/forecast_compare/protocol.py`
- `E23/src/malr/cli.py` and `E23/integrations/forecast_compare/protocol.py`
- `S4M/bridge/integrations/forecast_compare/s4m_p4.py` and
  `S4M/bridge/integrations/s4m/verify_upstream.py`

For each of these six, the verifier checks that the first relevant repository
commit contains the source-inventory hash and the working file matches the
current Git HEAD. Checkpoints, fixed masks, caches, run manifests and
evaluations have no such exception. The copied source inventory is checked
afresh before acceptance. The selection includes 33 explicit runs or scenario
records; none is chosen by test score. The verifier also checks each selected
record's manifest, evaluation, checkpoint hash and applicable Stage1 lineage.

The E22 and E23 repositories have separate non-main branches. E23 contains a
checked E22 source seed, the approved equal H/V/T multi-expert design and its
implementation plan, and no copied E22 result artifacts. No E23 training or
model implementation was started. `data -> ../data` is relative in all four
project roots. Large artifacts and data are outside Git.

SAITS is descended from upstream `660b87f19c1277065f314f24134f646229e89ca9`;
S4M is descended from `a718823addd3606e763dfc261174e0135b2535f4`.
The local baseline commits add the reviewed P4 bridge, S4M fair overlay,
approved S4M fair patch, relative data link and handoff notes. The reviewed
patch is installed on `s4m/run.py`, `s4m/data_provider/data_factory.py` and
`s4m/experiments/exp_pretrain1.py`. The S4M verifier checks that patch and
compares protected model, bank and S4 core Git blobs with pinned upstream
metadata. The source S4M checkout's other dirty files were not imported. The catalog has
10 SAITS and 32 S4M selected files; the detached baseline copy job reported
42 copied, 0 skipped, 71,511,029 bytes and exit code 0.

## Archived selected results

The six trained E22 rows below are read from the byte-checked copied JSON
evaluations. MSE and MAE use the `train_standardized` metric space. DRR values
are percentages with the archived Clean/Missing reference.

| Dataset | MSE | MAE | DRR-MSE % | DRR-MAE % |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 0.443195460875 | 0.433248216062 | 59.706939897720 | 55.572311686975 |
| Weather | 0.173257355652 | 0.220932086785 | 97.981130389429 | 84.329911840494 |
| ETTh2 | 0.316341559903 | 0.360221096177 | 43.939214670020 | 33.946072042402 |
| ETTm1 | 0.363905067811 | 0.385846006099 | 62.989474596394 | 62.003162158048 |
| ETTm2 | 0.192111627280 | 0.272766470335 | 50.239196229203 | 50.828072633733 |
| Exchange | 0.091106545745 | 0.210207511078 | 66.119325886028 | 65.373958071399 |

The selected SAITS P4 evaluation reports MSE/MAE
`0.4405843816917229/0.4369578956206099`; the selected official S4M P4
evaluation reports `0.985358276919892/0.7539715807092573`. Their raw
evaluation JSON does not report DRR. Of the 33 selected evaluation JSON
records, 23 lack DRR fields; the machine report labels these `absent`.
Where the copied scenario CSV ledger separately reports a derived DRR, the
machine report records it as `ledger_drr_*` and checks its MSE/MAE and, when
present in both sources, DRR against the copied JSON. No missing DRR value was
invented. The historical P4
comparison note is `ETTh1_P4_forecast_comparison.md`.

## Reproduce the read-only checks

From `/home/xyw/paper/E22`:

```bash
PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python -m malr.cli inspect-migrated \
  --migration-catalog migration/paper_catalog.jsonl \
  --run-manifest artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context/runs/ETTh1-ours-dea387986927.json \
  --mask-split validation

PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python -m pytest \
  tests/migration tests/forecast_compare/test_protocol.py -q

/home/xyw/miniforge3/envs/malr/bin/git -C /home/xyw/paper/E22 status --short --branch
/home/xyw/miniforge3/envs/malr/bin/git -C /home/xyw/paper/E23 status --short --branch
/home/xyw/miniforge3/envs/malr/bin/git -C /home/xyw/paper/SAITS status --short --branch
/home/xyw/miniforge3/envs/malr/bin/git -C /home/xyw/paper/S4M status --short --branch
```

The full hash audit reads about 418 MB. Run it detached using
`scripts/run_background.py` with a fresh job directory and the command
`/usr/bin/env PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python -m
scripts.paper_migration.verify --output migration/acceptance_report.json`.
The job must finish with exit code 0; the JSON `accepted` field must be true.
Read-only baseline preflight commands are in
`../../../SAITS/migration/README.md` and `../../../S4M/migration/README.md`.

## Cutover and exclusions

Use `cd /home/xyw/paper/E23` as the new development working directory, then
run the read-only seed smoke in `SEED_HANDOFF.md`. Shell directory changes are
per session; this document does not alter the user's shell startup directory.
E23 model work begins only after separate approval of its existing plan.

The authorized agent performed the E23 working-directory postcheck at
**2026-09-29 11:48:25 UTC** (`2026-09-29 19:48:25` Asia/Shanghai). It confirmed
`pwd=/home/xyw/paper/E23`, clean `paper/e23-migration` status, a 96/96 config,
`data` resolving to `/home/xyw/paper/data`, and the unchanged seeded README
SHA-256 `443ab20967ff6d0bc78565b5a34ef309a16e9048d83d797224e5d290598a23e5`.
This is the operator confirmation for the agent-run cutover; each interactive
shell still needs its own `cd /home/xyw/paper/E23`.

The migration excludes E1–E21, failed jobs, abandoned SAITS v1 attempts, the
two superseded E23 drafts, old `.git` worktree metadata, cached Python files,
and unapproved S4M dirty changes. No training, Clean/Missing cache generation
or selected test evaluation was rerun to migrate these results.
