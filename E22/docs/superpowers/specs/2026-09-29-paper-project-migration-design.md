# Paper workspace migration design

## Intent and boundary

Create `/home/xyw/paper` as the working home for the frozen E22 model, the
subsequent E23 experiments, and the SAITS and S4M forecast baselines. E22 must
retain its reproducible source and completed results across datasets. E23 must
start from the checked E22 source without inheriting old experiment outputs.
The earlier E1–E21 trial series is not part of this migration. Historical
sources remain untouched as a rollback until the new layout passes acceptance.

The human operator runs the key commands: bulk copies, Git repository creation,
and the final switch of the default working directory. The agent may prepare
manifests, scripts, documentation, and verification commands, and may inspect
their outputs, but must not execute those key migration commands on the
operator's behalf. No training or completed test is rerun to migrate results.

## Chosen approach

Use a curated, copy-first snapshot rather than copying the entire old checkout
or retaining only summary tables. A whole-checkout copy would import old trials,
temporary files, and the `.git` file whose `gitdir` is an invalid Windows path.
Tables alone would lose checkpoints, masks, and provenance. A curated snapshot
keeps the runnable E22 source and all files needed to explain each selected
E22 result while leaving the old tree available for rollback.

```text
/home/xyw/paper/
  data/                # one copy of raw/downloaded/processed datasets
  E22/                 # own Git repository; frozen model source and results
    src/ configs/ scripts/ tests/ integrations/ docs/ artifacts/
    data -> ../data    # or an equivalent relative path adapter
    migration/         # source inventory, path map, integrity and selection records
  E23/                 # own Git repository; checked E22 code seed and E23 design
    src/ configs/ scripts/ tests/ docs/ artifacts/
    data -> ../data
  SAITS/               # local clone of official upstream Git repository
    bridge/ artifacts/ migration/
  S4M/                 # local clone of official upstream Git repository
    bridge/ artifacts/ migration/
```

`data/` holds the source and prepared inputs, not model outputs. Git tracks
source, configuration, documentation, small catalogs, and verification code;
large datasets, masks, checkpoints, and logs stay on disk outside Git's object
database. A manifest records their content hashes. Do not depend on symlinks to
the old project: the only shared data link is relative to `/home/xyw/paper`.

## Source and result ownership

The latest E22 checkout is
`/home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout`.
Its runnable source, all imported modules required by it, package metadata,
configs, tests, background runner, current architecture document, and relevant
E22 result ledgers form the E22 source snapshot. Its dirty and untracked files
are real source inputs, not discarded just because they are absent from `HEAD`.
The original repository's Git history remains in the untouched old project;
the new E22 Git repository begins at the verified frozen snapshot. The E23 Git
repository begins from the same source revision, then owns the E23 design,
implementation plan, and future code. It does not copy E22 artifacts.

Completed E22-family results currently occur for ETTh1, Weather, ETTh2,
ETTm1, ETTm2, and Exchange. For each dataset/scenario included in the current
E22 comparison, copy the selected run plus its required Stage 1 lineage,
checkpoints, fixed masks, Clean/Missing baseline cache, run/evaluation
manifests, and useful job status/logs. ETTh1's main E22 run is the dynamic
temporal two-stage run; Weather's retained W02 run is the E22 architecture with
its selected batch profile. ETTm2's Stage 2 warm-start record retains its
original Stage 2 checkpoint lineage. Do not silently relabel a zero-shot stress
evaluation as a separately trained model. Extra failed, interrupted, and old
E1–E21 folders are excluded from the new curated result tree but remain in the
old project. If an existing ledger or manifest does not unambiguously identify
the selected E22 result, pause and ask the operator instead of selecting by
test error.

SAITS and S4M use the local upstream Git repositories under the old project's
`external/SAITS` and `external/S4M`. Preserve their upstream revision identity
in the new sibling projects and put locally written P4 bridges and completed P4
baseline outputs in the respective project. Their adapters may initially
depend on E22's frozen, version-identified common P4 data/evaluation contract;
that dependency must be explicit and tested rather than hidden through an old
absolute path. The existing cross-model comparison report belongs in E22 as a
historical comparison record and references both baseline projects through the
migration catalog. Only successful selected baseline runs enter the curated
tree; the old tree retains abandoned v1 attempts.

## Immutable records and relocation

Original checkpoints, fixed masks, cached baselines, evaluation JSON, and run
manifests are copied byte-for-byte. Some manifests and old run instructions
contain absolute paths under the old checkout. Rewriting them in place would
change their identity and make historical hashes unverifiable. Instead, a
versioned migration catalog stores each source path, destination path, size,
SHA-256, ownership, dataset/scenario, run identity, and selection/lineage role.
A relocation-aware loader or derived sidecar manifest resolves these paths to
the new local copies and checks their hashes before use. Sidecars point to and
identify the untouched original manifest; they never masquerade as its
original bytes. New run instructions use relative paths or a declared
`/home/xyw/paper` root, not the old checkout path.

At migration inventory time, record a stable source snapshot. If any source
file changes before its copy or the matching destination already contains
different content, stop that item and report the conflict. Do not overwrite a
destination merely because the names match. No old file is deleted or moved.
The current Conda environments remain in place, recorded as dependencies; this
migration does not rebuild CUDA, PyTorch, SAITS, or S4M environments.

## Operator-assisted sequence

1. Agent generates a read-only source inventory and proposed inclusion map,
   including selected result identities and baseline ancestry. Operator reviews
   and runs the provided preparation/copy command.
2. Agent checks copy output and content hashes, then provides the exact Git
   initialization/import commands. Operator runs those commands. The old
   Windows-linked `.git` file is never copied as E22's Git metadata.
3. Agent prepares and tests relative data links, relocation sidecars/loader,
   and executable smoke checks in the new trees. Any command that performs a
   material copy, creates/replaces a repository, or changes the active default
   path is handed to the operator instead of being run by the agent.
4. Agent reports the acceptance evidence. Only after all gates pass does the
   operator run the final working-directory switch. The old project stays
   untouched even after the switch unless the operator later asks to remove it.

Long-running verification, if needed, uses detached jobs with unique job
directories, `console.log`, `launch.json`, `pid`, and `status.json`. Progress
and completion are verified; an unchanged `running` status alone is not proof
of a live process. The migration does not restart cached Clean/Missing jobs.

## Acceptance and failure behavior

- The source inventory and destination catalog cover every selected E22
  dataset/scenario and each selected SAITS/S4M P4 run. The destination file
  count, byte size, and SHA-256 match the inventory; missing or mismatched
  items block cutover.
- `git -C` works independently in E22, E23, SAITS, and S4M. The new E22 and
  E23 repositories have no Windows `gitdir` reference. The SAITS and S4M
  upstream revisions match the recorded source revisions.
- Imports, configuration loading, shared-data paths, fixed-mask loading,
  selected-checkpoint loading, and one short forward pass work from their new
  locations without consulting the old checkout. The original observation
  masks and checkpoint hashes remain unchanged.
- For each migrated dataset, selected MSE, MAE, DRR-MSE, and DRR-MAE in the
  catalog reproduce the archived reported values without retraining or
  reselecting on test data. Where a metric was never reported, label it
  absent rather than inventing it.
- Historical manifest bytes and old directories remain unchanged. Any failed
  gate leaves the old directory as the active workspace and preserves copied
  destination files for diagnosis; there is no partial-success announcement.

After acceptance, `/home/xyw/paper/E23` is the development location for the
existing E23 implementation plan, whose execution still requires its own
review and approval. E22 and both baselines are frozen
comparison sources, with any future baseline rerun placed in a fresh, named
experiment directory rather than overwriting migrated outputs.
