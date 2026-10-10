# Paper workspace migration — handoff (2026-09-29)

## Continuation result

The migration is complete. See [the acceptance record](paper_migration_acceptance.md)
and `../../migration/acceptance_report.json` for the final checks. The final
audit accepted all 957 selected destinations (417,653,320 bytes): 951 exact
source copies, six reviewed Git source overlays, 33 selected metric records,
and zero errors. E22, E23, SAITS and S4M are clean on their separate
`paper/e22-migration`, `paper/e23-migration`, `paper/saits-p4` and
`paper/s4m-p4` branches. The approved S4M fair patch is installed and its
read-only bridge preflight passes. E23 remains a source seed and approved
design/plan; no E23 model or training was started. Start a new shell's E23
work with `cd /home/xyw/paper/E23`.

## Agreed scope

The destination is `/home/xyw/paper`: independent E22, E23, SAITS and S4M projects, plus one shared `data` directory. E22 is the frozen best-known model with selected completed results for ETTh1, Weather, ETTh2, ETTm1, ETTm2 and Exchange. E23 is **not being implemented now**: it contains a checked E22 source seed and only the latest agreed equal H/V/T multi-expert explicit-recovery design and implementation plan. Do not import the two earlier E23 drafts or start E23 training. The original `/home/xyw/missing_aware_latent_recovery` tree remains untouched as rollback. The user explicitly authorized the agent to execute migration/copy/Git commands; no need to hand critical commands back to the user.

Authoritative design and plan in the old checkout: `docs/superpowers/specs/2026-09-29-paper-project-migration-design.md` and `docs/superpowers/plans/2026-09-29-paper-project-migration.md`. New E23 documents already copied: `E23/docs/superpowers/specs/2026-09-29-e23-multi-expert-explicit-recovery-design.md` and `E23/docs/superpowers/plans/2026-09-29-e23-multi-expert-explicit-recovery.md`. Latest design supersedes old H/V-first E23 drafts.

## State recorded at handoff

- New E22 migration tools, explicit selection, inventory tests, safe hash-checked no-overwrite copy tool, and layout tests exist under `/home/xyw/paper/E22`.
- Data + selected E22 artifacts/source copied by detached job `.migration_jobs/copy_data_e22_20260929_01`, exit code 0: 676 catalog entries, 344,488,455 bytes. A separate read-only check found zero hash mismatches.
- E23 E22-source seed was copied by detached job `.migration_jobs/copy_e23_20260929_01`, exit code 0. Two earlier E23 drafts were then removed from E23, selection, and catalog; their originals remain in the old checkout. E23 has **no** E22 artifacts and **no** E23 model implementation/experiments.
- `E22/data` and `E23/data` are relative links to `../data`. `verify_project_layout(Path('/home/xyw/paper'))` passed and checked 472 cataloged source files. Migration tests: `10 passed`.
- Corrected reviewed selection: 957 files, 417,653,320 bytes, inventory digest `62d7ac52bca804ae1ebff484149032569deb29ee028f45f1ff0c6ba6c770855e`. Current catalog: 915 entries (E22 + data + E23); 42 SAITS/S4M selected entries remain.
- The old checkout's `.git` is an invalid Windows-path worktree pointer. Do not copy it. Explicit working Git binary: `/home/xyw/miniforge3/envs/malr/bin/git`.

## Original continuation checklist — completed

The numbered items below record the handoff-time checklist; all are complete.

1. Finish Task 3: initialize **separate** E22 and E23 Git repositories on non-main branches; audit `.gitignore`, `git status` and `git ls-files` to ensure no binary artifacts/data are tracked. Run short imports/config smoke from both roots. E23 README currently is the unchanged copied E22 README, so add a clear E23 seed handoff README or entry point without changing the hashed seed copy.
2. Task 4: TDD a strict migration-catalog resolver in *new* E22/E23 code. Existing historical manifests contain absolute old paths; keep manifest/checkpoint/mask bytes immutable. In catalog mode, old paths must resolve only to hash-checked new copies, never silently read the still-existing old tree. Test selected checkpoint/mask loading.
3. Task 5: clone local SAITS upstream commit `660b87f19c1277065f314f24134f646229e89ca9` and S4M upstream commit `a718823addd3606e763dfc261174e0135b2535f4` into sibling repos, then use reviewed copy tool to add selected P4 bridge/results. S4M old tree is detached/dirty; clone its committed HEAD and copy only approved overlay. No baseline rerun.
4. Task 6: write destination-wide verifier and acceptance report; compare every selected file's size/SHA-256, run dependencies, metric reports and Git/upstream identities. Label missing DRR as absent; do not invent it or select by test score.
5. Task 7: document portable commands and final cutover; no E23 model implementation until separately approved.

Plan-scoped ledger: `/home/xyw/paper/E22/.superpowers/sdd/2026-09-29-paper-project-migration/progress.md`. Selection: `/home/xyw/paper/E22/migration/paper_selection.json`; catalog: `/home/xyw/paper/E22/migration/paper_catalog.jsonl`. The original plan still says the operator runs key commands, but the user's later authorization supersedes that; see ledger ruling.

## Safety constraints

Use `apply_patch` for file edits; avoid overwriting conflicts; never delete or modify the original checkout. Do not rerun selected training, Clean/Missing caches or tests to migrate results. Long jobs must detach and retain visible `console.log` and `status.json`. The E22/E23 copy alone was **not** the finished migration; SAITS/S4M, Git identities, relocated runtime loading, and acceptance verification have since been completed.
