# Paper workspace migration

`paper_selection.json` records the selected source and result set. `paper_catalog.jsonl` maps each immutable source file to its verified copy under `/home/xyw/paper`, including size and SHA-256. The old workspace remains a read-only rollback source.

The catalog is an audit record, not permission to fall back to the old paths at runtime. E22 artifacts and shared data are intentionally outside Git; source code and this catalog are versioned.

## Read-only relocation check

From this E22 root, with the existing `malr` environment:

```bash
PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python -m malr.cli inspect-migrated \
  --migration-catalog migration/paper_catalog.jsonl \
  --run-manifest artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context/runs/ETTh1-ours-dea387986927.json \
  --mask-split validation
```

This reads the copied manifest, Stage 1 and Stage 2 checkpoints and fixed mask
without evaluating or writing to the archived experiment. `src/malr/migration.py`
also exposes `MigrationCatalog.resolve` for other checked files. Its catalog
mode refuses an unmapped old path even when the old tree remains present.

For a complete audit, run `scripts.paper_migration.verify` with
`PYTHONPATH=src:.`. The machine report is `migration/acceptance_report.json`;
the human summary is `docs/experiments/paper_migration_acceptance.md`.
