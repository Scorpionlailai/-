# E23 seed handoff

This repository starts from a SHA-256 checked copy of the frozen E22 source.
The copied `README.md` describes that E22 source and is preserved byte for byte
as migration evidence. E23 model work has not started.

The only approved next design is the equal H/V/T multi-expert explicit-recovery
proposal in `docs/superpowers/specs/2026-09-29-e23-multi-expert-explicit-recovery-design.md`,
with its implementation plan in
`docs/superpowers/plans/2026-09-29-e23-multi-expert-explicit-recovery.md`.
Earlier E23 drafts are not part of this repository. `data` links to the shared
`../data` directory; no E22 results or checkpoints belong here.

Run future E23 work from `/home/xyw/paper/E23` after separately approving that
implementation plan. Keep any new experiment outputs outside Git.

Read-only seed smoke from this root:

```bash
PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python -c 'from pathlib import Path; import malr.cli; from malr.config import load_config; print(load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml")).data.seq_len)'
```

For catalog mode, use the sibling
`../E22/migration/paper_catalog.jsonl` with
`MigrationCatalog.load(path, project="E23")`. The old E22 README remains a
checked seed file; this handoff is the E23 entry point.
