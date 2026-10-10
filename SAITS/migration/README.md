# SAITS paper baseline

This repository is a local clone of upstream commit
`660b87f19c1277065f314f24134f646229e89ca9`. The checked P4 adapter is
under `bridge/`; selected completed outputs are under `artifacts/` and outside
Git. The original upstream source is preserved at its pinned commit.

The common ETTh1 P4 data and evaluation contract comes from the frozen sibling
`../E22` repository. Set `PYTHONPATH=bridge:../E22/src` when importing the
adapter from this root. The artifact hashes are recorded in
`../E22/migration/paper_catalog.jsonl`.

Read-only preflight from this SAITS root:

```bash
PYTHONPATH=bridge:../E22/src /home/xyw/miniforge3/envs/malr-saits/bin/python -c 'from pathlib import Path; from integrations.forecast_compare.protocol import load_p4_protocol; p=load_p4_protocol(Path("../E22/configs/stage1_seq96_dynamic_temporal_repair.yaml"), Path("../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context")); print(p.source_sha256, p.mask_sha256)'
```

`data -> ../data` supplies the shared input path. This command only reads the
archived P4 masks and data; it does not start SAITS training or evaluation.
