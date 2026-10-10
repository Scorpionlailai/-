# S4M paper baseline

This repository is a local clone descended from upstream commit
`a718823addd3606e763dfc261174e0135b2535f4`. The reviewed P4 bridge is
under `bridge/`, the approved local overlay is `s4m/malr_fair/`, and the
reviewed `s4m-malr-fair.patch` is installed on `s4m/run.py`,
`s4m/data_provider/data_factory.py`, and `s4m/experiments/exp_pretrain1.py`. Selected
completed outputs are under `artifacts/` and outside Git. The protected
upstream model, bank and S4 core files retain their committed bytes.

The common ETTh1 P4 data and evaluation contract comes from the frozen sibling
`../E22` repository. Set `PYTHONPATH=bridge:../E22/src` for adapter imports.
The artifact hashes are recorded in `../E22/migration/paper_catalog.jsonl`.

Read-only preflight from this S4M root:

```bash
PYTHONPATH=bridge:../E22/src /home/xyw/miniforge3/envs/malr-s4m/bin/python -c 'from pathlib import Path; from integrations.forecast_compare.protocol import load_p4_protocol; p=load_p4_protocol(Path("../E22/configs/stage1_seq96_dynamic_temporal_repair.yaml"), Path("../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context")); print(p.source_sha256, p.mask_sha256)'
```

`data -> ../data` supplies the shared input path. The S4M verification checks
that `model/S4M.py`, `model/Bank.py` and the S4 core retain their upstream Git
blobs, the reviewed patch is installed, and the bridge accepts the pinned
upstream commit as an ancestor. This command does not start S4M training or
evaluation. For the full read-only bridge preflight, use
`preflight_s4m_p4` from `bridge/integrations/forecast_compare/s4m_p4.py` with
the E22 config and P4 mask artifact root.
