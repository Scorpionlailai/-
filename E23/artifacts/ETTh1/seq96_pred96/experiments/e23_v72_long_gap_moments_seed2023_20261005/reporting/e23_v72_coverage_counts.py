from pathlib import Path
import json,os
import numpy as np
from integrations.e23.config import load_e23_p4_protocol
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog

ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v72_long_gap_moments_seed2023_20261005')
plan=json.loads((ROOT/'plan.json').read_text());work=Path(plan['work'])
rows=[]
for scene in plan['scenarios']:
    os.environ['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
    p=load_e23_p4_protocol(catalog=MigrationCatalog.load(work.parent/'E22/migration/paper_catalog.jsonl',project='E23'),
        protocol_config=Path(plan['protocol_config']),mask_artifacts=work.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=2023)
    assert p.mask_sha256['validation']==scene['validation_mask_sha256']
    damaged=full=windows=no_observed_windows=0
    for batch in iter_p4_batches(p,'validation',batch_size=128):
        mask=batch.observed_mask.numpy();windows+=mask.shape[0]
        no_observed_windows+=int((~mask.reshape(len(mask),-1).any(-1)).sum())
        for start in range(0,96,8):
            visible=mask[:,start:min(start+16,96)]
            damaged+=int((~visible.all(1)).sum())
            full+=int((~visible.any(1)).sum())
    rows.append(dict(scene=scene['name'],validation_mask_sha256=p.mask_sha256['validation'],windows=windows,
                     damaged_patches=damaged,fully_hidden_patches=full,windows_without_observation=no_observed_windows))
out=ROOT/'validation-coverage-counts.json'
with out.open('x') as f:json.dump(rows,f,indent=2);f.write('\n')
print(json.dumps(rows))
