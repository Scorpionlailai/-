"""Post-training repair diagnosis on the fixed validation mask, no new test runs."""
import json
from pathlib import Path
import torch

from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_forecast_model,load_stage2_selection,_new_memory,_key_sha256,_validation_repair_metrics
from integrations.forecast_compare.protocol import load_p4_protocol,with_training_seed

ROOT=Path(__file__).resolve().parent
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
WORK=Path('/home/xyw/paper/.e23-v39-depth/E23')

def main():
    output=ROOT/'selected-stage2-repair-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(4);device=torch.device('cuda:1')
    protocol=with_training_seed(load_p4_protocol(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),502)
    candidates=[('E23-v31-B',FORMAL/'configs/e23_seed_502.yaml',FORMAL/'seed_502/E23-v31-B/stage2/runs/e23-stage2.json')]
    for version in ['v37','v38','v39']:
        candidates.append((version,ROOT/version/'e23_seed502.yaml',ROOT/version/'stage2/runs/e23-stage2.json'))
    records=[]
    for label,config,manifest in candidates:
        settings=load_e23_settings(config,protocol=protocol)
        payload=load_stage2_selection(manifest,protocol,settings,device=device)
        model=build_forecast_model(protocol,settings,device=device)
        model.load_state_dict(payload['model_state'],strict=True)
        model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state'])
        model.repair.historical.refresh_snapshot(memory.snapshot())
        identity=(memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict()))
        metrics=_validation_repair_metrics(model.repair,protocol,settings,device=device)
        assert identity==(memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict()))
        scales={k:float(v) for k,v in payload['model_state'].items()
                if k.endswith(('temporal_scale','variable_scale','later_step_logits')) and v.numel()==1}
        record=dict(model=label,split='fixed_validation',repair_metrics=metrics,learned_feature_scales=scales,historical_state_read_only=True)
        records.append(record);print(json.dumps(record),flush=True)
        del model,payload;torch.cuda.empty_cache()
    output.write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
