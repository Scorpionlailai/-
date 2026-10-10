import json,os
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,_starts
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.author_profiles import build_author_model,load_author_profiles
from integrations.forecast_compare.t1_p4 import T1ForecastModel
from malr.migration import MigrationCatalog
from scripts.run_author_aligned_p4 import _forecast_head,_seed
from scripts.prepare_highmissing import ROOT,WORK,MASKS,write

def main():
    torch.set_num_threads(4);device=torch.device('cuda:1');plan=json.loads((ROOT/'plan.json').read_text());rows=[]
    profiles=load_author_profiles(Path(plan['author_profiles']));cfg=profiles['models']['T1']
    for spec in plan['scenarios']:
        os.environ['MALR_MISSING_SCENARIO']=spec['scenario_manifest']
        p=load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts=MASKS,training_seed=502)
        s=load_e23_settings(Path(plan['e23_config']),protocol=p)
        torch.manual_seed(502);e=build_forecast_model(p,s,device=device)
        assert parameter_hash(e.predictor)==plan['predictor_initial_sha256'] and sum(x.numel() for x in e.repair.parameters())==174362
        _seed(502);t=build_author_model('T1',cfg,WORK/cfg['root'],device);_seed(502);t=T1ForecastModel(t,_forecast_head(p)).to(device)
        assert parameter_hash(t.predictor)==plan['predictor_initial_sha256']
        b=next(iter_p4_batches(p,'validation',batch_size=4)).to(device)
        for label,m in [('E23-v41-slow',e),('T1',t)]:
            initial_hash=parameter_hash(m.predictor)
            m.train();m.zero_grad(set_to_none=True)
            if label=='T1':prediction=m(b)
            else:prediction,_=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,_starts(b,stride=8,patches=12,device=device))
            loss=(prediction-b.forecast_target).square().mean();loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
            rows.append(dict(scenario=spec['name'],model=label,finite_loss=float(loss.detach()),predictor_initial_sha256=initial_hash,predictor_parameters=sum(v.numel() for v in m.predictor.parameters())))
        del e,t
    write(ROOT/'preflight.json',dict(status='passed',cells=rows,no_training_runs=True,no_test_performance_seen=True))
    print(json.dumps(rows))
if __name__=='__main__':main()
