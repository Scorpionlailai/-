"""Fixed-validation Stage1 weight averages; no optimization, future, or test."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import copy,json,os,sys

BASE=Path(__file__).resolve().parent
ROOT=BASE/'v121_v122_checkpoint_soup_validation'
RUNS=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8')
W=Path('/home/xyw/.cache/codex-worktrees/e23-v115-dense-width256-20261009')
sys.path[:0]=[str(W),str(W/'src'),str(RUNS/'e23_v115_weather_dense256_20261009')]
from weather_common import protocol_for
import torch
from types import SimpleNamespace
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection,build_repair_model,_new_memory,_file_sha256
from integrations.e23.repair_validation_cache import FixedRepairValidationCache,explicit_to
from stage1_iteration import repair_totals
from train_refinement import coarse_hash,bank_content

PAIRS=[('E23-v121',('e23_v115_weather_dense256_20261009','E23-v115','v115_weather.yaml','e23-stage1.json'),('e23_v115_weather_dense256_20261009','E23-v115','v115_weather.yaml','e23-stage1-mae.json')),
       ('E23-v122',('e23_v107_weather_context_gate_20261009','E23-v107','v107_weather.yaml','e23-stage1-mae.json'),('e23_v108_weather_mae_repair_20261009','E23-v108','v108_weather.yaml','e23-stage1-mae.json'))]
ALPHAS=(0.,.05,.1,.25,.5,.75,.9,.95,1.)


def status(state,**extra):
    record=dict(status=state,updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**extra)
    (ROOT/'status.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


@torch.no_grad()
def score(model,cache,device):
    model.eval();sums={k:0 for k in ('repair96_squared','repair96_absolute','repair32_squared','repair32_absolute','hidden96_coordinates','hidden32_coordinates')}
    for explicit,latents,points,mean,std,target,mask in cache.rows:
        result,_=model.recent_mean_projector(explicit_to(explicit,device),SimpleNamespace(latents=latents.to(device)),points.to(device),mean.to(device),std.to(device))
        assert torch.equal(result.normalized_sequence[mask.to(device)],explicit.observed_sequence.to(device)[mask.to(device)])
        parts=repair_totals(result.normalized_sequence,target.to(device),mask.to(device),result.sequence_validity,std.to(device))
        for k in sums:sums[k]+=parts[k]
    return {f'global{scope}_{metric}':sums[f'repair{length}_{numerator}']/sums[f'hidden{length}_coordinates']
            for length,scope in ((96,''),(32,'_last32')) for metric,numerator in (('mse','squared'),('mae','absolute'))}


def main():
    torch.set_num_threads(4);started=perf_counter();status('running')
    assert not os.popen('nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader').read().strip()
    device=torch.device('cuda:0');results=[]
    for label,left,right in PAIRS:
        plan=json.loads((RUNS/right[0]/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0])
        payloads=[];paths=[];settings=[]
        for definition in (left,right):
            root,name,config,selected=definition
            path=RUNS/root/name/'sr010_bl8/stage1/runs'/selected
            setting=load_e23_settings(RUNS/root/config,protocol=protocol,allow_variable_scaling=True)
            payloads.append(load_stage1_selection(path,protocol,setting,device=torch.device('cpu')));paths.append(path);settings.append(setting)
        a,b=payloads;names=set(a['model_state']);assert names==set(b['model_state'])
        prefix='recent_mean_projector.'
        for k in names:
            if not k.startswith(prefix):assert torch.equal(a['model_state'][k],b['model_state'][k]),k
        model=build_repair_model(protocol,settings[1],device=device)
        model.load_state_dict(b['model_state'],strict=True);model.historical.retrieval.load_state_dict(b['retrieval_state'])
        memory=_new_memory(protocol);memory.load_state_dict(b['memory_state']);model.historical.refresh_snapshot(memory.snapshot())
        old_memory=_new_memory(protocol);old_memory.load_state_dict(a['memory_state']);assert bank_content(memory)==bank_content(old_memory)
        model.eval().requires_grad_(False);model.recent_mean_projector.requires_grad_(True)
        cache=FixedRepairValidationCache(model,protocol,settings[1],coarse_hash);initial=cache.build(device=device)
        assert abs(initial['global_mae']-b['validation_repair_metrics']['global_mae'])<1e-6
        anchor=coarse_hash(model);rows=[]
        for alpha in ALPHAS:
            state={k:(a['model_state'][k]*(1-alpha)+b['model_state'][k]*alpha if k.startswith(prefix) else b['model_state'][k]) for k in names}
            model.load_state_dict(state,strict=True);assert coarse_hash(model)==anchor
            metrics=score(model,cache,device);rows.append(dict(alpha_later=alpha,**metrics));status('running',model=label,alpha_later=alpha,**metrics)
            if alpha in (0.,1.):
                parent=a if alpha==0 else b
                assert abs(metrics['global_mse']-parent['validation_repair_metrics']['global_mse'])<1e-6
                assert abs(metrics['global_mae']-parent['validation_repair_metrics']['global_mae'])<1e-6
        best_parent=min([rows[0],rows[-1]],key=lambda x:x['global_mae'])
        eligible=[r for r in rows if r['global_mse']<=best_parent['global_mse']+1e-8]
        chosen=min(eligible,key=lambda r:(r['global_mae'],r['global_mse']))
        improved=chosen['global_mae']<best_parent['global_mae']-1e-6 and chosen['global_mse']<=best_parent['global_mse']
        results.append(dict(model=label,source_selection_paths=[str(p) for p in paths],source_selection_sha256=[_file_sha256(p) for p in paths],source_epochs=[p['epoch'] for p in payloads],rows=rows,selected=chosen,best_parent=best_parent,improved_both=improved,zero_new_optimizer_updates=True,bank_and_experts_unchanged=True,settings_path=str(RUNS/right[0]/right[2])))
        (ROOT/'results.json').write_text(json.dumps(dict(scope='fixed validation only; no test or optimization; alpha not chosen from test',candidates=results,seconds=perf_counter()-started),indent=2))
        del model,cache,payloads,a,b,memory,old_memory;torch.cuda.empty_cache()
    status('completed',exit_code=0,seconds=perf_counter()-started)


if __name__=='__main__':
    try:main()
    except Exception as error:status('failed',exit_code=1,error=repr(error));raise
