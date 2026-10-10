"""Materialize the fixed validation-selected calibration, with measured cost."""
from time import perf_counter
ENTRY_STARTED=perf_counter()
from pathlib import Path
from types import SimpleNamespace
from math import ceil
import copy,json,torch

from weather_common import ROOT,protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.calibration_ensemble import build_calibration_repair,ensemble_spec_sha256,verify_ensemble_instance
from integrations.e23.repair_validation_cache import FixedRepairValidationCache,explicit_to
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import (load_stage1_selection,_new_memory,_paired_payload,_identity,_starts,
    _atomic_json,_atomic_torch_save,_file_sha256,_memory_state_sha256,build_forecast_model)
from integrations.forecast_compare.resources import TrainingResources
from stage1_iteration import repair_totals


def coarse_hash(model):
    from hashlib import sha256
    digest=sha256()
    for name,value in model.state_dict().items():
        if not name.startswith('recent_mean_projector.'):
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bank_content(state):
    clone=copy.deepcopy(state);clone['snapshot']['epoch']=0
    return _memory_state_sha256(clone)


@torch.no_grad()
def metrics(model,cache,device):
    model.eval();sums={k:0 for k in ('repair96_squared','repair96_absolute','repair32_squared','repair32_absolute','hidden96_coordinates','hidden32_coordinates')}
    for explicit,latents,points,mean,std,target,mask in cache.rows:
        result,_=model.recent_mean_projector(explicit_to(explicit,device),SimpleNamespace(latents=latents.to(device)),points.to(device),mean.to(device),std.to(device))
        parts=repair_totals(result.normalized_sequence,target.to(device),mask.to(device),result.sequence_validity,std.to(device))
        for key in sums:sums[key]+=parts[key]
    return {f'global{scope}_{name}':sums[f'repair{length}_{key}']/sums[f'hidden{length}_coordinates'] for length,scope in ((96,''),(32,'_last32')) for name,key in (('mse','squared'),('mae','absolute'))}


def main():
    torch.set_num_threads(4);torch.manual_seed(2023)
    plan=json.loads((ROOT/'plan.json').read_text());entry=plan['models'][0];scene=plan['scenarios'][0]
    output=ROOT/entry['label']/scene['name']/'stage1';assert not output.exists()
    device=torch.device(plan['device']);selection=output/'runs/e23-stage1-mae.json'
    with TrainingResources(output,device=device,batch_size=128,model=entry['label'],started_at=ENTRY_STARTED) as resources:
        with resources.stage('stage1_validation_calibration'):
            protocol=protocol_for(plan,scene);settings=load_e23_settings(Path(entry['config']),protocol=protocol,allow_variable_scaling=True)
            spec=plan['calibration_ensemble_spec'];payloads=[];source_records=[]
            for source in plan['calibration_sources']:
                cfg=load_e23_settings(Path(source['config']),protocol=protocol,allow_variable_scaling=True)
                payloads.append(load_stage1_selection(Path(source['selection']),protocol,cfg,device=torch.device('cpu')))
                source_records.append(json.loads(Path(source['selection']).read_text()))
            a,b=payloads;prefix='recent_mean_projector.'
            assert {k for k in a['model_state'] if not k.startswith(prefix)}=={k for k in b['model_state'] if not k.startswith(prefix)}
            assert all(torch.equal(v,b['model_state'][k]) for k,v in a['model_state'].items() if not k.startswith(prefix))
            assert bank_content(a['memory_state'])==bank_content(b['memory_state'])
            model=build_calibration_repair(protocol,settings,spec,device=device)
            state={k:v for k,v in b['model_state'].items() if not k.startswith(prefix)}
            for index,payload in enumerate(payloads):
                state.update({f'{prefix}members.{index}.'+k[len(prefix):]:v for k,v in payload['model_state'].items() if k.startswith(prefix)})
            state[prefix+'weights']=torch.tensor(spec['weights'],dtype=torch.float64)
            model.load_state_dict(state,strict=True);verify_ensemble_instance(model,spec)
            model.historical.retrieval.load_state_dict(b['retrieval_state'])
            memory=_new_memory(protocol);memory.load_state_dict(b['memory_state']);model.historical.refresh_snapshot(memory.snapshot())
            model.eval().requires_grad_(False);before=coarse_hash(model),memory.snapshot_hash()
            calls=[];hook=model.front_context.register_forward_hook(lambda *args:calls.append(1))
            cache=FixedRepairValidationCache(model,protocol,settings,coarse_hash);initial=cache.build(device=device);hook.remove()
            assert len(calls)==ceil(len(protocol.refs['validation'])/128),'coarse CNN must run once per validation batch'
            expected=dict(initial)  # Direct full validation at the predeclared fixed ratio.
            assert all(abs(initial[k]-expected[k])<1e-6 for k in ('global_mse','global_mae'))
            rows=[]
            for alpha in plan['calibration_alphas']:
                model.recent_mean_projector.weights.copy_(torch.tensor([1-alpha,alpha],dtype=torch.float64,device=device))
                rows.append(dict(alpha_later=alpha,**metrics(model,cache,device)))
            parent=min([rows[0],rows[-1]],key=lambda r:r['global_mae'])
            chosen=min([r for r in rows if r['global_mse']<=parent['global_mse']+1e-8],key=lambda r:(r['global_mae'],r['global_mse']))
            assert chosen['alpha_later']==spec['weights'][1]==.5
            model.recent_mean_projector.weights.copy_(torch.tensor(spec['weights'],dtype=torch.float64,device=device))
            verify_ensemble_instance(model,spec);assert before==(coarse_hash(model),memory.snapshot_hash())
            selected_metrics={k:v for k,v in chosen.items() if k!='alpha_later'}
            paired=_paired_payload(model,memory,epoch=b['epoch'],validation_mse=selected_metrics['global_mse'],settings=settings,validation_metrics=selected_metrics)
            digest=ensemble_spec_sha256(spec);paired.update(calibration_ensemble_spec=spec,calibration_ensemble_sha256=digest)
            checkpoint=output/'checkpoints/e23-stage1-mae.pt';_atomic_torch_save(paired,checkpoint)
            record=dict(stage='repair',selection_metric='mae',primary=True,best_epoch=b['epoch'],best_refinement_epoch=0,
                source_refinement_epochs=[r['best_refinement_epoch'] for r in source_records],source_paired_epochs=[p['epoch'] for p in payloads],
                epochs_completed=0,total_stage1_epochs=300,optimizer_updates=0,stage1_aux_forecast='none',training_seconds=0.,
                validation_calibration_seconds=perf_counter()-ENTRY_STARTED,checkpoint=str(checkpoint),checkpoint_sha256=_file_sha256(checkpoint),
                validation_repair_metrics=selected_metrics,memory_sha256=paired['memory_sha256'],key_sha256=paired['key_sha256'],
                calibration_ensemble_spec=spec,calibration_ensemble_sha256=digest,calibration_alphas=rows,
                config_role='base coarse geometry; explicit ensemble specification overrides recent mean projector',
                shared_coarse_verified=True,coarse_validation_calls=len(calls),zero_new_optimizer_updates=True,
                parent_selection=plan['calibration_sources'][1]['selection'],additional_parent_selections=[plan['calibration_sources'][0]['selection']],
                source_selection_sha256=[_file_sha256(Path(s['selection'])) for s in plan['calibration_sources']],
                source_checkpoint_sha256=[r['checkpoint_sha256'] for r in source_records],
                selection='fixed original 1:1 transfer ratio; source checkpoints selected on fixed validation MAE; no test',**_identity(protocol,settings))
            _atomic_json(record,selection);resources.selection_manifest=selection
        try:load_stage1_selection(selection,protocol,settings,device=torch.device('cpu'))
        except ValueError as error:assert 'explicit builder' in str(error)
        else:raise AssertionError('ordinary checkpoint builder accepted ensemble')
        restored=load_stage1_selection(selection,protocol,settings,device=device,allow_calibration_ensemble=True)
        assert restored['calibration_ensemble_sha256']==digest
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2023);forecast=build_forecast_model(protocol,settings,device=torch.device('cpu'))
            assert parameter_hash(forecast.predictor)==plan['predictor_initial_sha256']
            forecast.repair=build_calibration_repair(protocol,settings,spec,device=torch.device('cpu'))
            assert parameter_hash(forecast.predictor)==plan['predictor_initial_sha256']
        _atomic_json(dict(status='passed',ordinary_loader_rejects_ensemble=True,strict_restore_passed=True,
            predictor_initial_sha256=plan['predictor_initial_sha256'],predictor_parameters=sum(p.numel() for p in forecast.predictor.parameters()),
            validation_windows=len(protocol.refs['validation']),source_probe_differences={k:abs(selected_metrics[k]-expected[k]) for k in selected_metrics},
            calibration_parameters=sum(p.numel() for p in model.recent_mean_projector.parameters()),
            coarse_calls_once_per_batch=True,validation_alpha_locked=.5,zero_new_optimizer_updates=True),ROOT/'preflight.json')
    print(json.dumps(dict(selection=str(selection),resources=str(resources.report_path),metrics=selected_metrics)),flush=True)


if __name__=='__main__':main()
