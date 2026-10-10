"""Evaluate a validation-selected E24 model on its sealed 20-mask bank."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from readonly_weather_data import load_dataset_protocol, read_manifest, with_test_mask
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import (_atomic_json, _starts, _new_memory, _key_sha256,
    build_forecast_model, load_stage1_selection, load_stage2_selection)
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import selected_training_resources
from malr.evaluation.report import file_sha256


def summarize(rows):
    keys = ('forecast_mse','forecast_mae','repair96_mse','repair96_mae','repair32_mse','repair32_mae')
    result = {k:dict(mean=float(np.mean([r[k] for r in rows])),
                     mask_std_ddof1=float(np.std([r[k] for r in rows],ddof=1))) for k in keys}
    result['coordinate_weighted'] = {
        k:sum(r[k]*r[count] for r in rows)/sum(r[count] for r in rows)
        for k,count in ((k,'forecast_coordinates' if k.startswith('forecast')
                        else 'hidden96_coordinates' if k.startswith('repair96')
                        else 'hidden32_coordinates') for k in keys)}
    return result


@torch.no_grad()
def evaluate(manifest, settings_path, selection, output, device):
    started = perf_counter()
    manifest,selection,output = map(lambda p:Path(p).resolve(),(manifest,selection,output))
    if output.exists():
        raise FileExistsError(output)
    record = read_manifest(manifest)
    protocol = load_dataset_protocol(manifest,config_path=Path(record['config_path']),artifacts=manifest.parent)
    settings = load_e23_settings(Path(settings_path),protocol=protocol,allow_variable_scaling=True)
    selected = json.loads(selection.read_text())
    payload = load_stage2_selection(selection,protocol,settings,device=device)
    stage1 = load_stage1_selection(Path(selected['stage1_manifest']),protocol,settings,device=device)
    model = build_forecast_model(protocol,settings,device=device)
    model.load_state_dict(payload['model_state'],strict=True)
    for key,value in stage1['model_state'].items():
        if not torch.equal(value,payload['model_state']['repair.'+key]):
            raise RuntimeError('Stage2 repair differs from selected Stage1: '+key)
    model.repair.requires_grad_(False)
    model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(protocol);memory.load_state_dict(payload['memory_state'])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    model.eval()
    before = parameter_hash(model.repair),memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict())
    del stage1,payload
    validation = None
    rows = []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    inference_started = perf_counter()
    draws = [x for x in record['masks'] if x['split']=='test']
    for item in [next(x for x in record['masks'] if x['split']=='validation'),*draws]:
        split = item['split']
        current = protocol if split=='validation' else with_test_mask(protocol,record,item)
        sums = {k:0. for k in ('forecast_squared','forecast_absolute','repair96_squared','repair96_absolute','repair32_squared','repair32_absolute')}
        counts = dict(forecast_coordinates=0,hidden96_coordinates=0,hidden32_coordinates=0,windows=0)
        for cpu_batch in iter_p4_batches(current,split,batch_size=settings.batch_size):
            batch = cpu_batch.to(device)
            starts = _starts(batch,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
            prediction,repaired = model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
            forecast_error = (prediction-batch.forecast_target).double()
            sums['forecast_squared'] += forecast_error.square().sum().item()
            sums['forecast_absolute'] += forecast_error.abs().sum().item()
            counts['forecast_coordinates'] += forecast_error.numel()
            error = ((repaired.explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:,None]).double()
            hidden = ~batch.observed_mask
            for length in (96,32):
                part = error[:,-length:][hidden[:,-length:]]
                sums[f'repair{length}_squared'] += part.square().sum().item()
                sums[f'repair{length}_absolute'] += part.abs().sum().item()
                counts[f'hidden{length}_coordinates'] += part.numel()
            assert torch.equal(repaired.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
            counts['windows'] += len(batch.refs)
        assert counts['windows']==len(current.refs[split])
        metrics = dict(seed=item['seed'],mask_sha256=item['sha256'],**counts,
                       window_missing_fraction=counts['hidden96_coordinates']/(counts['windows']*96*21))
        for prefix,count in (('forecast','forecast_coordinates'),('repair96','hidden96_coordinates'),('repair32','hidden32_coordinates')):
            metrics[prefix+'_mse'] = sums[prefix+'_squared']/counts[count]
            metrics[prefix+'_mae'] = sums[prefix+'_absolute']/counts[count]
        if split=='validation':
            validation=metrics
            assert abs(metrics['forecast_mse']-selected['best_val_mse'])<1e-6
            assert abs(metrics['forecast_mae']-selected['best_val_mae'])<1e-6
        else:
            rows.append(metrics)
        print(json.dumps(dict(split=split,selection=selected['selection_metric'],**metrics)),flush=True)
    torch.cuda.synchronize(device)
    inference_seconds = perf_counter()-inference_started
    after = parameter_hash(model.repair),memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict())
    assert before == after
    resources = selected_training_resources([Path(selected['stage1_manifest']),selection])
    assert resources['resource_status']=='measured_all_stages'
    output.mkdir(parents=True)
    result = dict(protocol_id=record['protocol_id'],dataset='Weather',model='E24-v79',
                  split='test',metric_space='train_standardized',training_seed=2023,
                  start_rate=.10,block_len=8,seq_len=96,pred_len=96,n_vars=21,
                  selection_metric=selected['selection_metric'],primary=selected['selection_metric']=='mae',
                  best_epoch=selected['best_epoch'],selection_manifest=str(selection),
                  selection_manifest_sha256=file_sha256(selection),checkpoint_sha256=selected['checkpoint_sha256'],
                  stage1_manifest=selected['stage1_manifest'],stage1_checkpoint_sha256=selected['stage1_checkpoint_sha256'],
                  dataset_manifest=str(manifest),dataset_manifest_sha256=file_sha256(manifest),
                  validation=validation,draws=rows,summary=summarize(rows),
                  training_resources=resources,repair_frozen_verified=True,visible_values_exact=True,
                  test_inference_seconds=inference_seconds,evaluation_full_seconds=perf_counter()-started,
                  inference_peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/2**30,
                  inference_peak_reserved_GiB=torch.cuda.max_memory_reserved(device)/2**30,
                  aggregation='equal weight per mask; weighted-by-coordinates separately reported',
                  training_seed_std=None,drr=None,
                  drr_unavailable_reason='No matching Weather Clean/Missing runs yet.')
    _atomic_json(result,output/'twenty-test-draws.json')
    with (output/'per_mask.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(dataset='Weather',model='E24-v79',start_rate=.10,block_len=8,
                 training_seed=2023,selection_metric=selected['selection_metric'],best_epoch=selected['best_epoch'],test_masks=len(rows))
    for key,value in result['summary'].items():
        if key!='coordinate_weighted':
            summary[key]=value['mean'];summary[key+'_mask_std']=value['mask_std_ddof1']
    summary.update({k:resources[k] for k in ('full_workflow_seconds','peak_allocated_gib','peak_reserved_gib','gpu','resource_status')})
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary));writer.writeheader();writer.writerow(summary)
    print(json.dumps(dict(status='completed',output=str(output),summary=summary)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-manifest',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--e24-config',type=Path,default=Path(__file__).resolve().parents[2]/'configs/e24_v79.yaml')
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.dataset_manifest,args.e24_config,args.selection,args.output,torch.device(args.device))


if __name__=='__main__': main()
