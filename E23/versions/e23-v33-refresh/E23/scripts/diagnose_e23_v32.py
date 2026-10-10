"""Read-only repair and frozen auxiliary validation diagnostics for v32."""
import argparse,json
from pathlib import Path
from time import perf_counter
import torch
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.report_e23_v32_trial import CONFIG,digest
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection,load_stage2_selection,build_repair_model,build_forecast_model,_new_memory,_key_sha256,_starts
from integrations.e23.prediction_preservation import build_probe,freeze_probe,predict_sequence
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--cell',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();cell=a.cell
    if (cell/'repair_diagnostic.json').exists():raise FileExistsError('v32 diagnostic already exists')
    torch.set_num_threads(4);protocol=load_protocol();settings=load_e23_settings(CONFIG,protocol=protocol);device=torch.device(a.device)
    s1path=cell/'stage1/runs/e23-stage1.json';s1=json.loads(s1path.read_text());probe=None
    if s1.get('stage1_preservation')=='clean_forecast_v1':
        if digest(s1['preservation_probe_checkpoint'])!=s1['preservation_probe_checkpoint_sha256']:raise ValueError('auxiliary checksum mismatch')
        state=torch.load(s1['preservation_probe_checkpoint'],map_location=device,weights_only=False)
        probe=build_probe(protocol,device=device,seed=2053);probe.load_state_dict(state['model_state']);freeze_probe(probe)
        if parameter_hash(probe)!=s1['preservation_probe_final_sha256']:raise ValueError('auxiliary state mismatch')
    results=[]
    for stage in ('stage1','stage2'):
        started=perf_counter();selection=cell/stage/'runs'/('e23-'+stage+'.json');record=json.loads(selection.read_text())
        if stage=='stage1':
            payload=load_stage1_selection(selection,protocol,settings,device=device);model=build_repair_model(protocol,settings,device=device);repair=model;predictor_parameters=0
        else:
            payload=load_stage2_selection(selection,protocol,settings,device=device);model=build_forecast_model(protocol,settings,device=device);repair=model.repair;predictor_parameters=sum(p.numel() for p in model.predictor.parameters())
        model.load_state_dict(payload['model_state']);model.eval();repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot())
        before=(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash());sq=ab=count=windows=targets=0;clean_sq=clean_ab=repaired_sq=repaired_ab=consistency=0.
        with torch.inference_mode():
            for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
                b=cpu.to(device);starts=_starts(b,stride=8,patches=12,device=device);out=repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
                error=((out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None,:])[~b.observed_mask]
                sq+=error.square().double().sum().item();ab+=error.abs().double().sum().item();count+=error.numel();windows+=len(b.normalized_observed)
                if probe is not None:
                    clean=predict_sequence(probe,b.normalized_target,b);repaired=predict_sequence(probe,out.explicit.normalized_sequence,b)
                    clean_error=(clean-b.forecast_target).double();repair_error=(repaired-b.forecast_target).double()
                    clean_sq+=clean_error.square().sum().item();clean_ab+=clean_error.abs().sum().item();repaired_sq+=repair_error.square().sum().item();repaired_ab+=repair_error.abs().sum().item();consistency+=(clean-repaired).double().square().sum().item();targets+=clean.numel()
        if before!=(_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash()):raise ValueError('diagnostic changed historical state')
        result=dict(stage=stage,checkpoint_sha256=record['checkpoint_sha256'],selection_sha256=digest(selection),e23_config_sha256=record['e23_config_sha256'],source_csv_sha256=protocol.source_sha256,validation_mask_sha256=protocol.mask_sha256['validation'],repair_parameters=sum(p.numel() for p in repair.parameters()),predictor_parameters=predictor_parameters,diagnostic_seconds=perf_counter()-started,global_mse=sq/count,global_mae=ab/count,hidden_coordinates=count,windows=windows,split='original_fixed_validation',metric_space='train_standardized',h_state_unchanged=True)
        if probe is not None:
            if parameter_hash(probe)!=s1['preservation_probe_final_sha256']:raise ValueError('diagnostic changed auxiliary')
            result.update(auxiliary_state_sha256=parameter_hash(probe),auxiliary_target_elements=targets,auxiliary_clean_mse=clean_sq/targets,auxiliary_clean_mae=clean_ab/targets,auxiliary_repaired_mse=repaired_sq/targets,auxiliary_repaired_mae=repaired_ab/targets,auxiliary_consistency_mse=consistency/targets)
        results.append(result)
        if stage=='stage2':(cell/'repair_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
        del model,repair
    (cell/'information_diagnostic.json').write_text(json.dumps(dict(stages=results),indent=2)+'\n')


if __name__=='__main__':main()
