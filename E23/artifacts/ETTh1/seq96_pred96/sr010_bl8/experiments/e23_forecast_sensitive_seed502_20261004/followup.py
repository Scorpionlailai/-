"""Adaptive validation-only checks of joint regions and native T1 padding."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from malr.data.patching import patchify_preserving_values
from integrations.forecast_compare.protocol import iter_p4_batches, load_p4_protocol, with_training_seed
from integrations.forecast_compare.t1_p4 import t1_inputs
from integrations.e23.train import _starts
import diagnose as d


@torch.inference_mode()
def main():
    output = d.OUT / 'followup.json'
    if output.exists():
        raise FileExistsError(output)
    start = perf_counter()
    torch.set_num_threads(4)
    device = torch.device('cuda:1')
    protocol = with_training_seed(load_p4_protocol(d.WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml', d.MASKS), 502)
    models, memories, records, protected, _ = d.load_models(protocol, device)
    protected.update(d.protect_sources())
    protected[str(Path(__file__))] = d.sha(Path(__file__))
    protected[str(d.OUT/'followup-plan.json')] = d.sha(d.OUT/'followup-plan.json')
    before = d.state_identity(models, memories)
    torch.cuda.reset_peak_memory_stats(device)
    times = torch.arange(12, device=device)[:, None]*8+torch.arange(16, device=device)[None]
    selectors = {}
    for name, left, variable in [('LUFL_last16',80,4),('LUFL_last32',64,4),
                                 ('LUFL_all',0,4),('all_last16',80,None),
                                 ('all_last32',64,None),('all_hidden',0,None)]:
        selector = torch.zeros(7,12,16,device=device,dtype=torch.bool)
        if variable is None:
            selector[:] = (times >= left) & (times < 96)
        else:
            selector[variable] = (times >= left) & (times < 96)
        selectors[name] = selector
    # Named joint regions are evaluated together, never inferred by adding gains.
    cases = [(s, name, 'oracle', strength) for s in d.LABELS for name in selectors
             for strength in [1.,.5]]
    cases += [(s,name,'T1_donor',strength) for s in ['v31B','v38']
              for name in selectors for strength in [1.,.5]]
    sums = {case:np.zeros(3) for case in cases}
    bases = {s:np.zeros(3) for s in d.LABELS}
    gains = {case:[] for case in cases}
    t1_regions = {kind:[] for kind in ['oracle','oracle_half']}
    corrected_oracle = np.zeros(3)
    for index,cpu in enumerate(iter_p4_batches(protocol,'validation',batch_size=128)):
        b=cpu.to(device); B=len(b.normalized_observed)
        clean=patchify_preserving_values(b.normalized_target,b.observed_mask,16,8)
        hidden=~clean.effective_mask & clean.validity
        values={}
        for s,model in models.items():
            if s=='T1':
                observed,mask=t1_inputs(b)
                candidate=model.imputer(observed,mask=mask)
                seq=(torch.where(b.observed_mask,observed,candidate)-b.query_mean[:,None])/b.query_std[:,None]
                seq=torch.where(b.observed_mask,b.normalized_observed,seq)
                values[s]=patchify_preserving_values(seq,b.observed_mask,16,8).values
            else:
                x=model.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,
                               _starts(b,stride=8,patches=12,device=device)).explicit
                values[s]=x.repaired_patches
        base={}
        for s in d.LABELS:
            base[s]=models[s].predictor(values[s],clean.effective_mask,b.query_mean,b.query_std,patch_validity=clean.validity)
            d.add_metric(bases[s],base[s],b.forecast_target)
        for case in cases:
            s,name,kind,strength=case
            donor=values['T1'] if kind=='T1_donor' else clean.values
            selected=selectors[name][None]&hidden
            replacement=donor if strength==1 else values[s]+strength*(donor-values[s])
            x=torch.where(selected,replacement,values[s])
            pred=models[s].predictor(x,clean.effective_mask,b.query_mean,b.query_std,patch_validity=clean.validity)
            d.add_metric(sums[case],pred,b.forecast_target)
            gain=((base[s]-b.forecast_target).square().mean((1,2))-(pred-b.forecast_target).square().mean((1,2)))
            gains[case].append(gain.cpu().numpy())
        # T1's own regional responses with its original replicated padding fixed.
        t1base=(base['T1']-b.forecast_target).square().mean((1,2))
        for kind,strength in [('oracle',1.),('oracle_half',.5)]:
            replacement=clean.values if strength==1 else values['T1']+strength*(clean.values-values['T1'])
            chunks=[]
            for band in range(6):
                region=torch.zeros(7,7,12,16,device=device,dtype=torch.bool)
                for c in range(7):
                    region[c,c]=(times>=band*16)&(times<(band+1)*16)
                x=torch.where(region[:,None]&hidden[None],replacement[None],values['T1'][None])
                pred=models['T1'].predictor(x.reshape(7*B,7,12,16),clean.effective_mask.repeat(7,1,1,1),
                    b.query_mean.repeat(7,1),b.query_std.repeat(7,1),patch_validity=clean.validity.repeat(7,1,1,1)).reshape(7,B,7,96)
                chunks.append((t1base[None]-(pred-b.forecast_target[None]).square().mean((2,3))).T.cpu().numpy())
            t1_regions[kind].append(np.concatenate(chunks,1))
        # Actual clean numeric sequence also refreshes padding; stats/mask remain fixed.
        pred=models['T1'].predictor(clean.values,clean.effective_mask,b.query_mean,b.query_std,patch_validity=clean.validity)
        d.add_metric(corrected_oracle,pred,b.forecast_target)
    rows=[]; arrays={}
    for case,acc in sums.items():
        s,name,kind,strength=case; base=bases[s]; g=np.concatenate(gains[case])
        arrays['__'.join(map(str,case))]=g
        rows.append(dict(model=s,region=name,intervention=kind,strength=strength,padding='native_fixed',
                         base_mse=float(base[0]/base[2]),mse=float(acc[0]/acc[2]),
                         base_mae=float(base[1]/base[2]),mae=float(acc[1]/acc[2]),
                         mse_gain=float((base[0]-acc[0])/acc[2]),mae_gain=float((base[1]-acc[1])/acc[2]),
                         **{f'chronological_third_{i+1}_mse_gain':float(v.mean(dtype=np.float64))
                            for i,v in enumerate(np.array_split(g,3))}))
    t1_rows=[]
    for kind,parts in t1_regions.items():
        g=np.concatenate(parts);arrays['T1_native_'+kind]=g
        for i,(c,left,right) in enumerate(d.GROUPS):
            t1_rows.append(dict(model='T1',intervention=kind,padding='native_fixed',variable=d.VARIABLES[c],
                history_start=left,history_end_exclusive=right,mean_mse_gain=float(g[:,i].mean(dtype=np.float64))))
    assert before==d.state_identity(models,memories)
    assert all(d.sha(p)==v for p,v in protected.items())
    for s,acc in bases.items():
        expected=records[s]['best_validation']['mse'] if s=='T1' else records[s]['best_val_mse']
        assert abs(acc[0]/acc[2]-expected)<1e-6
    d.write_csv(d.OUT/'joint_region_interventions.csv',rows)
    d.write_csv(d.OUT/'T1_native_position_interventions.csv',t1_rows)
    np.savez_compressed(d.OUT/'followup_window_gains.npz',**arrays)
    d.write_json(output,dict(status='complete',adaptive_validation_followup=True,new_training=False,
        test_predictions_computed=False,parameters_and_files_unchanged=True,
        T1_full_oracle_refreshed_padding_mse=float(corrected_oracle[0]/corrected_oracle[2]),
        T1_full_oracle_refreshed_padding_mae=float(corrected_oracle[1]/corrected_oracle[2]),
        diagnostic_seconds=perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,
        output_sha256={p:d.sha(d.OUT/p) for p in ['joint_region_interventions.csv',
            'T1_native_position_interventions.csv','followup_window_gains.npz']},rows=len(rows)))
    print(json.dumps({'status':'complete','rows':len(rows),'seconds':perf_counter()-start}),flush=True)


if __name__=='__main__':
    main()
