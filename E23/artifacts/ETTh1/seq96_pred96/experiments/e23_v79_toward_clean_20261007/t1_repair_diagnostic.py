"""Compare frozen official T1 and v79 hidden-history repair on identical validation."""
import importlib.util
import json
from pathlib import Path
import torch
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.t1_p4 import t1_inputs,t1_clean_history

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007')

def main():
    plan=json.loads((BASE/'t1-plan.json').read_text());torch.set_num_threads(4)
    module_path=Path(plan['work'])/'scripts/run_t1_frozen_stage2.py'
    spec=importlib.util.spec_from_file_location('locked_t1',module_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    refs={r['scene']:r for r in json.loads((ROOT/'validation-diagnostics.json').read_text())};rows=[]
    for scene in plan['scenarios'][1:]:
        protocol,_,parent,model=m.setup(plan,scene);before=m.parameter_hash(model.imputer)
        model.eval();totals={k:[0.,0.,0] for k in ['all_hidden','last32_hidden']}
        with torch.inference_mode():
            for cpu in iter_p4_batches(protocol,'validation',batch_size=128):
                b=cpu.to(plan['device']);values,mask=t1_inputs(b);candidate=model.imputer(values,mask=mask).float()
                error=candidate-t1_clean_history(b);hidden=~b.observed_mask
                for label,e,h in [('all_hidden',error,hidden),('last32_hidden',error[:,-32:],hidden[:,-32:])]:
                    residual=e[h].double();assert torch.isfinite(residual).all()
                    totals[label][0]+=residual.square().sum().item();totals[label][1]+=residual.abs().sum().item();totals[label][2]+=residual.numel()
        m.assert_frozen(model,before)
        own=refs[scene['name']]
        t1={k:dict(mse=s/n,mae=a/n,hidden_coordinates=n) for k,(s,a,n) in totals.items()}
        assert all(t1[k]['hidden_coordinates']==own['repair_regions'][k]['hidden_coordinates'] for k in totals)
        row=dict(scene=scene['name'],split='fixed_validation',metric_space='train_standardized',
            t1_stage1_checkpoint_sha256=parent['checkpoint_sha256'],t1_precision='fp32 frozen inference',
            t1=t1,v79=own['repair_regions'],repair_state_unchanged=True)
        rows.append(row);print(json.dumps(row),flush=True)
        del model;torch.cuda.empty_cache()
    with (ROOT/'t1-v79-repair-comparison.json').open('x') as f:json.dump(rows,f,indent=2);f.write('\n')

if __name__=='__main__':main()
