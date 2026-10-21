"""Small real-PAMAP2 adapter/gradient check; no GPU performance or optimizer updates."""
from pathlib import Path
import json
import torch
from run_acceptance import build_model,capture,shared_output,cpu_clone
from integrations.v125_ablation.common import data_for,STUDY,seed_all,rng_state
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero
from acceptance_utils import compare_trees,stream_call

def main():
    torch.set_num_threads(4);out=Path('/home/xyw/v125-shared-prefix-acceptance-20261021')
    a=json.load(open('/home/xyw/paper/E24/artifacts/v125_calibration_confirmation_20261020/source_audit.json'));d=data_for('PAMAP2','sr015_bl8',STUDY)
    m=build_model(d,torch.device('cpu'),a['sources']['A0_2023']);truth,obs,starts=next(d.batches('train',8,epoch=0,seed=2023));valid=torch.from_numpy(d.valid_for(starts));batch=(truth,obs,starts,valid)
    context=capture(m,torch.where(obs,truth,0.),obs,starts);checks=[]
    for i in (0,1):
        m.inference_path='calibration'+str((64,256)[i]);params=tuple(m.core.recent_mean_projector.members[i].parameters())
        seed_all(2023);rs=rng_state();ss=cpu_clone(rs)
        def forward(shared):
            pred=shared_output(m,i,batch,context) if shared else m(torch.where(obs,truth,0.),obs,starts)
            grads=torch.autograd.grad(mean_or_zero(hidden_error(pred,truth,obs,valid).abs()),params,allow_unused=True)
            return pred.detach(),tuple(None if g is None else g.detach() for g in grads)
        ref=stream_call(rs,lambda:forward(False));candidate=stream_call(ss,lambda:forward(True))
        c=compare_trees((ref,rs),(candidate,ss));assert c['exact_equal'],c;checks.append(dict(branch=(64,256)[i],**c))
    (out/'cpu_coordinate_preflight.json').write_text(json.dumps(dict(passed=True,batch_size=8,CPU_only=True,optimizer_updates=0,checks=checks,
        pilot_probe_note='Initial one-off CPU probe omitted resetting dropout RNG and correctly showed output-head gradient differences; fixed probe uses the full independent stream contexts already required in the runner.'),indent=2)+'\n')
    print('CPU actual PAMAP2 coordinate/gradient/RNG preflight passed for2branches;not GPU performance acceptance')
if __name__=='__main__':main()
