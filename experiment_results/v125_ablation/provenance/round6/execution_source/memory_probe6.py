"""Finite discarded preflight: monitoring must preserve updates and release batch graphs."""
from contextlib import nullcontext
import gc,json,time
import torch
from policy6 import OUT
from quality6 import monitor_quality
from integrations.v125_ablation.common import data_for,STUDY,seed_all,state_sha,write_json,resources
from integrations.v125_ablation.adapter import V125Adapter
from integrations.v125_ablation.run import configure_phase

def trial(monitored):
    seed_all(2024);data=data_for('PAMAP2','sr015_bl8',STUDY);device=torch.device('cuda:0')
    m=V125Adapter(data,'A5',2024).to(device);params=configure_phase(m,'coarse')
    optimizer=torch.optim.Adam(params,lr=.001,weight_decay=0.)
    m.train();m.epoch=1;m.core.historical.retrieval.begin_epoch(1)
    rows=[];begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with monitor_quality(m) if monitored else nullcontext():
        for index,(truth,obs,starts) in enumerate(data.batches('train',128,epoch=0,seed=2023),1):
            if index>12:break
            m.batch_index=index;truth,obs=truth.to(device),obs.to(device)
            optimizer.zero_grad(set_to_none=True);loss=m.loss(truth,obs,starts);value=float(loss.detach())
            loss.backward();optimizer.step();m.after_step();del loss,truth,obs
            torch.cuda.synchronize();rows.append(dict(batch=index,loss=value,allocated_bytes=torch.cuda.memory_allocated()))
            print(json.dumps(dict(monitored=monitored,**rows[-1])),flush=True)
    result=dict(monitored=monitored,batches=rows,state_sha256=state_sha(m),seconds=time.perf_counter()-begin,resources=resources(device))
    del optimizer,params,m,data;gc.collect();torch.cuda.empty_cache();return result

def main():
    torch.set_num_threads(4)
    plain=trial(False);repeat=trial(False);monitored=trial(True)
    loss_difference=max(abs(a['loss']-b['loss']) for a,b in zip(plain['batches'],monitored['batches']))
    unmonitored_repeat_difference=max(abs(a['loss']-b['loss']) for a,b in zip(plain['batches'],repeat['batches']))
    assert loss_difference<2e-6
    allocated=[x['allocated_bytes'] for x in monitored['batches'][-5:]]
    assert max(allocated)-min(allocated)<128*1024**2
    write_json(OUT/'memory_probe.json',dict(passed=True,control=plain,repeat_control=repeat,monitored=monitored,
        maximum_training_loss_difference=loss_difference,unmonitored_repeat_maximum_loss_difference=unmonitored_repeat_difference,
        all12_training_losses_within_existing_2e6_tolerance=True,byte_identical_final_state=plain['state_sha256']==monitored['state_sha256'],
        determinism_note='Original seed_all seeds RNGs but does not enforce deterministic CUDA kernels; byteidentity is not a protocol promise. No formal determinism settings changed.',discarded_updates=36,no_formal_epoch_or_checkpoint_used=True,
        root_cause='unittest.mock side_effect call history retained old expert/cache/state tensor arguments; direct new=function replacement has no history',
        fix='monitor wrappers changed to direct call-through functions; original model/batch/optimizer/trainingbudget unchanged'))

if __name__=='__main__':main()
