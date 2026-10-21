"""A real full-batch calibration update verifies freeze and own-source inheritance."""
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from integrations.v125_ablation.adapter import V125Adapter
from integrations.v125_ablation.common import STUDY,data_for,verify_lock,seed_all,file_sha256,write_json
from policy import initialize_branch,frozen_hash,tensor_hash,calibration_loss
OUT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4);verify_lock(STUDY);audit=json.loads((OUT/'source_audit.json').read_text())
    data=data_for('PAMAP2','sr015_bl8',STUDY);device=torch.device('cuda:0');checks=[]
    truth,observed,starts=next(data.batches('train',128,epoch=0,seed=2023))
    fingerprint=hashlib.sha256(starts.tobytes()+observed.numpy().tobytes()).hexdigest()
    for arm in ('A0','A1'):
        for index,width in enumerate((64,256)):
            binding=audit['sources'][f'{arm}_2024'];seed_all(2024);model=V125Adapter(data,arm,2024).to(device)
            model.load_state_dict(torch.load(binding['coarse_checkpoint'],map_location=device,weights_only=True))
            params,initial=initialize_branch(model,index,2024)
            assert initial==audit['initializations'][f'{arm}_2024'][str(width)]
            model.eval();member=model.core.recent_mean_projector.members[index];member.train()
            frozen=frozen_hash(model.state_dict(),index);library=model.memory.snapshot_hash();initial_member=tensor_hash(member.state_dict())
            optimizer=torch.optim.Adam(params,lr=.001,weight_decay=0.);y=truth.to(device);obs=observed.to(device)
            candidate=model(torch.where(obs,y,0.),obs,starts);assert torch.equal(candidate[obs],y[obs])
            valid=torch.from_numpy(data.valid_for(starts)).to(device);loss=calibration_loss(candidate,y,obs,valid)
            assert torch.isfinite(loss);loss.backward()
            assert not any(p.grad is not None for p in model.parameters() if not p.requires_grad)
            optimizer.step()
            assert frozen_hash(model.state_dict(),index)==frozen and model.memory.snapshot_hash()==library
            assert tensor_hash(member.state_dict())!=initial_member
            assert set(int(v['step']) for v in optimizer.state.values())=={1}
            checks.append(dict(arm=arm,width=width,model_seed=2024,batch_size=128,loss=float(loss),frozen_prefix_unchanged=True,
                retrieval_keys_and_prototypes_unchanged=True,only_allowed_parameters_updated=True,observed_unchanged=True,
                novel_initialization_sha256=initial['novel_tensor_hash'],training_input_sha256=fingerprint))
            del model,optimizer,params,member,candidate,loss;gc.collect();torch.cuda.empty_cache()
    assert checks[0]['novel_initialization_sha256']==checks[2]['novel_initialization_sha256']
    assert checks[1]['novel_initialization_sha256']==checks[3]['novel_initialization_sha256']
    write_json(OUT/'preflight.json',dict(passed=True,checked_unix=time.time(),checks=checks,formal_training_not_started=True,
        own_coarse_checkpoint_required=True,calibration_only=True,protocol_sha256=audit['protocol_sha256']))
    print(json.dumps(dict(passed=True,real_full_batch_updates=len(checks),frozen_prefix_keys_and_libraries_all_unchanged=True)))
if __name__=='__main__':main()
