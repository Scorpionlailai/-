"""Two recent levels must preserve evidence boundaries and the forecast interface."""
from dataclasses import replace
import torch
from integrations.e23 import train as tr
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_cross_variable_level import configuration,batch

def models(tmp_path):
    protocol,cfg=configuration(tmp_path)
    torch.manual_seed(2023)
    old=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu')).eval()
    rng=torch.get_rng_state()
    torch.manual_seed(2023)
    new=tr.build_forecast_model(protocol,replace(cfg,recent_mean_projection='global_recent_bands_v3'),device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state())
    return old,new,batch()

def run(model,b,values=None,valid=None):
    starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    return model.repair(b.normalized_observed if values is None else values,b.observed_mask,
        b.query_mean,b.query_std,starts,valid_mask=valid)

def test_zero_start_preserves_original_function_parameters_rng_and_head(tmp_path):
    old,new,b=models(tmp_path)
    for k,v in old.state_dict().items(): assert torch.equal(v,new.state_dict()[k]),k
    assert parameter_hash(old.predictor)==parameter_hash(new.predictor)
    assert sum(p.numel() for p in new.repair.parameters())-sum(p.numel() for p in old.repair.parameters())==6210
    with torch.no_grad():
        a=run(old,b);z=run(new,b)
    assert torch.equal(a.explicit.normalized_sequence,z.explicit.normalized_sequence)
    assert torch.equal(a.explicit.repaired_patches,z.explicit.repaired_patches)

def test_two_band_outputs_have_separate_scope_and_preserve_observations(tmp_path):
    _,new,b=models(tmp_path)
    with torch.no_grad():
        a=run(new,b)
        new.repair.recent_mean_projector.recent_band_network[-1].bias.copy_(torch.tensor([.2,-.3]))
        z=run(new,b)
    delta=(z.explicit.normalized_sequence-a.explicit.normalized_sequence)*b.query_std[:,None]
    assert torch.equal(a.explicit.normalized_sequence[:,:64],z.explicit.normalized_sequence[:,:64])
    assert torch.equal(z.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    for i,(start,stop) in enumerate(((64,80),(80,96))):
        mask=~b.observed_mask[:,start:stop];n=mask.sum(1)
        expected=2*torch.tanh(torch.tensor([.2,-.3])[i]/2)*(16/n.clamp_min(1)).clamp_max(4)
        expected=expected[:,None].expand_as(delta[:,start:stop])
        torch.testing.assert_close(delta[:,start:stop][mask],expected[mask],rtol=1e-4,atol=1e-5)

def test_global_source_features_ignore_hidden_payload_and_all_empty_windows(tmp_path):
    _,new,b=models(tmp_path)
    with torch.no_grad():new.repair.recent_mean_projector.recent_band_network[-1].bias.fill_(.3)
    valid=torch.ones_like(b.observed_mask);valid[:,93,2]=False
    with torch.no_grad():
        a=run(new,b,torch.where(b.observed_mask & valid,b.normalized_observed,0),valid)
        z=run(new,b,torch.where(b.observed_mask & valid,b.normalized_observed,float('nan')),valid)
        assert torch.equal(a.explicit.normalized_sequence,z.explicit.normalized_sequence)
        empty=replace(b,observed_mask=torch.zeros_like(b.observed_mask))
        base=run(new,empty)
        new.repair.recent_mean_projector.recent_band_network[-1].bias.fill_(1.)
        changed=run(new,empty)
    assert torch.isfinite(base.explicit.normalized_sequence).all()
    assert torch.equal(base.explicit.normalized_sequence,changed.explicit.normalized_sequence)

def test_subwindow_source_levels_can_change_target_band_without_future(tmp_path):
    _,new,b=models(tmp_path)
    with torch.no_grad():
        net=new.repair.recent_mean_projector.recent_band_network
        net[0].weight.zero_();net[0].bias.fill_(1);net[-1].weight.zero_();net[-1].bias.zero_()
        net[0].weight[0,50]=.3 # recent source variable1 mean8 in global units
        net[-1].weight[1,0]=.2
        a=run(new,b)
        values=b.normalized_observed.clone();values[:,88:,1]+=b.observed_mask[:,88:,1]*.5
        z=run(new,b,values)
    hidden=~b.observed_mask[:,80:,0]
    assert (a.explicit.standardized_sequence[:,80:,0]-z.explicit.standardized_sequence[:,80:,0])[hidden].abs().max()>1e-5

def test_finite_repair_gradients_reach_both_bands(tmp_path):
    _,new,b=models(tmp_path);new.train();opt=torch.optim.Adam(new.repair.parameters(),lr=.001)
    for step in range(2):
        opt.zero_grad();out=run(new,b)
        loss,_=tr.stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,
            torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std,structure_mode='recent_mean_v2')
        loss.backward();net=new.repair.recent_mean_projector.recent_band_network
        assert (net[-1].weight.grad.abs().sum(-1)>0).all()
        if step:assert net[0].weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
        opt.step()
