from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from integrations.e23 import train as training
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.data.patching import patchify_with_state, patchify_preserving_values
from tests.integration.test_e23_stage1 import _protocol_settings, _batch

ROOT = Path(__file__).resolve().parents[2]


def case_batch(split):
    b=_batch(split)
    mask=b.observed_mask.clone();mask[:,79:89]=False
    return replace(b,observed_mask=mask,normalized_observed=b.normalized_target*mask)


def trial(tmp_path, protocol, expanded=False):
    raw = yaml.safe_load((ROOT/'configs/e23_multi_expert_p4_width64_iterative_tv_h_r1_v29.yaml').read_text())
    raw['training']['seed'] = protocol.training_seed
    raw['model'].update(point_refiner_span=32, repair_stride=16,
        point_feature_width=32 if expanded else 0, temporal_local_radius=1,
        temporal_boundaries=[1,2,4,8])
    path = tmp_path/('expanded.yaml' if expanded else 'basic.yaml')
    path.write_text(yaml.safe_dump(raw,sort_keys=False))
    return load_e23_settings(path,protocol=protocol)


def test_nonoverlap_tiles_and_partial_final_tile_keep_unique_real_coordinates():
    for T in [96,99]:
        x = torch.arange(T*2).reshape(1,T,2).float()
        mask = torch.ones_like(x,dtype=torch.bool)
        p = patchify_with_state(x,mask,16,16)
        assert p.values.shape[2] == (T+15)//16
        assert p.validity.sum() == T*2
        for i in range(p.values.shape[2]):
            length = min(16,T-i*16)
            torch.testing.assert_close(p.values[:,:,i,:length].transpose(1,2),x[:,i*16:i*16+length])


@pytest.mark.parametrize('expanded',[False,True])
def test_nonoverlap_repair_preserves_observation_and_original_prediction_geometry(tmp_path,expanded):
    protocol,_ = _protocol_settings(); s=trial(tmp_path,protocol,expanded)
    torch.manual_seed(502)
    model = training.build_forecast_model(protocol,s,device=torch.device('cpu')).eval()
    assert model.repair.patch_count == 6 and model.predictor.core.patch_num == 12
    assert sum(p.numel() for p in model.predictor.parameters()) == 35427
    model.forecast_padding='replicate_last_value_v1'
    b=case_batch('validation');starts=training._starts(b,stride=16,patches=6,device=torch.device('cpu'))
    captured=[]
    hook=model.predictor.register_forward_pre_hook(lambda _m,a:captured.append(a[0].detach().clone()))
    with torch.no_grad():
        prediction,out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    hook.remove()
    assert prediction.shape==b.forecast_target.shape
    assert out.explicit.coverage.eq(1).all()
    assert out.explicit.repaired_patches.shape[2]==6
    assert torch.equal(out.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    p=patchify_preserving_values(out.explicit.normalized_sequence,b.observed_mask,16,8)
    torch.testing.assert_close(captured[0],p.values,rtol=0,atol=0)
    torch.testing.assert_close(captured[0][:,:,:-1,8:],captured[0][:,:,1:,:8],rtol=0,atol=0)


def test_expansion_does_not_change_initial_function_head_or_read_hidden_values(tmp_path):
    protocol,_=_protocol_settings();s=trial(tmp_path,protocol);expanded=trial(tmp_path,protocol,True)
    torch.manual_seed(502);a=training.build_forecast_model(protocol,s,device=torch.device('cpu')).eval()
    torch.manual_seed(502);b=training.build_forecast_model(protocol,expanded,device=torch.device('cpu')).eval()
    assert parameter_hash(a.predictor)==parameter_hash(b.predictor)
    # Compare with the same original seed; constructor RNG is isolated for both refiners.
    torch.manual_seed(502);original=training.build_forecast_model(protocol,replace(s,repair_stride=8,
        temporal_local_radius=2,temporal_boundaries=(2,4,8,16)),device=torch.device('cpu'))
    assert parameter_hash(a.predictor)==parameter_hash(original.predictor)
    batch=case_batch('validation');starts=training._starts(batch,stride=16,patches=6,device=torch.device('cpu'))
    with torch.no_grad():
        pa,oa=a(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        pb,ob=b(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    torch.testing.assert_close(pa,pb,rtol=0,atol=0)
    torch.testing.assert_close(oa.explicit.normalized_sequence,ob.explicit.normalized_sequence,rtol=0,atol=0)
    with torch.no_grad():b.repair.point_refiner.output.weight.normal_(std=.02)
    canary=batch.normalized_observed.clone();canary[~batch.observed_mask]=1e6
    with torch.no_grad():
        p1,_=b(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        p2,_=b(canary,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    torch.testing.assert_close(p1,p2,rtol=0,atol=0)


def test_dense_history_staging_is_identical_for_nonoverlap_queries(tmp_path):
    protocol,_=_protocol_settings();s=trial(tmp_path,protocol)
    torch.manual_seed(502);non=training.build_repair_model(protocol,s,device=torch.device('cpu'))
    torch.manual_seed(502);old=training.build_repair_model(protocol,replace(s,repair_stride=8,
        temporal_local_radius=2,temporal_boundaries=(2,4,8,16)),device=torch.device('cpu'))
    b=_batch('train');dense=training._new_memory(protocol);sparse=training._new_memory(protocol)
    training._stage_observed(dense,old,b,training._starts(b,stride=8,patches=12,device=torch.device('cpu')))
    training._stage_observed(sparse,non,b,training._starts(b,stride=16,patches=6,device=torch.device('cpu')))
    assert dense.snapshot_hash()==sparse.snapshot_hash()
    assert training._memory_state_sha256(dense.state_dict())==training._memory_state_sha256(sparse.state_dict())


def test_expansion_and_temporal_refinement_receive_stage1_gradient(tmp_path):
    protocol,_=_protocol_settings();s=trial(tmp_path,protocol,True)
    torch.manual_seed(502);model=training.build_repair_model(protocol,s,device=torch.device('cpu'))
    b=case_batch('validation');starts=training._starts(b,stride=16,patches=6,device=torch.device('cpu'))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for step in range(2):
        optimizer.zero_grad()
        out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        hidden=~b.observed_mask
        (out.explicit.normalized_sequence[hidden]-b.normalized_target[hidden]).square().mean().backward()
        assert model.point_refiner.output.weight.grad.abs().sum()>0
        if step:
            assert model.point_refiner.feature_expand.weight.grad.abs().sum()>0
            assert model.point_refiner.blocks[0].depthwise.weight.grad.abs().sum()>0
        optimizer.step()
    assert model.point_refiner.blocks[0].depthwise.kernel_size==(5,)


def test_pure_mse_slow_stage2_saves_and_reloads_both_selectors(monkeypatch,tmp_path):
    protocol,_=_protocol_settings();s=trial(tmp_path,protocol,True)
    monkeypatch.setattr(training,'iter_p4_batches',lambda _p,split,**kw:iter((case_batch(split),)))
    stage1=training.train_repair(protocol,s,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    selected=training.train_forecast(protocol,s,stage1_manifest=stage1,output=tmp_path/'s2',
        device=torch.device('cpu'),epochs=2,selection_metric='mae',common_stage2=True,
        forecast_loss='mse',repair_retention='none',repair_update_policy='slow_lr_0p1_v1')
    record=json.loads(selected.read_text())
    assert record['repair_learning_rate']==pytest.approx(1e-5)
    assert record['predictor_learning_rate']==1e-4 and record['loss']=='forecast_mse_only'
    for name in ['e23-stage2.json','e23-stage2-mse.json']:
        payload=training.load_stage2_selection(selected.parent/name,protocol,s,device=torch.device('cpu'))
        assert payload['repair_patch_count']==6 and payload['point_feature_width']==32
