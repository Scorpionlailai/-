from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
import yaml

from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.recent_state import recent_state_loss
from malr.models.explicit_repair import build_explicit_repair
from tests.integration.test_e23_stage1 import _protocol_settings, _batch

WORK = Path(__file__).resolve().parents[2]


def settings(tmp_path, protocol, point=True, overlap=False, structure=False):
    doc = yaml.safe_load((WORK/'configs/e23_v46_front_context_seed502.yaml').read_text())
    doc['model'].update(point_router='pointwise_v1' if point else 'none', overlap_fusion='confidence_v1' if overlap else 'none')
    doc['training'].update(seed=protocol.training_seed, stage1_structure='recent_state_v1' if structure else 'none')
    path = tmp_path/f'{point}-{overlap}-{structure}.yaml';path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return load_e23_settings(path, protocol=protocol)


@pytest.mark.parametrize('point,overlap', [(True,False),(False,True),(True,True)])
def test_zero_start_original_parameters_rng_function_and_available_weights(tmp_path, point, overlap):
    p, _ = _protocol_settings();cfg = settings(tmp_path, p, point, overlap)
    torch.manual_seed(502);old = tr.build_forecast_model(p, replace(cfg, point_router='none', overlap_fusion='none'), device=torch.device('cpu')).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(502);new = tr.build_forecast_model(p, cfg, device=torch.device('cpu')).eval()
    assert torch.equal(rng, torch.get_rng_state()) and parameter_hash(new.predictor) == parameter_hash(old.predictor)
    for k, v in old.repair.state_dict().items():assert torch.equal(v, new.repair.state_dict()[k]), k
    b = _batch('validation');starts = tr._starts(b, stride=8, patches=12, device=torch.device('cpu'))
    with torch.no_grad():
        a, oa = old(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        c, oc = new(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
    assert torch.equal(a, c) and torch.equal(oa.explicit.normalized_sequence, oc.explicit.normalized_sequence)
    if point:
        w = oc.router.point_weights
        assert torch.equal(w, oc.router.weights[..., None, :].expand_as(w))
        assert w.masked_select(~oc.router.available[..., None, :].expand_as(w)).eq(0).all()
    if overlap:assert oc.explicit.aggregation_weights[oc.explicit.validity_patches].eq(1).all()


def test_nonzero_point_weights_vary_across_patch_and_isolate_hidden_values_and_disabled_experts(tmp_path):
    p, _ = _protocol_settings();cfg = settings(tmp_path, p, True, True)
    torch.manual_seed(502);m = tr.build_forecast_model(p, cfg, device=torch.device('cpu')).eval()
    with torch.no_grad():
        m.repair.point_router.score[-1].weight.normal_(std=.05)
        m.repair.overlap_confidence.score[-1].weight.normal_(std=.05)
    b = _batch('validation');starts = tr._starts(b, stride=8, patches=12, device=torch.device('cpu'))
    contaminated = b.normalized_observed.clone();contaminated[~b.observed_mask] = 1e9
    with torch.no_grad():
        a, oa = m(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        c, oc = m(contaminated, b.observed_mask, b.query_mean, b.query_std, starts)
    assert torch.equal(a, c)
    assert torch.equal(oa.router.point_weights, oc.router.point_weights)
    assert torch.equal(oa.explicit.normalized_sequence[b.observed_mask], b.normalized_observed[b.observed_mask])
    w = oa.router.point_weights
    torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]), atol=2e-7, rtol=0)
    assert w.min() >= -1e-7
    assert (w[:, :, 0, 8]-w[:, :, 0, 15]).abs().sum() > 0
    assert w.masked_select(~oa.router.available[..., None, :].expand_as(w)).eq(0).all()
    r = m.repair(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts,
        disabled_experts=frozenset({'historical', 'variable', 'temporal'}))
    assert r.router.point_weights[..., 0].eq(1).all() and r.router.point_weights[..., 1:].eq(0).all()
    assert torch.isfinite(r.explicit.normalized_sequence).all()


def test_new_modules_receive_repair_gradient_and_body_learns_after_first_step(tmp_path):
    p, _ = _protocol_settings();cfg = settings(tmp_path, p, True, True)
    m = tr.build_forecast_model(p, cfg, device=torch.device('cpu'));b = _batch('train')
    starts = tr._starts(b, stride=8, patches=12, device=torch.device('cpu'))
    opt = torch.optim.Adam(m.repair.parameters(), lr=.001)
    for step in range(2):
        opt.zero_grad();r = m.repair(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        loss, _ = tr.stage1_objective(r.explicit.normalized_sequence, b.normalized_target, b.observed_mask,
            torch.tensor(0.), retrieval_weight=.1, loss_mode='global_mse_v1', query_std=b.query_std)
        loss.backward()
        for module in (m.repair.point_router, m.repair.overlap_confidence):
            assert module.score[-1].weight.grad.abs().sum() > 0
            if step:assert module.score[0].weight.grad.abs().sum() > 0
        assert all(torch.isfinite(x.grad).all() for x in m.repair.parameters() if x.grad is not None)
        opt.step()


def test_weighted_copies_known_answer_gradients_padding_and_observation_preservation():
    candidates = torch.tensor([[[[1.,2.,3.,4.],[9.,10.,11.,12.],[13.,14.,999.,999.]]]], requires_grad=True)
    valid = torch.ones_like(candidates, dtype=torch.bool);valid[:, :, -1, 2:] = False
    weights = torch.ones_like(candidates);weights[:, :, 1] = 3
    mask = torch.zeros_like(valid);mask[:, :, 0, 0] = True
    observed = torch.zeros_like(candidates);observed[:, :, 0, 0] = 77
    result = build_explicit_repair(candidates, observed, mask, valid, torch.zeros(1,1), torch.ones(1,1),
        window_len=6, patch_len=4, stride=2, candidate_weights=weights)
    torch.testing.assert_close(result.normalized_sequence.flatten(), torch.tensor([77.,2.,7.5,8.5,11.5,12.5]))
    assert torch.equal(result.coverage.flatten(), torch.tensor([1,1,2,2,2,2]))
    result.normalized_sequence.sum().backward()
    assert candidates.grad[0,0,-1,2:].eq(0).all() and candidates.grad[0,0,0,0] == 0
    assert candidates.grad[0,0,0,2] == .25 and candidates.grad[0,0,1,0] == .75


def test_recent_state_loss_formula_global_scale_mask_isolation_and_gradients():
    candidate = torch.zeros(1,96,1,requires_grad=True);target = torch.zeros_like(candidate)
    hidden = torch.zeros_like(candidate, dtype=torch.bool);hidden[:,64:] = True
    x = torch.linspace(-1,1,32)
    with torch.no_grad():candidate[:,64:,0] = 2+.5*x
    loss, parts = recent_state_loss(candidate, target, hidden, torch.ones(1,1))
    torch.testing.assert_close(loss, torch.tensor(.2*4+.2*.25))
    target2 = target.clone();target2[:, :64] = float('nan')
    assert torch.equal(loss, recent_state_loss(candidate, target2, hidden, torch.ones(1,1))[0])
    scaled = recent_state_loss(candidate*100, target, hidden, torch.tensor([[.01]]))[0]
    torch.testing.assert_close(loss, scaled)
    loss.backward();assert candidate.grad[:, :64].eq(0).all() and candidate.grad[:,64:].abs().sum() > 0


def test_point_series_checkpoint_roundtrip_and_identity_rejection(monkeypatch, tmp_path):
    p, _ = _protocol_settings();cfg = settings(tmp_path, p, True, True, True)
    monkeypatch.setattr(tr, 'iter_p4_batches', lambda _p, split, **kw: iter((_batch(split),)))
    s1 = tr.train_repair(p, cfg, output=tmp_path/'s1', device=torch.device('cpu'), epochs=1)
    s2 = tr.train_forecast(p, cfg, stage1_manifest=s1, output=tmp_path/'s2', device=torch.device('cpu'), epochs=1,
        selection_metric='mae', forecast_loss='mse_plus_0p2_mae', repair_retention='uniform_v1', repair_update_policy='freeze_front_v1')
    payload = tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'))
    assert payload['point_router']=='pointwise_v1' and payload['stage1_structure']=='recent_state_v1'
    record=json.loads(s2.read_text());record['overlap_fusion']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='overlap_fusion.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('key,value', [('point_router','wrong'),('overlap_fusion',True),('stage1_structure','wrong')])
def test_invalid_policy_config_rejected(tmp_path,key,value):
    p, _ = _protocol_settings();cfg=settings(tmp_path,p)
    path=tmp_path/'True-False-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['training' if key=='stage1_structure' else 'model'][key]=value
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
