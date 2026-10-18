"""Catch pseudo-label scoring, exposure of native gaps, and cache regeneration."""
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def modules():
    assert importlib.util.find_spec('integrations.physionet_baselines') is not None, 'PhysioNet2012 pipeline is missing'
    from integrations.physionet_baselines import run, data, cache
    return run, data, cache


class ZeroRepair(torch.nn.Module):
    def forward(self, values, observed, starts):
        return torch.where(observed, values, torch.zeros_like(values))


class ZeroForecast(torch.nn.Module):
    def forward(self, patches, mask, mean, std, **kwargs):
        return patches.new_zeros(len(patches), 1, 96)


def test_scores_exclude_forward_filled_labels():
    run, _, _ = modules()
    values = np.full((192, 1), 1000., dtype=np.float32)
    values[0, 0], values[95, 0], values[96, 0] = 1., 3., 4.
    original = np.zeros_like(values, dtype=bool)
    original[[0, 95, 96]] = True
    data = SimpleNamespace(values=values, starts={'test': np.array([0])},
        bounds=SimpleNamespace(train=SimpleNamespace(end=96)))
    data.valid_for = lambda starts, offset=0: np.stack([original[s+offset:s+offset+96] for s in starts])
    data.batches = lambda *args, **kwargs: iter([(torch.from_numpy(values[:96][None]),
        torch.zeros(1, 96, 1, dtype=torch.bool), np.array([0]))])
    result = run.evaluate(ZeroRepair(), data, 'test', torch.device('cpu'), batch_size=1,
                          mask_seed=123, head=ZeroForecast())
    assert result['full96']['coordinates'] == 2
    assert result['full96']['mse'] == 5.
    assert result['full96']['mae'] == 2.
    assert result['recent32']['coordinates'] == 1
    assert result['recent32']['mse'] == 9.
    assert result['forecast']['coordinates'] == 1
    assert result['forecast']['mse'] == 16.


def test_native_gaps_are_never_exposed_as_observations(tmp_path):
    _, data_module, _ = modules()
    data = data_module.SharedData('PhysioNet2012', 'sr010_bl8',
        data_module.ROOT/'E24/artifacts/physionet_reproduction_20261015/shared_assets')
    assert data.real_observed.shape == (2304000, 35)
    assert int((~data.real_observed).sum()) == 76039243
    for split in ('train', 'validation', 'test'):
        _, observed, starts = next(data.batches(split, 64, mask_seed=data.test_seeds[0]))
        assert not (observed.numpy() & ~data.valid_for(starts)).any()
    assert len(data.test_seeds) == len(set(data.test_seeds)) == 20


def test_forecast_cache_refuses_missing_output(tmp_path):
    _, _, cache_module = modules()
    stage = tmp_path / 'stage1'
    stage.mkdir()
    torch.save({}, stage / 'selected.pt')
    data = SimpleNamespace(n_vars=1, manifest_sha='fixture', starts={'test':np.array([0])})
    cache = cache_module.RepairCache(ZeroRepair(), data, tmp_path, torch.device('cpu'), 1,
                                    allow_generation=False)
    with pytest.raises(FileNotFoundError):
        cache.load('test', 123)


def test_hidden_loss_has_no_gradient_on_native_missing_values():
    modules()
    from integrations.physionet_baselines.losses import hidden_error
    candidate = torch.tensor([[[2.], [100.], [9.]]], requires_grad=True)
    truth = torch.tensor([[[0.], [-1000.], [1.]]])
    observed = torch.tensor([[[False], [False], [True]]])
    valid = torch.tensor([[[True], [False], [True]]])
    error = hidden_error(candidate, truth, observed, valid)
    assert error.tolist() == [2.]
    error.square().mean().backward()
    assert candidate.grad.tolist() == [[[4.], [0.], [0.]]]


def test_v125_checkpoint_preserves_published_history(tmp_path):
    run, data_module, _ = modules()
    from integrations.physionet_baselines.references import build_reference
    torch.set_num_threads(4)
    torch.manual_seed(1)
    data = data_module.SharedData('PhysioNet2012', 'sr010_bl8',
        data_module.ROOT/'E24/artifacts/physionet_reproduction_20261015/shared_assets')
    model = build_reference('v125', data, torch.device('cpu')).eval()
    model.memory.stage_patch(0, 0, np.ones(16), np.ones(64), observed=True)
    model.publish_epoch(1)
    state = model.state_dict()
    restored = build_reference('v125', data, torch.device('cpu')).eval()
    restored.load_state_dict(state)
    assert restored.memory.snapshot_hash() == model.memory.snapshot_hash()
    assert run.state_sha(restored) == run.state_sha(model)
    assert restored.core.historical.index is not None


def test_v125_loss_handles_a_native_missing_patch(tmp_path):
    _, data_module, _ = modules()
    from integrations.physionet_baselines.references import build_reference
    data = data_module.SharedData('PhysioNet2012', 'sr010_bl8',
        data_module.ROOT/'E24/artifacts/physionet_reproduction_20261015/shared_assets')
    model = build_reference('v125', data, torch.device('cpu'))
    model.inference_path = 'coarse'
    truth, observed, starts = next(data.batches('train', 2))
    data.real_observed = np.array(data.real_observed, copy=True)
    data.real_observed[starts[0]:starts[0]+32,0] = False
    observed[0,:32,0] = False
    loss = model.loss(truth, observed, starts)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_chunked_contrastive_loss_preserves_all_negatives_and_gradients():
    from integrations.physionet_baselines.losses import chunked_contrastive_loss
    from malr.prototypes.retrieval import masked_contrastive_loss
    q = torch.tensor([[1.,2.],[3.,-.5],[-2.,1.],[.1,-.2]],requires_grad=True)
    teacher = torch.tensor([[.2,.4],[-.1,.7],[.6,.3],[.4,-.5]],requires_grad=True)
    expected = masked_contrastive_loss(q,teacher,temperature=.7)
    expected_gradient = torch.autograd.grad(expected,q)[0]
    actual = chunked_contrastive_loss(q,teacher,temperature=.7,chunk_size=3)
    actual_gradient, teacher_gradient = torch.autograd.grad(actual,(q,teacher),allow_unused=True)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(actual_gradient,expected_gradient,rtol=1e-6,atol=1e-7)
    assert teacher_gradient is None
