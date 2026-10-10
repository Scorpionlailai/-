"""Width affects repair capacity and checkpoint identity, never the head."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
import yaml

from integrations.e23.config import load_e23_settings
from integrations.e23.train import _trial_identity, _verify_trial_payload
from malr.models.e23.global_dense import GlobalDenseRefiner
from malr.models.e23.recent_mean import RecentMeanProjector


def settings_file(tmp_path, width):
    source = Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v108_weather_mae_repair_20261009/v108_weather.yaml')
    document = yaml.safe_load(source.read_text())
    document['model']['global_dense_width'] = width
    path = tmp_path/'candidate.yaml'
    path.write_text(yaml.safe_dump(document))
    protocol = SimpleNamespace(training_seed=2023,bounds=SimpleNamespace(name='Weather'),
        config=SimpleNamespace(data=SimpleNamespace(seq_len=96,pred_len=96,
            datasets={'Weather':SimpleNamespace(n_vars=21)}),
            patch=SimpleNamespace(length=16,stride=8)))
    return path, protocol


def test_default_width_preserves_rng_parameters_and_outputs():
    torch.manual_seed(17)
    old = GlobalDenseRefiner(21, contextual_gate=True)
    state = torch.random.get_rng_state()
    torch.manual_seed(17)
    explicit = GlobalDenseRefiner(21, width=64, contextual_gate=True)
    assert torch.equal(state, torch.random.get_rng_state())
    assert old.parameter_count == explicit.parameter_count == 87402
    assert all(torch.equal(v, explicit.state_dict()[k]) for k,v in old.state_dict().items())


def test_wide_zero_residual_and_protected_coordinates_with_finite_gradient():
    torch.manual_seed(11)
    model = GlobalDenseRefiner(3, width=256, contextual_gate=True)
    observed = torch.zeros(2,96,3,dtype=torch.bool)
    observed[0,::9,:2] = True
    valid = torch.ones_like(observed)
    valid[:,-2:] = False
    sequence = torch.randn(2,96,3)
    explicit = SimpleNamespace(sequence_validity=valid,sequence_observed=observed,
        observed_sequence=torch.randn_like(sequence))
    mean, std = torch.zeros(2,3), torch.ones(2,3)
    assert torch.equal(model(sequence,explicit,mean,std), sequence)
    model(sequence,explicit,mean,std).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        model.output.weight.normal_(std=.01)
        model.guide_gate.weight.normal_(std=.01)
    changed = model(sequence,explicit,mean,std)
    protected = observed | ~valid
    assert torch.equal(changed[protected],sequence[protected])
    assert torch.equal(changed[1],sequence[1])
    with torch.no_grad():
        model.output.weight.zero_()
    gate_only = model(sequence,explicit,mean,std)
    assert torch.equal(gate_only[0,:,2],sequence[0,:,2])


def test_settings_builder_identity_and_checkpoint_width_match(tmp_path):
    path, protocol = settings_file(tmp_path,256)
    settings = load_e23_settings(path,protocol=protocol,allow_variable_scaling=True)
    assert settings.d_model == 64 and settings.global_dense_width == 256
    projector = RecentMeanProjector(21,mode=settings.recent_mean_projection,
                                   global_dense_width=settings.global_dense_width)
    assert sum(p.numel() for p in projector.parameters()) == 1208748
    identity = _trial_identity(settings)
    assert identity['global_dense_width'] == 256
    assert identity['global_dense_extra_parameters'] == 1179669
    assert identity['global_dense_guide_gate_parameters'] == 5397
    _verify_trial_payload(identity,settings)
    with pytest.raises(ValueError,match='checkpoint mismatch'):
        _verify_trial_payload(identity,replace(settings,global_dense_width=64))


@pytest.mark.parametrize('width',[True,0,32,65,512])
def test_invalid_width_is_rejected(tmp_path,width):
    path,protocol=settings_file(tmp_path,width)
    with pytest.raises(ValueError,match='global_dense_width'):
        load_e23_settings(path,protocol=protocol,allow_variable_scaling=True)
