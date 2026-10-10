from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model, _trial_identity, _verify_trial_payload


def test_saved_recipe_binds_calibration_and_keeps_legacy_weights_initialization(tmp_path):
    protocol = SimpleNamespace(training_seed=2023, config=SimpleNamespace(
        data=SimpleNamespace(seq_len=96, pred_len=96, datasets={'ETTh1': SimpleNamespace(n_vars=7)}),
        patch=SimpleNamespace(length=16, stride=8)))
    original = Path(__file__).resolve().parents[2]/'configs/e23_v69_signature_position_seed2023.yaml'
    legacy = load_e23_settings(original, protocol=protocol)
    document = yaml.safe_load(original.read_text()); document['model']['historical_calibration'] = 'ridge_offset_loo_v1'
    path = tmp_path/'calibration.yaml'; path.write_text(yaml.safe_dump(document))
    settings = load_e23_settings(path, protocol=protocol)
    torch.manual_seed(2023); old = build_repair_model(protocol, legacy, device=torch.device('cpu'))
    torch.manual_seed(2023); new = build_repair_model(protocol, settings, device=torch.device('cpu'))
    assert new.historical.calibration == 'ridge_offset_loo_v1'
    assert sum(p.numel() for p in new.parameters()) == 420973
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    identity = _trial_identity(settings)
    assert identity['historical_calibration'] == 'ridge_offset_loo_v1'
    _verify_trial_payload(identity, settings)
    with pytest.raises(ValueError, match='historical_calibration'):
        _verify_trial_payload(_trial_identity(legacy), settings)
    document['model']['historical_calibration'] = 'wrong'; path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError):
        load_e23_settings(path, protocol=protocol)
