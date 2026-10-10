from pathlib import Path
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings, load_e23_p4_protocol
from integrations.e23.forecast_sensitive import parameter_hash
from malr.migration import MigrationCatalog


def test_loss_policy_preserves_model_parameters_rng_and_checkpoint_identity(tmp_path):
    work = Path(__file__).resolve().parents[2]
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(work.parent/'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=work/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=work.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',
        training_seed=2023,
    )
    base_path = Path('configs/e23_v69_signature_position_seed2023.yaml')
    old = load_e23_settings(base_path, protocol=protocol)
    assert hasattr(old, 'stage1_gap_structure'), 'new gap supervision configuration is missing'
    path = tmp_path/'gap.yaml'
    cfg = yaml.safe_load(base_path.read_text())
    cfg['training']['stage1_gap_structure'] = 'long_gap_moments_v1'
    path.write_text(yaml.safe_dump(cfg))
    new = load_e23_settings(path, protocol=protocol)
    assert old.stage1_gap_structure == 'none' and new.stage1_gap_structure == 'long_gap_moments_v1'
    torch.manual_seed(2023)
    a = tr.build_forecast_model(protocol, old, device=torch.device('cpu'))
    rng = torch.get_rng_state()
    torch.manual_seed(2023)
    b = tr.build_forecast_model(protocol, new, device=torch.device('cpu'))
    assert torch.equal(rng, torch.get_rng_state())
    assert parameter_hash(a) == parameter_hash(b)
    assert sum(p.numel() for p in b.repair.parameters()) == 420973
    assert sum(p.numel() for p in b.predictor.parameters()) == 35427
    record = tr._identity(protocol, new)
    assert record['gap_min_length'] == 16
    assert record['gap_level_weight'] == record['gap_trend_weight'] == .2
    assert record['gap_weighting'] == 'coordinate_count'
    tr._verify_identity(record, protocol, new)
    record['stage1_gap_structure'] = 'none'
    with pytest.raises(ValueError):
        tr._verify_identity(record, protocol, new)
    cfg['training']['stage1_gap_structure'] = 'unknown'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_e23_settings(path, protocol=protocol)
