"""Verify partial freezing, original predictor and reused Stage1 checkpoint."""
import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, load_stage1_selection, _new_memory, _starts, stage2_objective
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v47_frozen_front_sr015_bl8_seed502_20261005')


def main():
    torch.set_num_threads(4);device = torch.device('cuda:1');plan = json.loads((ROOT/'plan.json').read_text())
    protocol = load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context', training_seed=502)
    settings = load_e23_settings(Path(plan['config']), protocol=protocol)
    payload = load_stage1_selection(Path(plan['cached_stage1']), protocol, settings, device=device)
    torch.manual_seed(502);model = build_forecast_model(protocol, settings, device=device)
    head = parameter_hash(model.predictor);assert head == plan['predictor_initial_sha256']
    assert (WORK/'src/malr/models/patchtst.py').read_bytes() == Path('/home/xyw/paper/.e23-v46-front/E23/src/malr/models/patchtst.py').read_bytes()
    model.repair.load_state_dict(payload['model_state']);model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot())
    for p in model.repair.historical.key_encoder.parameters():p.requires_grad_(False)
    optimizer = build_stage2_optimizer(model, mode='freeze_front_v1', learning_rate=1e-4, weight_decay=0)
    before = parameter_hash(model.repair.front_context);old_decoder = model.repair.decoder.weight.detach().clone()
    b = next(iter_p4_batches(protocol, 'validation', batch_size=8)).to(device)
    starts = _starts(b, stride=8, patches=12, device=device);model.eval()
    with torch.no_grad():
        a, out = model(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        polluted = torch.where(b.observed_mask, b.normalized_observed, torch.full_like(b.normalized_observed, 1e9))
        c, _ = model(polluted, b.observed_mask, b.query_mean, b.query_std, starts)
        assert torch.equal(a, c)
        assert torch.equal(out.explicit.normalized_sequence[b.observed_mask], b.normalized_observed[b.observed_mask])
    stage2_training_mode(model, 'freeze_front_v1');optimizer.zero_grad(set_to_none=True)
    forecast, out = model(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
    loss = stage2_objective(forecast, b.forecast_target, loss_mode='mse_plus_0p2_mae');loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.repair.front_context.parameters())
    assert model.repair.decoder.weight.grad.abs().sum() > 0
    optimizer.step();assert before == parameter_hash(model.repair.front_context)
    assert not torch.equal(old_decoder, model.repair.decoder.weight)
    result = dict(status='passed', original_predictor_source_identical=True, predictor_initial_sha256=head,
        predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),
        repair_parameters=sum(p.numel() for p in model.repair.parameters()),
        front_context_parameters=sum(p.numel() for p in model.repair.front_context.parameters()),
        repair_trainable_parameters=sum(p.numel() for p in model.repair.parameters() if p.requires_grad),
        frozen_front_unchanged_after_optimizer_update=True, other_repair_parameters_learn=True,
        hidden_input_canary_independent=True, visible_values_preserved=True, no_test_evaluation=True,
        fixed_stage1_reused=True, finite_loss=float(loss.detach()))
    with (ROOT/'preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
