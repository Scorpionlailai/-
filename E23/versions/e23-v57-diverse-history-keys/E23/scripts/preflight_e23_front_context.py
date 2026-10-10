"""Check v46 against original v41 initialization on the sealed high-missing data."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, _new_memory, _starts, stage1_objective, _retrieval_loss
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v46_front_context_sr015_bl8_seed502_20261005')


def main():
    torch.set_num_threads(4);device = torch.device('cuda:1')
    plan = json.loads((ROOT/'plan.json').read_text())
    protocol = load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context', training_seed=502)
    assert protocol.config.missing.start_rate == .15 and protocol.config.missing.block_len == 8
    settings = load_e23_settings(Path(plan['config']), protocol=protocol)
    torch.manual_seed(502);old = build_forecast_model(protocol, replace(settings, front_context='none'), device=device).eval()
    torch.manual_seed(502);new = build_forecast_model(protocol, settings, device=device).eval()
    head_hash = parameter_hash(new.predictor)
    assert head_hash == parameter_hash(old.predictor) == plan['predictor_initial_sha256']
    original_parameters = sum(p.numel() for p in old.repair.parameters())
    assert original_parameters == 174362
    assert sum(p.numel() for p in new.predictor.parameters()) == 35427
    for name, value in old.repair.state_dict().items():assert torch.equal(value, new.repair.state_dict()[name]), name
    reference_source = Path('/home/xyw/paper/.e23-highmissing/E23/src/malr/models/patchtst.py')
    assert (WORK/'src/malr/models/patchtst.py').read_bytes() == reference_source.read_bytes()
    reference = ROOT.parent/'t1_v41_higher_missing_seed502_20261004/sr015_bl8/E23-v41-slow/stage1/runs/e23-stage1.json'
    saved = json.loads(reference.read_text());checkpoint = Path(saved['checkpoint'])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cached = torch.load(checkpoint, map_location=device, weights_only=False)
    # The reference bank is used for this read-only check only. Training starts fresh.
    for model in (old, new):
        model.repair.historical.retrieval.load_state_dict(cached['retrieval_state'])
        memory = _new_memory(protocol);memory.load_state_dict(cached['memory_state'])
        model.repair.historical.refresh_snapshot(memory.snapshot())
    b = next(iter_p4_batches(protocol, 'validation', batch_size=4)).to(device)
    starts = _starts(b, stride=8, patches=12, device=device)
    with torch.no_grad():
        a, oa = old(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        p, out = new(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        assert torch.equal(a, p) and torch.equal(oa.explicit.normalized_sequence, out.explicit.normalized_sequence)
        new.repair.front_context.output.weight.normal_(std=.01)
        p, out = new(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        polluted = torch.where(b.observed_mask, b.normalized_observed, torch.full_like(b.normalized_observed, 1e9))
        p2, out2 = new(polluted, b.observed_mask, b.query_mean, b.query_std, starts)
        assert torch.equal(p, p2) and torch.equal(out.base_latents, out2.base_latents)
        assert torch.equal(out.explicit.normalized_sequence[b.observed_mask], b.normalized_observed[b.observed_mask])
    new.train();new.zero_grad(set_to_none=True)
    prediction, out = new(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts,
                          specialist_exposure=.2, specialist_seed=503)
    loss, _ = stage1_objective(out.explicit.normalized_sequence, b.normalized_target, b.observed_mask,
        _retrieval_loss(new.repair, b), retrieval_weight=.1, loss_mode='global_mse_v1', query_std=b.query_std)
    loss = loss+(prediction-b.forecast_target).square().mean();loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
    assert new.repair.front_context.input[0].weight.grad.abs().sum() > 0
    assert new.repair.front_context.blocks[1].long.weight.grad.abs().sum() > 0
    assert checkpoint_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = dict(status='passed', original_predictor_source_identical=True,
        predictor_initial_sha256=head_hash, predictor_parameters=35427,
        original_repair_parameters=original_parameters,
        repair_parameters=sum(p.numel() for p in new.repair.parameters()),
        front_context_parameters=sum(p.numel() for p in new.repair.front_context.parameters()),
        original_initial_parameters_and_function_identical=True, hidden_input_canary_independent=True,
        finite_loss=float(loss.detach()), frontend_gradient_verified=True,
        no_test_evaluation=True, cached_bank_used_only_for_preflight=True,
        fresh_training_required=True, mask_start_rate=.15, block_len=8)
    with (ROOT/'preflight.json').open('x') as f:json.dump(result, f, indent=2);f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':main()
