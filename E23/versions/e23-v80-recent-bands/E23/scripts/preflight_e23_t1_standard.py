"""Disposable CUDA checks; no weights are reused in formal training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import torch

from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import build_forecast_model, _starts
from integrations.forecast_compare.author_profiles import load_author_profiles, build_author_model
from integrations.forecast_compare.author_provenance import write_new_record
from integrations.forecast_compare.protocol import load_p4_protocol, with_training_seed, iter_p4_batches
from integrations.forecast_compare.t1_p4 import T1ForecastModel
from scripts.run_author_aligned_p4 import _forecast_head, _repair_loss, _seed, _autocast

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    masks = ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'
    protocol = with_training_seed(load_p4_protocol(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml', masks), 102)
    batch = next(iter_p4_batches(protocol, 'train', epoch=0, batch_size=16)).to(device)
    altered = replace(batch, normalized_target=batch.normalized_target+1000,
                      forecast_target=batch.forecast_target-1000)
    with tempfile.TemporaryDirectory() as temp:
        cfg = Path(temp)/'e23.yaml'
        cfg.write_text((ROOT/'configs/e23_multi_expert_p4_width64_forecast_sensitive_v31.yaml').read_text().replace('seed: 2021', 'seed: 102'))
        settings = load_e23_settings(cfg, protocol=protocol)
        _seed(102)
        ours = build_forecast_model(protocol, settings, device=device)
    def ours_predict(b):
        starts = _starts(b, stride=ours.repair.stride, patches=ours.repair.patch_count, device=device)
        return ours(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)[0]
    _seed(102)
    head = _forecast_head(protocol).to(device)
    if parameter_hash(head) != parameter_hash(ours.predictor):
        raise ValueError('prediction head initialization mismatch')
    profiles = load_author_profiles()
    author = profiles['models']['T1']
    _seed(102)
    imputer = build_author_model('T1', author, ROOT/author['root'], device)
    imputer.train()
    with _autocast(author['training'], device):
        repair_loss = _repair_loss('T1', imputer, batch)
    repair_loss.backward()
    if not torch.isfinite(repair_loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in imputer.parameters()):
        raise ValueError('T1 BF16 repair gradients are nonfinite')
    imputer.zero_grad(set_to_none=True)
    baseline = T1ForecastModel(imputer, head).to(device)
    initial_heads = {'E23-v31-B': parameter_hash(ours.predictor), 'T1': parameter_hash(baseline.predictor)}
    results = {}
    for name, model, predict in [('E23-v31-B', ours, ours_predict), ('T1', baseline, baseline)]:
        model.eval()
        with torch.no_grad():
            first, second = predict(batch), predict(altered)
        if not torch.equal(first, second):
            raise ValueError(f'{name} prediction depends on hidden/future ground truth')
        model.train()
        loss = (predict(batch)-batch.forecast_target).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f'{name} forecast gradients are nonfinite')
        results[name] = dict(forecast_loss=float(loss.detach()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),
                             predictor_initial_sha256=initial_heads[name],
                             hidden_and_future_truth_not_inputs=True, finite_cuda_gradients=True)
    write_new_record(args.output, dict(device=str(device), seed=102, T1_bf16_repair_loss=float(repair_loss.detach()),
                                       formal_weights_reused=False, models=results))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
