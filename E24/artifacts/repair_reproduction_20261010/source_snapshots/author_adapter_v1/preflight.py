"""Exercise real adapters, loss gradients and common forecast freeze boundaries."""
import argparse
import json
from pathlib import Path

import torch

from .adapters import build_adapter
from .data import SCENES, SharedData, write_json
from .run import build_head, forecast, repair, seed_all, state_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    records = []
    for dataset in ('ETTh1', 'Weather'):
        for scene in SCENES:
            seed_all(2023)
            data = SharedData(dataset, scene, args.root / 'shared_assets')
            model = build_adapter(args.model, data, device)
            truth, observed, starts = next(data.batches('train', 2, epoch=0))
            truth, observed = truth.to(device), observed.to(device)
            loss = model.loss(truth, observed, starts)
            assert torch.isfinite(loss), 'non-finite loss'
            loss.backward()
            assert any(p.grad is not None for p in model.parameters()), 'no repair gradient'
            model.zero_grad(set_to_none=True)
            model.requires_grad_(False).eval()
            # Preflight diffusion uses 2 samples solely to test interfaces.
            # Production profile always uses the author's 100 samples.
            if args.model == 'CSDI':
                model.n_samples = 2
            state_before = state_sha(model)
            with torch.no_grad():
                values = torch.where(observed, truth, 0.)
                output = repair(model, values, observed, starts, 2023)
                assert output.shape == truth.shape and torch.isfinite(output).all()
                assert torch.equal(output[observed], truth[observed])
                # A fully missing variable and a fully missing sample remain
                # valid input cases. No truth is used for either fallback.
                empty_mask = observed.clone()
                empty_mask[0, :, 0] = False
                empty_mask[1] = False
                empty_values = torch.where(empty_mask, truth, 0.)
                empty_output = repair(model, empty_values, empty_mask, starts, 2023)
                assert torch.isfinite(empty_output).all(), 'empty input causes non-finite repair'
            head = build_head(data, device)
            initial_sha = state_sha(head)
            repeated_head = build_head(data, device)
            assert state_sha(repeated_head) == initial_sha, 'head initialization not repeatable'
            optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
            head.train()
            prediction = forecast(head, output.detach(), observed, data, starts)
            assert prediction.shape == (2, data.n_vars, 96)
            prediction.square().mean().backward()
            assert all(p.grad is None for p in model.parameters())
            optimizer.step()
            assert state_sha(model) == state_before
            record = dict(model=args.model, dataset=dataset, scene=scene, loss=float(loss),
                shape=list(output.shape), head_initial_sha256=initial_sha, freeze_verified=True,
                observed_preservation=True, empty_variable_and_empty_sample_finite=True,
                validation_windows=len(data.starts['validation']), test_windows=len(data.starts['test']),
                asset_status=data.manifest['status'], sampling='2_preflight_only' if args.model=='CSDI' else 'deterministic')
            records.append(record)
            print(json.dumps(record), flush=True)
            del model, head, repeated_head
    write_json(args.root / f'preflight_{args.model}.json', dict(passed=True, cases=records))


if __name__ == '__main__':
    main()
