"""Validation-only stronger MAE screens using the existing v97 prefix cache."""
from pathlib import Path
from time import perf_counter
import copy
import hashlib
import json
import os
import torch
from screen_v104_cpu import suffix, score
from malr.models.e23.recent_mean import RecentMeanProjector

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'v106_stronger_mae_screen'
CANDIDATES = [
    ('v106_A_mse_plus_mae2', 1.0, 2.0, 0.25, 0.05),
    ('v106_B_mse_plus_mae5', 1.0, 5.0, 0.25, 0.05),
    ('v106_C_mae_only', 0.0, 1.0, 0.0, 0.25),
]


def main():
    os.nice(10)
    torch.set_num_threads(4)
    assert not DEST.exists(), DEST
    DEST.mkdir()
    started = perf_counter()
    cache_path = ROOT / 'v97_cpu_screen_inputs.pt'
    payload = torch.load(cache_path, map_location='cpu', weights_only=False)
    refiner = RecentMeanProjector(21, mode='global_dense_time_v4')
    refiner.load_state_dict(payload['initial_state'], strict=True)
    initial, _ = score(refiner, payload['validation_rows'])
    assert initial == payload['metadata']['initial_metrics'], (initial, payload['metadata']['initial_metrics'])
    source_record = dict(
        source='v97 initial refiner with exact cached frozen-prefix inputs',
        source_cache=str(cache_path), source_cache_bytes=cache_path.stat().st_size,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_input_metadata=payload['metadata'],
        candidates=[dict(name=n, full_mse_weight=m, full_mae_weight=a,
                         tail_mse_weight=tm, tail_mae_weight=ta) for n,m,a,tm,ta in CANDIDATES],
        epochs=20, optimizer_updates=160, batch_size=128, learning_rate=1e-4,
        training_mask_banks_repeated=True, stage2_modified=False,
        checkpoint_primary_for_screen='fixed-validation full hidden MSE; MAE saved separately',
        scope='CPU objective screening only; requires separate formal full training; no future targets or test masks',
    )
    (DEST / 'design.json').write_text(json.dumps(source_record, indent=2))
    results = []
    for name, m, a, tm, ta in CANDIDATES:
        refiner.load_state_dict(payload['initial_state'], strict=True)
        refiner.requires_grad_(True)
        torch.set_rng_state(payload['initial_rng'])
        optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-4, weight_decay=0)
        choices = {}
        history = []
        candidate_start = perf_counter()
        for epoch in range(1,21):
            refiner.train()
            for row in payload['train_rows'][(epoch-1) % 2]:
                optimizer.zero_grad(set_to_none=True)
                out, error = suffix(refiner, row)
                hidden = ~row[-1] & out.sequence_validity
                e, tail = error[hidden], error[:,-32:][hidden[:,-32:]]
                loss = m*e.square().mean() + a*e.abs().mean() + tm*tail.square().mean() + ta*tail.abs().mean()
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
            metrics, variables = score(refiner, payload['validation_rows'])
            history.append(dict(epoch=epoch, **metrics))
            for criterion in ('mse','mae'):
                if criterion not in choices or metrics['full_'+criterion] < choices[criterion]['metrics']['full_'+criterion]:
                    choices[criterion] = dict(epoch=epoch, metrics=metrics, per_variable=variables)
                    torch.save(dict(refiner_state=copy.deepcopy(refiner.state_dict()),
                                    **choices[criterion], source=source_record['scope']),
                               DEST / (name+'_'+criterion+'_screen.pt'))
            (DEST / (name+'_history.json')).write_text(json.dumps(history, indent=2))
            print(json.dumps(dict(candidate=name, epoch=epoch, **metrics)), flush=True)
        results.append(dict(candidate=name, mse_selected=choices['mse'], mae_selected=choices['mae'],
                            seconds=perf_counter()-candidate_start))
        (DEST / 'results.json').write_text(json.dumps(dict(candidates=results,
            total_seconds=perf_counter()-started, scope=source_record['scope']), indent=2))
    print('SCREEN COMPLETED', flush=True)


if __name__ == '__main__':
    main()
