"""Small Stage1 screen: learn when to shrink coarse values towards visible guides.

Uses the same v97 prefix and training banks as v104/v106, never future targets.
The temporary feature hook is restricted to this CPU screening prototype.
"""
from pathlib import Path
from time import perf_counter
import copy
import hashlib
import json
import os
import torch
from torch import nn
from screen_v104_cpu import suffix, score
from malr.models.e23.recent_mean import RecentMeanProjector
from malr.models.e23.global_dense import GlobalDenseRefiner

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'v107_contextual_calibration_screen'
CANDIDATES = [('v107_A_original_loss', 0.0), ('v107_B_full_mae0p5', 0.5), ('v107_C_full_mae2', 2.0)]


class ContextualDenseRefiner(GlobalDenseRefiner):
    def __init__(self, variables=21):
        super().__init__(variables)
        self.guide_gate = nn.Linear(64, variables)
        nn.init.zeros_(self.guide_gate.weight)
        nn.init.zeros_(self.guide_gate.bias)

    def forward(self, normalized, explicit, mean, std):
        captured = []
        hook = self.output.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            additive = super().forward(normalized, explicit, mean, std)
        finally:
            hook.remove()
        assert len(captured) == 1
        valid = explicit.sequence_validity
        observed = explicit.sequence_observed & valid
        coarse = normalized * std[:, None] + mean[:, None]
        visible = torch.where(observed, explicit.observed_sequence * std[:, None] + mean[:, None], 0)
        index = torch.arange(normalized.shape[1], device=normalized.device)[None,:,None]
        left = torch.cummax(torch.where(observed, index, -1), 1).values
        right = torch.cummin(torch.where(observed, index, normalized.shape[1]).flip(1), 1).values.flip(1)
        lv = visible.gather(1, left.clamp_min(0))
        rv = visible.gather(1, right.clamp_max(normalized.shape[1]-1))
        both = (left >= 0) & (right < normalized.shape[1]) & (right > left)
        guide = torch.where(both, lv+(rv-lv)*(index-left)/(right-left).clamp_min(1),
                            torch.where(left >= 0, lv, rv))
        gate = torch.tanh(self.guide_gate(captured[0]))
        active = valid & ~observed & observed.any(1, keepdim=True)
        # Zero gate keeps the parent output exact. Positive gate can remove a
        # large false event; cross-variable temporal context determines the gate.
        correction = torch.where(active, gate*(guide-coarse.detach())/std[:,None], 0)
        return additive + correction


def build_refiner(initial_state):
    model = RecentMeanProjector(21, mode='global_dense_time_v4')
    model.global_dense_refiner = ContextualDenseRefiner(21)
    missing = model.load_state_dict(initial_state, strict=False)
    assert not missing.unexpected_keys
    assert set(missing.missing_keys) == {
        'global_dense_refiner.guide_gate.weight', 'global_dense_refiner.guide_gate.bias'}
    return model


def main():
    os.nice(10)
    torch.set_num_threads(4)
    assert not DEST.exists()
    DEST.mkdir()
    started = perf_counter()
    payload = torch.load(ROOT/'v97_cpu_screen_inputs.pt', map_location='cpu', weights_only=False)
    model = build_refiner(payload['initial_state'])
    original = RecentMeanProjector(21, mode='global_dense_time_v4')
    original.load_state_dict(payload['initial_state'], strict=True)
    original.eval(); model.eval()
    with torch.no_grad():
        for row in payload['train_rows'][0][:2] + payload['validation_rows'][:2]:
            before, _ = suffix(original, row)
            after, _ = suffix(model, row)
            assert torch.equal(before.normalized_sequence, after.normalized_sequence)
    initial = copy.deepcopy(model.state_dict())
    metrics, _ = score(model, payload['validation_rows'])
    assert metrics == payload['metadata']['initial_metrics']
    design = dict(scope='CPU limited-bank Stage1 screen only, not a formal checkpoint; no future targets or test masks',
        source_metadata=payload['metadata'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        zero_gate_initial_outputs_exact=True, visible_values_exact=True,
        additional_parameters=21*65, total_refiner_parameters=sum(p.numel() for p in model.parameters()),
        learned_gate='tanh(context_projection) * (visible_boundary_guide - coarse_repair)',
        no_target_visible_values_means_no_guide_gate=True,
        stages_affected='Stage1 output only; unchanged predictor, pure MSE frozen Stage2',
        candidates=CANDIDATES, epochs=20, batch_size=128, optimizer_updates=160, lr=1e-4)
    (DEST/'design.json').write_text(json.dumps(design, indent=2))
    results = []
    for name, mae_weight in CANDIDATES:
        model.load_state_dict(initial, strict=True)
        model.requires_grad_(True)
        torch.set_rng_state(payload['initial_rng'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        history, choices = [], {}
        candidate_start = perf_counter()
        for epoch in range(1,21):
            model.train()
            for row in payload['train_rows'][(epoch-1)%2]:
                optimizer.zero_grad(set_to_none=True)
                out, error = suffix(model, row)
                h = ~row[-1] & out.sequence_validity
                e, tail = error[h], error[:,-32:][h[:,-32:]]
                loss = e.square().mean()+mae_weight*e.abs().mean()+.25*tail.square().mean()+.05*tail.abs().mean()
                assert torch.isfinite(loss)
                loss.backward()
                assert torch.isfinite(model.global_dense_refiner.guide_gate.weight.grad).all()
                if epoch == 1:
                    assert model.global_dense_refiner.guide_gate.weight.grad.abs().sum() > 0
                optimizer.step()
            metrics, variables = score(model, payload['validation_rows'])
            history.append(dict(epoch=epoch, **metrics))
            for criterion in ('mse','mae'):
                if criterion not in choices or metrics['full_'+criterion] < choices[criterion]['metrics']['full_'+criterion]:
                    choices[criterion] = dict(epoch=epoch, metrics=metrics, per_variable=variables)
                    torch.save(dict(refiner_state=copy.deepcopy(model.state_dict()), **choices[criterion], source=design['scope']),
                               DEST/(name+'_'+criterion+'_screen.pt'))
            (DEST/(name+'_history.json')).write_text(json.dumps(history, indent=2))
            print(json.dumps(dict(candidate=name, epoch=epoch, **metrics)), flush=True)
        results.append(dict(candidate=name, full_mae_weight=mae_weight,
                            mse_selected=choices['mse'], mae_selected=choices['mae'], seconds=perf_counter()-candidate_start))
        (DEST/'results.json').write_text(json.dumps(dict(candidates=results, total_seconds=perf_counter()-started,
                                                        scope=design['scope']), indent=2))
    print('SCREEN COMPLETED', flush=True)


if __name__ == '__main__':
    main()
