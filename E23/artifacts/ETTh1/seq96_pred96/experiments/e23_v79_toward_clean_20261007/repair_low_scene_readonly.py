"""Complete the missing low-rate repair audit using saved weights only."""
import importlib.util
import json
import os
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, load_stage2_selection, _starts, _new_memory, _key_sha256
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.t1_p4 import t1_inputs, t1_clean_history
from malr.migration import MigrationCatalog
from malr.evaluation.report import file_sha256

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'e23_t1_pure_mse_frozen_four_scenarios_20261007'
V79 = ROOT.parent / 'e23_v79_v69_global_level_four_pureMSE_20261007'

def accumulate(totals, error, hidden):
    for label, e, h in [('all_hidden', error, hidden), ('last32_hidden', error[:, -32:], hidden[:, -32:])]:
        r = e[h].double()
        assert torch.isfinite(r).all()
        totals[label][0] += r.square().sum().item()
        totals[label][1] += r.abs().sum().item()
        totals[label][2] += r.numel()

def result(totals):
    return {k: dict(mse=s/n, mae=a/n, hidden_coordinates=n) for k, (s, a, n) in totals.items()}

def main():
    torch.set_num_threads(4)
    plan = json.loads((V79 / 'plan.json').read_text())
    scene = plan['scenarios'][0]
    assert scene['name'] == 'sr010_bl8'
    if scene['scenario_manifest']:
        os.environ['MALR_MISSING_SCENARIO'] = scene['scenario_manifest']
    else:
        os.environ.pop('MALR_MISSING_SCENARIO', None)
    device = torch.device('cuda:1')
    protocol = load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=Path(plan['protocol_config']), mask_artifacts=Path(plan['mask_artifacts']), training_seed=2023)
    settings = load_e23_settings(Path(plan['e23_config']), protocol=protocol)
    selection = V79 / scene['name'] / 'E23-v79/stage2/runs/e23-stage2.json'
    record = json.loads(selection.read_text())
    selection_sha = file_sha256(selection)
    payload = load_stage2_selection(selection, protocol, settings, device=device)
    model = build_forecast_model(protocol, settings, device=device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(protocol)
    memory.load_state_dict(payload['memory_state'])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    model.eval().requires_grad_(False)
    before = parameter_hash(model), memory.snapshot_hash(), _key_sha256(model.repair.historical.retrieval.state_dict())
    totals = {k: [0., 0., 0] for k in ['all_hidden', 'last32_hidden']}
    del payload
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol, 'validation', batch_size=128):
            b = cpu.to(device)
            starts = _starts(b, stride=model.repair.stride, patches=model.repair.patch_count, device=device)
            out = model.repair(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts).explicit
            assert torch.equal(out.normalized_sequence[b.observed_mask], b.normalized_observed[b.observed_mask])
            accumulate(totals, (out.normalized_sequence - b.normalized_target) * b.query_std[:, None],
                ~b.observed_mask & out.sequence_validity)
    assert before == (parameter_hash(model), memory.snapshot_hash(), _key_sha256(model.repair.historical.retrieval.state_dict()))
    assert file_sha256(selection) == selection_sha
    assert file_sha256(Path(record['checkpoint'])) == record['checkpoint_sha256']
    own = result(totals)
    stage1 = json.loads((V79 / scene['name'] / 'E23-v79/stage1/runs/e23-stage1.json').read_text())
    assert abs(own['all_hidden']['mse'] - stage1['validation_repair_metrics']['global_mse']) < 1e-6
    del model, memory, out, b
    torch.cuda.empty_cache()
    tplan = json.loads((BASE / 't1-plan.json').read_text())
    ts = tplan['scenarios'][0]
    assert ts['name'] == scene['name']
    path = Path(tplan['work']) / 'scripts/run_t1_frozen_stage2.py'
    import integrations.forecast_compare as compare_package
    compare_package.__path__.append(str(Path(tplan['work']) / 'integrations/forecast_compare'))
    spec = importlib.util.spec_from_file_location('locked_t1', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tp, _, parent, tm = m.setup(tplan, ts)
    assert tp.mask_sha256['validation'] == protocol.mask_sha256['validation']
    tm.eval()
    initial = m.parameter_hash(tm.imputer)
    totals = {k: [0., 0., 0] for k in ['all_hidden', 'last32_hidden']}
    with torch.inference_mode():
        for cpu in iter_p4_batches(tp, 'validation', batch_size=128):
            b = cpu.to(tplan['device'])
            values, mask = t1_inputs(b)
            candidate = tm.imputer(values, mask=mask).float()
            accumulate(totals, candidate - t1_clean_history(b), ~b.observed_mask)
    m.assert_frozen(tm, initial)
    t1 = result(totals)
    assert all(t1[k]['hidden_coordinates'] == own[k]['hidden_coordinates'] for k in t1)
    row = dict(scene=scene['name'], split='fixed_validation', metric_space='train_standardized',
        validation_mask_sha256=protocol.mask_sha256['validation'], t1_stage1_checkpoint_sha256=parent['checkpoint_sha256'],
        v79_selection_sha256=selection_sha, t1=t1, v79=own, repair_state_unchanged=True, training_performed=False)
    with (ROOT / 't1-v79-repair-low-scene-readonly.json').open('x') as f:
        json.dump(row, f, indent=2)
        f.write('\n')
    print(json.dumps(row), flush=True)

if __name__ == '__main__':
    main()
