"""Real bank/batch CUDA checks; does not train or evaluate new test results."""
import ast
import hashlib
import json
from pathlib import Path
import torch
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.report_e23_v33_trial import CONFIGS, ROOT, PREDICTOR_INITIAL_SHA
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_forecast_model, _new_memory, _key_sha256, _starts
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.round_supervision import supervised_round_objective
from integrations.forecast_compare.protocol import iter_p4_batches


def function_source(path, name):
    source = Path(path).read_text(); tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return ast.get_source_segment(source, node)


def main():
    root = ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v33_refreshed_rounds'
    baseline = Path('/home/xyw/paper/.e23-v32-preservation/E23')
    for path, name in [('integrations/e23/train.py','train_forecast'),
                       ('integrations/e23/train.py','build_forecast_model'),
                       ('src/malr/models/e23/model.py','E23ForecastModel')]:
        if function_source(ROOT/path, name) != function_source(baseline/path, name):
            raise ValueError('prediction head/interface/Stage2 function changed: '+name)
    source_hashes = json.loads((root/'baseline-source-files.json').read_text())
    head_files = [n for n in source_hashes if n.startswith('src/malr/vendor/patchtst/') or
                  (n.startswith('src/malr/models/') and
                   ('patchtst' in n.lower() or n.endswith('repaired_forecast.py')))]
    for name in head_files:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != source_hashes[name]:
            raise ValueError('predictor source changed: '+name)
    protocol = load_protocol(); device = torch.device('cuda:1')
    b = next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
    old = ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v32_prediction_preservation/A/stage1/runs/e23-stage1.json'
    old_record = json.loads(old.read_text()); payload = torch.load(old_record['checkpoint'],map_location=device,weights_only=False)
    rows = []
    for condition, config in CONFIGS.items():
        torch.manual_seed(2021)
        settings = load_e23_settings(config,protocol=protocol)
        model = build_forecast_model(protocol,settings,device=device)
        if parameter_hash(model.predictor) != PREDICTOR_INITIAL_SHA: raise ValueError('predictor initialization changed')
        repair = model.repair
        # Compatibility/preflight only: formal runs initialize from scratch.
        result = repair.load_state_dict(payload['model_state'],strict=False)
        if result.unexpected_keys or any(not n.startswith('iterative.update_gate.') for n in result.missing_keys):
            raise ValueError('unexpected base state compatibility failure')
        repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory = _new_memory(protocol); memory.load_state_dict(payload['memory_state'])
        repair.historical.refresh_snapshot(memory.snapshot())
        before = (_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash())
        out = repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,
            _starts(b,stride=8,patches=12,device=device),epoch=1,explore_variable=True,
            specialist_exposure=.2,specialist_seed=2021)
        loss, parts = supervised_round_objective(repair,out,b,torch.tensor(.1,device=device),retrieval_weight=.1)
        loss.backward()
        grads = [p.grad for p in repair.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads): raise ValueError('nonfinite CUDA gradients')
        if not torch.equal(out.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask]):
            raise ValueError('observed values changed')
        if not torch.equal(out.explicit.sequence_observed,b.observed_mask): raise ValueError('observation provenance changed')
        if before != (_key_sha256(repair.historical.retrieval.state_dict()),memory.snapshot_hash()):
            raise ValueError('historical key/bank changed')
        rows.append(dict(condition=condition,loss=loss.item(),batch_size=128,
            h_key_sha256=before[0],h_bank_sha256=before[1],finite_gradients=True,
            predictor_initial_sha256=parameter_hash(model.predictor),predictor_parameters=35427,
            repair_parameters=sum(p.numel() for p in repair.parameters()),
            formal_training_reuses_preflight_weights=False))
        del model,repair,memory,out,loss,grads
        torch.cuda.empty_cache()
    record=dict(cuda_preflight=rows,head_and_stage2_source_unchanged=True,head_files=head_files)
    (root/'cuda_preflight.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
