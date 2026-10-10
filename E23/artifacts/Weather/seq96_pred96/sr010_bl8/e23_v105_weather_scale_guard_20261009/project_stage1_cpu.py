"""Add the v105 output operator to selected v97 weights, without optimization."""
from time import perf_counter
ENTRY_STARTED = perf_counter()
from pathlib import Path
import json
import os
import copy
import torch
from weather_common import ROOT, protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (build_repair_model, load_stage1_selection, _new_memory,
    _validation_repair_metrics, _paired_payload, _atomic_torch_save, _atomic_json,
    _file_sha256, _identity, _revision, _memory_state_sha256, _key_sha256)
from integrations.forecast_compare.resources import TrainingResources


def main():
    os.nice(10)
    torch.set_num_threads(4)
    plan = json.loads((ROOT/'plan.json').read_text())
    output = ROOT/'E23-v105/sr010_bl8/stage1'
    assert not output.exists()
    device = torch.device('cpu')
    with TrainingResources(output, device=device, batch_size=128, model='E23-v105-output-projection',
                           started_at=ENTRY_STARTED) as resources:
        with resources.stage('stage1_output_projection_and_validation'):
            protocol = protocol_for(plan, plan['scenarios'][0])
            parent_settings = load_e23_settings(Path(plan['parent_settings']), protocol=protocol,
                                                allow_variable_scaling=True)
            settings = load_e23_settings(ROOT/'v105_weather.yaml', protocol=protocol,
                                        allow_variable_scaling=True)
            parent_selection = Path(plan['parent_selection'])
            parent_record = json.loads(parent_selection.read_text())
            parent = load_stage1_selection(parent_selection, protocol, parent_settings, device=device)
            model = build_repair_model(protocol, settings, device=device)
            model.load_state_dict(parent['model_state'], strict=True)
            assert all(torch.equal(v, parent['model_state'][k]) for k, v in model.state_dict().items())
            model.historical.retrieval.load_state_dict(parent['retrieval_state'])
            memory = _new_memory(protocol)
            memory.load_state_dict(parent['memory_state'])
            # The legacy bank loader re-normalizes float64 clean keys. Preserve
            # the exact parent's serialization, so baseline and candidate each
            # load it once and use the same runtime bank. Re-saving the loaded
            # bank would add a second normalization on the candidate only.
            runtime_bank_hash = _memory_state_sha256(memory.state_dict())
            assert _memory_state_sha256(parent['memory_state']) == parent['memory_sha256']
            model.historical.refresh_snapshot(memory.snapshot())
            model.eval().requires_grad_(False)
            assert model.recent_mean_projector.normalized_output_guard
            metrics = _validation_repair_metrics(model, protocol, settings, device=device)
            paired = _paired_payload(model, memory, epoch=parent['epoch'],
                                     validation_mse=metrics['global_mse'], settings=settings,
                                     validation_metrics=metrics)
            assert all(torch.equal(v, parent['model_state'][k]) for k, v in paired['model_state'].items())
            assert paired['memory_sha256'] == runtime_bank_hash
            assert paired['key_sha256'] == parent['key_sha256']
            paired['memory_state'] = copy.deepcopy(parent['memory_state'])
            paired['retrieval_state'] = copy.deepcopy(parent['retrieval_state'])
            paired['memory_sha256'] = _memory_state_sha256(paired['memory_state'])
            paired['key_sha256'] = _key_sha256(paired['retrieval_state'])
            assert paired['memory_sha256'] == parent['memory_sha256']
            assert paired['key_sha256'] == parent['key_sha256']
            reloaded = _new_memory(protocol)
            reloaded.load_state_dict(paired['memory_state'])
            assert _memory_state_sha256(reloaded.state_dict()) == runtime_bank_hash
            checkpoint = output/'checkpoints/e23-stage1.pt'
            manifest = output/'runs/e23-stage1.json'
            _atomic_torch_save(paired, checkpoint)
            record = dict(stage='repair', best_epoch=int(parent['epoch']), epochs_completed=0,
                          best_val_repair_mse=metrics['global_mse'], validation_repair_metrics=metrics,
                          checkpoint=str(checkpoint), checkpoint_sha256=_file_sha256(checkpoint),
                          memory_sha256=paired['memory_sha256'], key_sha256=paired['key_sha256'],
                          code_revision=_revision(), training_seconds=perf_counter()-ENTRY_STARTED,
                          selection='inherits v97 fixed-validation-MSE selected weights; '
                                    'fixed, validation-only numerical output operator; no optimizer',
                          parent_selection=str(parent_selection),
                          parent_checkpoint_sha256=parent_record['checkpoint_sha256'],
                          parent_epochs_completed=parent_record['epochs_completed'],
                          total_stage1_epochs=parent_record['total_stage1_epochs'],
                          stage1_weights_reused=True, optimizer_updates=0,
                          additional_training_epochs=0, stage1_aux_forecast='none',
                          all_repair_parameters_exact_equal_to_parent=True,
                          bank_and_key_exact_equal_to_parent=True, trainable_parameters=0,
                          runtime_bank_exact_equal_to_parent_after_one_load=True,
                          runtime_bank_sha256=runtime_bank_hash,
                          projection_only=True,
                          source_freeze_manifest_sha256=_file_sha256(ROOT/'source_freeze.json'),
                          **_identity(protocol, settings))
            _atomic_json(record, manifest)
            # Verify strict loading of the paired bank/key and guard identity.
            load_stage1_selection(manifest, protocol, settings, device=device)
            resources.selection_manifest = manifest
    print(json.dumps(dict(status='completed', selection=str(manifest), metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
