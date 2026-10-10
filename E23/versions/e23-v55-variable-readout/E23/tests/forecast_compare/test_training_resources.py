"""Resource reports must count all stages and preserve missing measurements."""

import json

import pytest
import torch


def test_cpu_run_records_preparation_and_both_stages(tmp_path):
    from integrations.forecast_compare.resources import TrainingResources

    with TrainingResources(tmp_path, device='cpu', batch_size=128, model='example') as run:
        with run.stage('repair'):
            model = torch.nn.Linear(3, 2)
            model(torch.ones(4, 3)).square().mean().backward()
        with run.stage('forecast'):
            model(torch.ones(4, 3)).sum().backward()
    report = json.loads(run.report_path.read_text())
    assert report['status'] == 'completed'
    assert set(report['stages']) == {'repair', 'forecast'}
    assert report['full_workflow_seconds'] >= report['training_stages_seconds'] > 0
    assert report['preparation_and_reporting_seconds'] >= 0
    assert report['peak_allocated_bytes'] is None
    assert report['peak_reserved_bytes'] is None
    assert report['device']['type'] == 'cpu'


def test_failure_is_not_reported_as_a_completed_training_run(tmp_path):
    from integrations.forecast_compare.resources import TrainingResources

    run = TrainingResources(tmp_path, device='cpu', batch_size=128, model='example')
    with pytest.raises(RuntimeError, match='training failed'):
        with run:
            with run.stage('forecast'):
                raise RuntimeError('training failed')
    report = json.loads(run.report_path.read_text())
    assert report['status'] == 'failed'
    assert report['stages']['forecast']['status'] == 'failed'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA measurement requires a GPU')
def test_cuda_peak_includes_optimizer_state_and_keeps_largest_stage(tmp_path):
    from integrations.forecast_compare.resources import TrainingResources

    with TrainingResources(tmp_path, device='cuda:0', batch_size=4, model='example') as run:
        with run.stage('repair'):
            model = torch.nn.Linear(512, 512, device='cuda:0')
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.zero_grad(set_to_none=True)
            model(torch.ones(4, 512, device='cuda:0')).square().mean().backward()
            optimizer.step()
        del optimizer, model
        with run.stage('forecast'):
            tiny = torch.ones(1, device='cuda:0')
    report = json.loads(run.report_path.read_text())
    assert report['peak_allocated_bytes'] >= report['stages']['repair']['peak_allocated_bytes'] > 3_000_000
    assert report['peak_reserved_bytes'] >= report['peak_allocated_bytes']
    assert report['peak_allocated_bytes'] >= report['stages']['forecast']['peak_allocated_bytes']
    del tiny


def test_summary_sums_stage_time_and_uses_maximum_memory():
    from integrations.forecast_compare.resources import summarize_resources

    records = [
        {'status': 'completed', 'mode': 'training', 'full_workflow_seconds': 12.,
         'training_stages_seconds': 10., 'peak_allocated_bytes': 1024**3,
         'peak_reserved_bytes': 2 * 1024**3, 'device': {'name': 'GPU'},
         'batch_size': 128},
        {'status': 'completed', 'mode': 'training', 'full_workflow_seconds': 25.,
         'training_stages_seconds': 20., 'peak_allocated_bytes': 3 * 1024**3,
         'peak_reserved_bytes': 4 * 1024**3, 'device': {'name': 'GPU'},
         'batch_size': 128},
    ]
    summary = summarize_resources(records)
    assert summary['full_workflow_seconds'] == 37.
    assert summary['training_stages_seconds'] == 30.
    assert summary['peak_allocated_gib'] == 3.
    assert summary['peak_reserved_gib'] == 4.


def test_summary_does_not_turn_partial_or_failed_measurements_into_full_metrics():
    from integrations.forecast_compare.resources import summarize_resources

    measured = {'status': 'completed', 'mode': 'training', 'full_workflow_seconds': 3.,
                'training_stages_seconds': 2., 'peak_allocated_bytes': 1024,
                'peak_reserved_bytes': 2048, 'device': {'name': 'GPU'}, 'batch_size': 128}
    missing = {**measured, 'full_workflow_seconds': None, 'peak_allocated_bytes': None,
               'peak_reserved_bytes': None}
    summary = summarize_resources([measured, missing])
    assert summary['full_workflow_seconds'] is None
    assert summary['peak_allocated_gib'] is None
    assert summary['peak_reserved_gib'] is None
    with pytest.raises(ValueError, match='completed training'):
        summarize_resources([{**measured, 'status': 'failed'}])


def test_selected_summary_keeps_legacy_time_without_inventing_full_time(tmp_path):
    from integrations.forecast_compare.resources import selected_training_resources

    paths = [tmp_path / 'repair.json', tmp_path / 'forecast.json']
    paths[0].write_text(json.dumps({'training_seconds': 10., 'epochs_completed': 100}))
    paths[1].write_text(json.dumps({'training_seconds': 20., 'epochs_completed': 100}))
    report = selected_training_resources(paths)
    assert report['legacy_training_loop_seconds'] == 30.
    assert report['stage1_legacy_loop_seconds'] == 10.
    assert report['stage2_legacy_loop_seconds'] == 20.
    assert report['full_workflow_seconds'] is None
    assert report['peak_allocated_gib'] is None


def test_selected_summary_rejects_resources_from_a_different_checkpoint(tmp_path):
    from integrations.forecast_compare.resources import selected_training_resources

    path = tmp_path / 'selected.json'
    path.write_text(json.dumps({'training_seconds': 10., 'epochs_completed': 100}))
    sidecar = tmp_path / 'resources' / 'wrong.json'
    sidecar.parent.mkdir()
    sidecar.write_text(json.dumps({'selection_manifest_sha256': 'wrong',
                                   'status': 'completed', 'mode': 'training',
                                   'full_workflow_seconds': 999.}))
    report = selected_training_resources([path])
    assert report['full_workflow_seconds'] is None
    assert report['resource_record_paths'] == []
