"""Train the unchanged head on fully observed histories under the current recipe."""
from time import perf_counter
STARTED = perf_counter()
import json
from pathlib import Path
import random
import numpy as np
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches, prepare_p4_batch
from integrations.forecast_compare.resources import TrainingResources, selected_training_resources
from malr.data.patching import patchify_preserving_values
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog

ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007')

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, indent=2); f.write('\n')

def clean_batch(batch, protocol):
    # Use exactly the same windows, but normalize with all genuine past observations.
    history = torch.from_numpy(np.stack([protocol.standardized[r.input_start:r.input_end] for r in batch.refs]))
    return prepare_p4_batch(history, torch.ones_like(history, dtype=torch.bool), batch.forecast_target,
                           protocol.fallback, refs=batch.refs)

def predict(head, batch):
    validity = torch.ones_like(batch.observed_mask)
    p = patchify_preserving_values(batch.normalized_observed, batch.observed_mask, 16, 8, validity=validity)
    return head(p.values.masked_fill(~p.validity, 0), p.effective_mask, batch.query_mean,
                batch.query_std, patch_validity=p.validity)

@torch.no_grad()
def score(head, protocol, split, device):
    head.eval(); squared = absolute = count = windows = 0
    per_mse, per_mae, starts = [], [], []
    for raw in iter_p4_batches(protocol, split, batch_size=128):
        batch = clean_batch(raw, protocol).to(device)
        error = predict(head, batch) - batch.forecast_target
        assert torch.isfinite(error).all()
        squared += error.double().square().sum().item(); absolute += error.double().abs().sum().item()
        count += error.numel(); windows += len(batch.refs)
        per_mse.extend(error.double().square().mean((1,2)).cpu().tolist())
        per_mae.extend(error.double().abs().mean((1,2)).cpu().tolist())
        starts.extend(r.target_start for r in batch.refs)
    assert (windows,count) == (2688,1806336)
    return dict(mse=squared/count,mae=absolute/count,windows=windows,target_elements=count), (per_mse,per_mae,starts)

def main():
    import sys
    plan = json.loads((PARENT/'plan.json').read_text()); device = torch.device('cuda:1')
    torch.set_num_threads(4)
    output = ROOT/'clean_reference'
    if '--evaluate-only' in sys.argv:
        protocol = load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),
            protocol_config=Path(plan['protocol_config']),mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)
        settings = load_e23_settings(Path(plan['e23_config']),protocol=protocol)
        torch.manual_seed(2023)
        reference = build_forecast_model(protocol,settings,device=device); head = reference.predictor
        del reference
        evaluate(head,protocol,device,output)
        return
    output.mkdir(parents=True, exist_ok=False)
    with TrainingResources(output, device=device, batch_size=128, model='Clean-shared-PatchTST-seed2023', started_at=STARTED) as resources:
        protocol = load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),
            protocol_config=Path(plan['protocol_config']),mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)
        settings = load_e23_settings(Path(plan['e23_config']),protocol=protocol)
        random.seed(2023);np.random.seed(2023);torch.manual_seed(2023)
        reference = build_forecast_model(protocol,settings,device=device); head = reference.predictor
        assert parameter_hash(head) == plan['predictor_initial_sha256']
        assert sum(p.numel() for p in head.parameters()) == 35427
        del reference; torch.cuda.empty_cache()
        optimizer = torch.optim.Adam((p for p in head.parameters() if p.requires_grad),lr=1e-4,weight_decay=0.)
        best = {'mae':float('inf'),'mse':float('inf')}; selections={};updates=0
        with resources.stage('forecast'):
            for epoch in range(1,101):
                head.train(); loss_sum=0.; batches=0
                for raw in iter_p4_batches(protocol,'train',batch_size=128,epoch=epoch):
                    batch=clean_batch(raw,protocol).to(device);optimizer.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.mse_loss(predict(head,batch),batch.forecast_target)
                    assert torch.isfinite(loss);loss.backward();optimizer.step();updates+=1
                    loss_sum+=float(loss.detach());batches+=1
                assert batches==66
                validation,_=score(head,protocol,'validation',device)
                for metric in ['mae','mse']:
                    if validation[metric] < best[metric]:
                        best[metric]=validation[metric];checkpoint=output/f'head-{metric}-selected.pt'
                        torch.save(dict(model_state=head.state_dict(),epoch=epoch,validation=validation),checkpoint)
                        selections[metric]=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
                            best_epoch=epoch,best_validation=validation,selection_metric=metric)
                print(json.dumps(dict(epoch=epoch,train_mse=loss_sum/batches,validation=validation,
                                      best_epochs={k:v['best_epoch'] for k,v in selections.items()})),flush=True)
            assert updates==6600
            for metric in ['mae','mse']:
                record=dict(**selections[metric],seed=2023,epochs_completed=100,optimizer_updates=6600,
                    loss='forecast_mse_only',predictor_parameters=35427,predictor_trainable_parameters=35424,
                    predictor_initial_sha256=plan['predictor_initial_sha256'],source_csv_sha256=protocol.source_sha256,
                    forecast_padding='zero_invalid_v1',history_mask='fully_observed',window_statistics='all_true_history',
                    stage1='none',batch_size=128,learning_rate=1e-4,weight_decay=0.,test_metrics_seen=False)
                write(output/f'head-{metric}-selection.json',record)
            resources.selection_manifest=output/'head-mae-selection.json'
    evaluate(head,protocol,device,output)

def evaluate(head,protocol,device,output):
    for metric in ['mae','mse']:
        selection=output/f'head-{metric}-selection.json';record=json.loads(selection.read_text());seal=file_sha256(selection)
        assert file_sha256(Path(record['checkpoint']))==record['checkpoint_sha256']
        head.load_state_dict(torch.load(record['checkpoint'],map_location=device,weights_only=False)['model_state'])
        validation,_=score(head,protocol,'validation',device)
        assert all(abs(validation[k]-record['best_validation'][k])<1e-10 for k in ['mse','mae'])
        metrics,arrays=score(head,protocol,'test',device)
        path=output/f'test-{metric}-windows.npz';np.savez_compressed(path,mse=arrays[0],mae=arrays[1],target_start=arrays[2])
        write(output/f'test-{metric}.json',dict(metrics=metrics,validation_replay=validation,
            selection_manifest=str(selection),selection_sha256=seal,checkpoint_sha256=record['checkpoint_sha256'],
            windows_file=str(path),windows_sha256=file_sha256(path),missing_mask_groups='not applicable; fully observed history',
            training_resources=selected_training_resources([output/'head-mae-selection.json']),
            role='clean reference; not a guaranteed theoretical lower bound'))
        assert file_sha256(selection)==seal
        print(json.dumps(dict(event='clean_reference_complete',selection_metric=metric,**metrics)),flush=True)

if __name__=='__main__':main()
