"""Read-only migration acceptance: v79 weights, same validation predictions, frozen repair."""
import json, hashlib, os
from pathlib import Path
import torch
from dataclasses import replace
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, load_stage2_selection, _new_memory, _key_sha256, _validation_forecast_metrics
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer, stage2_training_mode
from integrations.e24.run import inherited_arguments
from malr.migration import MigrationCatalog

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v79_v69_global_level_four_pureMSE_20261007'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    frozen=json.loads((PARENT/'frozen-definition.json').read_text())
    checked=0
    for rel,digest in frozen['sources'].items():
        if rel.startswith(('src/','integrations/')) and rel!='integrations/e23/config.py':
            assert sha(ROOT/rel)==digest, rel
            checked+=1
    plan=json.loads((PARENT/'plan.json').read_text()); rows=[]
    device=torch.device('cuda:1')
    # The inherited E22 migration has paired E22/E23 aliases: keep E23 lineage selection.
    catalog=MigrationCatalog.load(ROOT/'migration/paper_catalog.jsonl',project='E23')
    for scene in plan['scenarios']:
        os.environ.pop('MALR_MISSING_SCENARIO',None)
        if scene['scenario_manifest']: os.environ['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
        protocol=load_e23_p4_protocol(catalog=catalog,protocol_config=ROOT/'configs/etth1_seq96_pred96.yaml',mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)
        assert protocol.mask_sha256['validation']==scene['validation_mask_sha256']
        settings=load_e23_settings(ROOT/'configs/e24_v79.yaml',protocol=protocol)
        torch.manual_seed(2023)
        model=build_forecast_model(protocol,settings,device=device)
        assert parameter_hash(model.predictor)==plan['predictor_initial_sha256']
        assert sum(p.numel() for p in model.repair.parameters())==424238
        assert sum(p.numel() for p in model.predictor.parameters())==35427
        selection=PARENT/scene['name']/'E23-v79/stage2/runs/e23-stage2.json'
        record=json.loads(selection.read_text())
        assert sha(protocol.config_path)==sha(record['protocol_config_path'])
        # Check old weights with their original path identity; new E24 runs use the new alias.
        protocol=replace(protocol,config_path=Path(record['protocol_config_path']))
        payload=load_stage2_selection(selection,protocol,settings,device=device)
        model.load_state_dict(payload['model_state'],strict=True)
        model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot())
        optimizer=build_stage2_optimizer(model,mode='frozen_v1',learning_rate=1e-4,weight_decay=0)
        trainable={id(p) for g in optimizer.param_groups for p in g['params']}
        assert trainable=={id(p) for p in model.predictor.parameters() if p.requires_grad}
        stage2_training_mode(model,'frozen_v1')
        assert all(not module.training for module in model.repair.modules())
        assert not any(p.requires_grad or p.grad is not None for p in model.repair.parameters())
        before=parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict())
        metrics=_validation_forecast_metrics(model,protocol,settings,device=device)
        assert all(abs(metrics[k]-record['best_val_'+k])<1e-6 for k in ['mse','mae'])
        assert before==(parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.repair.historical.retrieval.state_dict()))
        rows.append(dict(scene=scene['name'],validation_replay=metrics,checkpoint_sha256=record['checkpoint_sha256'],repair_frozen=True,optimizer_only_predictor=True,state_unchanged=True))
        print(json.dumps(rows[-1]),flush=True)
        del payload,model,memory,optimizer
        torch.cuda.empty_cache()
    # Reject violations before importing or entering training.
    for flag,value in [('--repair-update-policy','joint'),('--forecast-loss','mse_plus_0p2_mae'),('--forecast-mixer','patch_attention_v1')]:
        try: inherited_arguments(['train-forecast',flag,value])
        except SystemExit: pass
        else: raise AssertionError('protocol guard did not reject '+flag)
    profiles=json.loads((ROOT/'configs/baselines/etth1_p4_author_20261002.json').read_text())
    assert all((ROOT/cfg['root']).resolve().parent==ROOT.parent for cfg in profiles['models'].values())
    assert all((ROOT/cfg['root']).is_dir() for cfg in profiles['models'].values())
    result=dict(status='passed',training_performed=False,source_files_identical=checked,scenarios=rows,baseline_roots_valid=True,stage2_policy_guards_passed=True)
    out=ROOT/'migration/e24_inheritance_acceptance_20261007.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='passed',scenarios=len(rows),source_files_identical=checked)),flush=True)

if __name__=='__main__':main()
