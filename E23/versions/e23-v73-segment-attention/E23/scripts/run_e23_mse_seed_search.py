"""Predeclared validation-only geometry/budget/seed search with a fixed cached T1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime,timezone
import yaml

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v42_mse_updates_seed_search_20261004')
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
V40=ROOT.parent/'e23_v40_local_point_refinement_seed502_20261004'
MASKS=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def sources():
    files=[WORK/'pyproject.toml']
    for folder in ['src','integrations','scripts','configs','tests']:
        files.extend(p for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    return {str(p.relative_to(WORK)):digest(p) for p in sorted(files)}
def selected(cell,metric='mae'):return read(Path(cell)/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json'))
def rank(cell):
    r=selected(cell,'mse');return r['best_val_mse'],r['best_val_mae']


def main():
    assert read(ROOT/'preflight.json')['status']=='passed'
    plan=read(ROOT/'plan.json')
    definition=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=digest(ROOT/'plan.json'),
        sources=sources(),protected=read(ROOT/'protected-existing-files.json'),T1=read(FORMAL/'seed_502/T1/evaluation/twenty-test-draws.json'),
        T1_evaluation_sha256=digest(FORMAL/'seed_502/T1/evaluation/twenty-test-draws.json'),test_bank_sha256=digest(FORMAL/'test_masks/manifest.json'))
    write(ROOT/'matrix-definition.json',definition)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as archive:
        for relative in definition['sources']:archive.add(WORK/relative,arcname=relative,recursive=False)
    env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    def verify():
        assert sources()==definition['sources'],'Source changed'
        for p,h in definition['protected'].items():assert digest(p)==h,p
        assert digest(ROOT/'plan.json')==definition['plan_sha256']
        assert digest(FORMAL/'test_masks/manifest.json')==definition['test_bank_sha256']
        assert digest(FORMAL/'seed_502/T1/evaluation/twenty-test-draws.json')==definition['T1_evaluation_sha256']
    def run(command,log):
        verify();print(json.dumps(dict(event='start',log=str(log),utc=datetime.now(timezone.utc).isoformat())),flush=True)
        with log.open('x') as f:subprocess.run(command,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        verify()
    def score(cell):
        chosen=selected(cell);trial=read(cell/'trial-definition.json')
        results={}
        for metric in ['mae','mse']:
            manifest=cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json')
            command=[sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B','--model-label',trial['label']+'-select-'+metric,
                '--training-seed',str(trial['seed']),'--selection',str(manifest),'--e23-config',trial['config'],
                '--test-mask-bank',str(FORMAL/'test_masks/manifest.json'),'--output',str(cell/f'evaluation_{metric}'),'--device','cuda:1']
            write(cell/f'evaluation-command-{metric}.json',command);run(command,cell/f'test-{metric}.log')
            results[metric]=read(cell/f'evaluation_{metric}/twenty-test-draws.json')
        write(cell/'completed-audit.json',dict(status='complete',trial=trial['label'],test_mean_by_selection={k:d['test_mean'] for k,d in results.items()},
            validation=read(cell/'validation-lock-before-tests.json'),training_resources=results['mae']['training_resources'],
            downstream_protocol='t1_mse_v1',new_T1_stage1_training=False))
        print(json.dumps(dict(event='scored',cell=str(cell),test_means={k:d['test_mean'] for k,d in results.items()})),flush=True)
    def trial(name,label,seed,source_config,stage1=None,batch=128,epochs=100,immediate_test=False):
        cell=ROOT/name;cell.mkdir(exist_ok=False)
        if stage1 is None:
            document=yaml.safe_load(Path(source_config).read_text());document['training'].update(seed=seed,stage1_epochs=epochs)
            config=cell/'e23.yaml';config.write_text(yaml.safe_dump(document,sort_keys=False))
            stage1=cell/'stage1/runs/e23-stage1.json'
            stage1_reused=False
        else:
            config=Path(source_config);stage1=Path(stage1);stage1_reused=True
        definition_trial=dict(name=name,label=label,seed=seed,config=str(config.resolve()),config_sha256=digest(config),
            stage1=str(stage1.resolve()),stage1_reused=stage1_reused,repair_batch=batch,repair_epochs=epochs,
            downstream_protocol='t1_mse_v1',validation_only_selection=True)
        write(cell/'trial-definition.json',definition_trial)
        common=['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
            '--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(MASKS),
            '--e23-config',str(config.resolve()),'--training-seed',str(seed),'--device','cuda:1']
        commands=[]
        if not stage1_reused:
            commands.append([sys.executable,'-m','integrations.e23.run','train-repair',*common,
                '--repair-batch-size',str(batch),'--output',str(cell/'stage1')])
        commands.append([sys.executable,'-m','integrations.e23.run','train-forecast',*common,
            '--stage1-manifest',str(stage1),'--common-stage2','--selection-metric','mae','--forecast-loss','mse',
            '--repair-retention','none','--output',str(cell/'stage2')])
        write(cell/'training-commands.json',commands)
        for step,command in enumerate(commands):run(command,cell/f'train_{step}.log')
        assert digest(config)==definition_trial['config_sha256']
        chosen=selected(cell)
        assert chosen['loss']=='forecast_mse_only' and chosen.get('repair_retention','none')=='none'
        assert chosen['downstream_protocol']=='t1_mse_v1' and chosen['epochs_completed']==100
        write(cell/'validation-lock-before-tests.json',dict(choices={metric:dict(mse=selected(cell,metric)['best_val_mse'],
            mae=selected(cell,metric)['best_val_mae'],checkpoint_sha256=selected(cell,metric)['checkpoint_sha256'],
            checkpoint_epoch=selected(cell,metric)['best_epoch']) for metric in ['mae','mse']},test_metrics_seen=False))
        if immediate_test:score(cell)
        return cell
    t1=ROOT/'T1_seed502';t1.mkdir()
    command=[sys.executable,'scripts/train_t1_dual_stage2.py','--profiles',str(FORMAL/'configs/author_seed_502.json'),
        '--stage1',str(FORMAL/'seed_502/T1/training/repair-selection.json'),'--output',str(t1/'training')]
    write(t1/'training-command.json',command);run(command,t1/'training.log')
    for metric in ['mae','mse']:
        command=[sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','T1','--model-label','T1-select-'+metric,
            '--training-seed','502','--profiles',str(FORMAL/'configs/author_seed_502.json'),
            '--selection',str(t1/'training'/('forecast-selection.json' if metric=='mae' else 'forecast-mse-selection.json')),
            '--test-mask-bank',str(FORMAL/'test_masks/manifest.json'),'--output',str(t1/f'evaluation_{metric}'),'--device','cuda:1']
        write(t1/f'test-command-{metric}.json',command);run(command,t1/f'test-{metric}.log')
    write(t1/'completed-audit.json',dict(status='complete',Stage1_reused=True,training_seed=502,dual_checkpoint_selection=True))
    base_config=FORMAL/'configs/e23_seed_502.yaml';tail_config=V40/'tail32/e23_seed502.yaml'
    base=trial('aligned_v31_seed502','E23-v42-aligned-v31',502,base_config,
        FORMAL/'seed_502/E23-v31-B/stage1/runs/e23-stage1.json',immediate_test=True)
    tail=trial('aligned_tail32_seed502','E23-v42-aligned-tail32',502,tail_config,
        V40/'tail32/stage1/runs/e23-stage1.json',immediate_test=True)
    geometry_cell=min([base,tail],key=rank);geometry='v31' if geometry_cell==base else 'tail32'
    write(ROOT/'geometry-validation-choice.json',dict(selected=geometry,cell=str(geometry_cell),
        ranks={str(c):rank(c) for c in [base,tail]},test_used_for_choice=False))
    source_config=read(geometry_cell/'trial-definition.json')['config']
    high=trial('updates52800_seed502','E23-v42-updates52800',502,source_config,batch=32,epochs=200,immediate_test=True)
    winning=min([geometry_cell,high],key=rank);high_selected=winning==high
    write(ROOT/'budget-validation-choice.json',dict(selected='batch32_epoch200' if high_selected else 'batch128_epoch100',
        geometry=geometry,cell=str(winning),ranks={str(c):rank(c) for c in [geometry_cell,high]},test_used_for_choice=False))
    seed_cells={502:winning}
    for seed in plan['phase3']['seeds']:
        if seed==502:continue
        cached=None;config=source_config
        if not high_selected and geometry=='v31':
            if seed==2021:
                config=WORK/'configs/e23_multi_expert_p4_width64_forecast_sensitive_v31.yaml'
                cached=ROOT.parent/'e23_v31_forecast_sensitive/common_stage1/runs/e23-stage1.json'
            else:
                config=FORMAL/f'configs/e23_seed_{seed}.yaml'
                cached=FORMAL/f'seed_{seed}/E23-v31-B/stage1/runs/e23-stage1.json'
            assert cached.is_file()
        seed_cells[seed]=trial(f'search_seed{seed}',f'E23-v42-search-seed{seed}',seed,config,stage1=cached,
            batch=32 if high_selected else 128,epochs=200 if high_selected else 100)
    choices={metric:min(seed_cells,key=lambda seed:(selected(seed_cells[seed],metric)['best_val_'+metric],
        selected(seed_cells[seed],metric)['best_val_'+('mse' if metric=='mae' else 'mae')],plan['phase3']['seeds'].index(seed))) for metric in ['mae','mse']}
    write(ROOT/'seed-validation-choice.json',dict(selected_seeds=choices,selected_cells={k:str(seed_cells[s]) for k,s in choices.items()},
        validation_scores={str(s):{m:{k:selected(c,m)['best_val_'+k] for k in ['mse','mae']} for m in ['mae','mse']} for s,c in seed_cells.items()},seed_cells={str(s):str(c) for s,c in seed_cells.items()},
        seed_search_test_results_seen=False,all_six_validation_checkpoints_locked=True,T1_fixed_seed=502))
    for seed in plan['phase3']['seeds']:
        cell=seed_cells[seed]
        if not (cell/'completed-audit.json').exists():score(cell)
    verify()
    write(ROOT/'matrix-completed.json',dict(status='complete',selected_seeds=choices,selected_cells={k:str(seed_cells[s]) for k,s in choices.items()},
        geometry=geometry,increased_budget_selected=high_selected,new_T1_stage1_training=False,new_T1_stage2_training=True,source_unchanged=True,
        protected_files_unchanged=True,utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(read(ROOT/'matrix-completed.json')),flush=True)


if __name__=='__main__':main()
