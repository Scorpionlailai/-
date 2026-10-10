"""Reuse exact author T1 Stage1 and train one downstream trajectory with two selectors."""
from time import perf_counter
_PROCESS_STARTED=perf_counter()
import argparse
import json
from pathlib import Path
import torch
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.author_profiles import build_author_model,load_author_profiles
from integrations.forecast_compare.author_provenance import author_source_identity,write_new_record
from integrations.forecast_compare.protocol import load_p4_protocol,with_training_seed
from integrations.forecast_compare.resources import TrainingResources,selected_training_resources
from integrations.forecast_compare.t1_p4 import T1ForecastModel
from malr.evaluation.report import file_sha256
from scripts.run_author_aligned_p4 import _forecast_head,_seed,_train_stage,ROOT,DEFAULT_MASKS


def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',type=Path,required=True)
    p.add_argument('--stage1',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();device=torch.device('cuda:1');torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError(args.output)
    profiles=load_author_profiles(args.profiles);cfg=profiles['models']['T1']
    first=json.loads(args.stage1.read_text());seed=first['seed']
    assert seed==502 and first['stage']=='repair' and first.get('stage1_aux_forecast','none')=='none'
    assert file_sha256(Path(first['checkpoint']))==first['checkpoint_sha256']
    with TrainingResources(args.output,device=device,batch_size=128,model='T1',started_at=_PROCESS_STARTED) as resources:
        protocol=with_training_seed(load_p4_protocol(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml',DEFAULT_MASKS),seed)
        identity=author_source_identity(args.profiles,cfg,protocol)
        assert first['upstream_source_sha256']==identity['upstream_source_sha256'] and first['author_profiles_sha256']==identity['author_profiles_sha256']
        _seed(seed);imputer=build_author_model('T1',cfg,ROOT/cfg['root'],device)
        imputer.load_state_dict(torch.load(first['checkpoint'],map_location=device,weights_only=False)['model_state'],strict=True)
        _seed(seed);model=T1ForecastModel(imputer,_forecast_head(protocol)).to(device)
        identity.update(repair_selection_manifest=str(args.stage1.resolve()),repair_selection_sha256=file_sha256(args.stage1),
            predictor_initial_sha256=parameter_hash(model.predictor),stage1_reused=True,dual_checkpoint_selection=True)
        old_resources=selected_training_resources([args.stage1.parent/'forecast-selection.json'])
        original=json.loads(Path(old_resources['resource_record_paths'][0]).read_text())
        repair=original['stages']['repair']
        cached={**original,'model':'T1','selection_manifest':str(args.stage1.resolve()),
            'selection_manifest_sha256':file_sha256(args.stage1),'checkpoint_sha256':first['checkpoint_sha256'],
            'epochs_completed':first['epochs_completed'],'stages':{'repair':repair},
            'full_workflow_seconds':repair['wall_seconds']+original['preparation_and_reporting_seconds'],
            'training_stages_seconds':repair['wall_seconds'],'peak_allocated_bytes':repair['peak_allocated_bytes'],
            'peak_reserved_bytes':repair['peak_reserved_bytes'],'cached_stage1':True,'legacy_training_seconds':first['training_seconds'],
            'source_resource_record':old_resources['resource_record_paths'][0],
            'source_resource_record_sha256':file_sha256(Path(old_resources['resource_record_paths'][0])),
            'timing_scope':'cached original repair stage + original entrypoint non-stage overhead, all assigned to cached Stage1; excludes old forecast stage'}
        cached_path=args.output/'cached-stage1-resources.json';write_new_record(cached_path,cached)
        identity.update(cached_stage1_resource_record=str(cached_path.resolve()),cached_stage1_resource_sha256=file_sha256(cached_path))
        assert identity['predictor_initial_sha256']=='26cee63e94dc41607f346f09a29e01cf7d02e75e513b1d9b631ce270ec3e0b7c'
        training={**profiles['shared_forecast_stage'],'lradj':'constant'}
        with resources.stage('forecast'):
            chosen=_train_stage('T1',model,protocol,None,cfg,training,'forecast',args.output,device,identity,dual_selection=True)
        resources.selection_manifest=chosen


if __name__=='__main__':main()
