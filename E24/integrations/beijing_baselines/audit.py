"""Verify saved files and metric support without calling any repair model."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .data import SharedData, SCENES, file_sha256, write_json
from .report import MODELS, DATASET, read


def verify(root):
    root=Path(root)
    cases=[];head_shas=set();files_checked=0
    for scene in SCENES:
        data=SharedData(DATASET,scene,root/'shared_assets')
        starts=data.starts['test'];support=data.bounds.test.start-96
        native=data.real_observed[support:data.bounds.test.end]
        future_prefix=np.r_[0,data.real_observed.sum(1).cumsum()]
        forecast_count=int((future_prefix[starts+192]-future_prefix[starts+96]).sum())
        for model in MODELS:
            out=root/DATASET/scene/model/'seed_2023'
            assert (read(out/'status.json') or {}).get('phase')=='stage2_complete',f'incomplete run: {out}'
            selection=read(out/'stage2/selection.json')
            assert selection['parent_stage1_sha256']==file_sha256(out/'stage1/selected.pt')
            head_shas.add(selection['head_initial_sha256'])
            actual=hashlib.sha256()
            state=torch.load(out/'stage1/selected.pt',map_location='cpu',weights_only=True)
            for key,value in sorted(state.items()):
                actual.update(key.encode());actual.update(value.contiguous().numpy().tobytes())
            repair_sha=actual.hexdigest()
            assert repair_sha==selection['repair_state_sha256']
            for split,seed in [('train',2021),('validation',2021)]+[('test',s) for s in data.test_seeds]:
                folder=out/'repair_cache'/f'{split}_{seed}'
                cached=read(folder/'manifest.json');assert cached is not None
                binding=cached['binding']
                assert binding['state_sha256']==repair_sha
                assert binding['checkpoint_sha256']==selection['parent_stage1_sha256']
                assert binding['data_manifest_sha256']==data.manifest_sha
                for name,sha in binding['implementation_sha256'].items():
                    assert file_sha256(Path(__file__).parent/name)==sha
                for name,sha in cached['files'].items():
                    assert file_sha256(folder/name)==sha;files_checked+=1
            repair=read(out/'evaluation/repair_summary.json')
            main=read(out/'evaluation/forecast_mae_summary.json')
            second=read(out/'evaluation/forecast_mse_summary.json')
            assert [g['mask_seed'] for g in repair['groups']]==data.test_seeds
            for r,a,b in zip(repair['groups'],main['groups'],second['groups']):
                seed=r['mask_seed']
                assert a['mask_seed']==b['mask_seed']==seed
                expected_sha=next(x['sha256'] for x in data.manifest['masks'] if x['split']=='test' and x['seed']==seed)
                assert r['mask_sha256']==a['mask_sha256']==b['mask_sha256']==expected_sha
                timeline=(~data.masks['test',seed] & native).sum(1)
                prefix=np.r_[0,timeline.cumsum()]
                expected={'full96':int((prefix[starts-support+96]-prefix[starts-support]).sum()),
                    'recent32':int((prefix[starts-support+96]-prefix[starts-support+64]).sum()),'forecast':forecast_count}
                for section in ('full96','recent32'):
                    assert r[section]==a[section]==b[section]
                for group in (r,a,b):
                    for section in ('full96','recent32','forecast'):
                        if section not in group:continue
                        score=group[section];assert score['coordinates']==expected[section]
                        assert abs(score['mse']-score['squared_error_sum']/score['coordinates'])<1e-12
                        assert abs(score['mae']-score['absolute_error_sum']/score['coordinates'])<1e-12
            for record in (repair,main,second):
                for section,scores in record['summary'].items():
                    for name in ('mse','mae'):
                        values=np.array([g[section][name] for g in record['groups']])
                        assert abs(values.mean()-scores[name+'_mean'])<1e-12
                        assert abs(values.std(ddof=1)-scores[name+'_mask_std_ddof1'])<1e-12
            cases.append(dict(model=model,scene=scene,checkpoint_and_state_verified=True,
                repair_cache_entries=22,test_groups=20,native_valid_scoring_verified=True))
    assert len(cases)==24 and len(head_shas)==1
    result=dict(passed=True,cases=cases,total_test_groups=480,cache_files_sha_verified=files_checked,
        head_initial_sha256=list(head_shas),repair_model_calls=0)
    write_json(root/'final_saved_output_audit.json',result)
    return result
