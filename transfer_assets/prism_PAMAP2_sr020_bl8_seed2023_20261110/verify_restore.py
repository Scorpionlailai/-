"""Read-only restore/replay helper; original model/runtime files are unchanged."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch

BUNDLE=Path(__file__).resolve().parent
RUN=BUNDLE/'r3_final_runs_20261030'
SOURCE=RUN/'snapshot/E24/versions/r3_20261030'
EXPECTED={
 'outputs/evaluation/PAMAP2/sr020_bl8/seed_2023/H_GLOBAL/model.pt':'cf4ae288225ac7f4f3410b9676fbdef373ff7800c1c307a42ee5592b1831f159',
 'outputs/runs/PAMAP2/sr020_bl8/seed_2023/H_GLOBAL/main/best.pt':'159667fcd23372010421ee876596da8f5d76c105167afbd07f2962f846e368a6',
 'outputs/runs/PAMAP2/sr020_bl8/seed_2023/H_GLOBAL/calibration/best_mae.pt':'1a05192b33cba1fb9eb31d28b1904abdbb0373833627c555b0f58ceb0478debd'}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=Path('/home/xyw/paper'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--mask-seed',type=int,default=2026101401)
    p.add_argument('--report',type=Path)
    p.add_argument('--save-window',type=Path)
    a=p.parse_args()
    assert 2026101401<=a.mask_seed<=2026101420,'Only the original 20261014 baseline test bank is permitted'
    torch.set_num_threads(4)
    for rel,digest in EXPECTED.items():assert sha(RUN/rel)==digest,rel
    for name in ['release_lock.json','vendor_lock.json']:
        for rel,digest in json.loads((SOURCE/name).read_text())['files'].items():assert sha(SOURCE/rel)==digest,rel
    for rel,digest in json.loads((RUN/'execution_lock.json').read_text())['files'].items():assert sha(RUN/rel)==digest,rel
    deps=json.loads((BUNDLE/'DATA_DEPENDENCIES.json').read_text())
    for f in deps['files']:assert sha(a.data_root/f['relative_to_data_root'])==f['sha256'],f
    os.environ.update(A64_DATASET='PAMAP2',R2_DATA_ROOT=str(a.data_root.resolve()),R2_OUTPUT_ROOT=str(RUN/'outputs'))
    sys.path.insert(0,str(RUN))
    from adapter import install
    data_module=install()
    from train import load_export
    from reuse import load_model_checkpoint
    from variants import library_hash
    from support import digest
    data=data_module.load_data('PAMAP2','sr020_bl8',RUN/'outputs/assets')
    assert data.n_vars==40 and data.test_seeds==list(range(2026101401,2026101421))
    manifest=json.loads((RUN/'outputs/assets/PAMAP2/sr020_bl8/manifest.json').read_text())
    for rec in manifest['masks']:
        path=data.assets/rec['path'];assert sha(path)==rec['sha256']
        with np.load(path,allow_pickle=False) as z:assert np.array_equal(data.masks[rec['split'],rec['seed']],z['observed'])
    for split,expected in manifest['windows'].items():
        starts=data.starts[split]
        assert len(starts)==expected['count'] and hashlib.sha256(starts.tobytes()).hexdigest()==expected['sha256']
    device=torch.device(a.device)
    exported=load_export(RUN/'outputs/evaluation/PAMAP2/sr020_bl8/seed_2023/H_GLOBAL/model.pt',data,device).eval()
    restored,info=load_model_checkpoint(data,'sr020_bl8',2023,'H_GLOBAL',device)
    assert info['coarse']['epoch']==99 and info['epoch']==34
    assert library_hash(exported.frozen.repair)==library_hash(restored.frozen.repair)
    assert digest(exported)==digest(restored),'Export and selected-stage full states differ'
    before=digest(exported);library_before=library_hash(exported.frozen.repair)
    checks=[]
    with torch.inference_mode():
        for split in ['validation','test']:
            truth,seen,starts=next(data.batches(split,2,mask_seed=a.mask_seed if split=='test' else None,max_batches=1))
            truth=truth.to(device);seen=seen.to(device);safe=torch.where(seen,truth,0.)
            pred=exported(safe,seen,starts);other=restored(safe,seen,starts)
            assert torch.equal(pred,other),'Restored-stage and exported predictions differ'
            assert torch.isfinite(pred).all() and torch.equal(pred[seen],safe[seen])
            valid=data.valid_for(starts);assert np.all(~seen.cpu().numpy()|valid)
            checks.append({'split':split,'mask_seed':2021 if split=='validation' else a.mask_seed,
                           'starts':starts.tolist(),'prediction_sha256':hashlib.sha256(pred.cpu().contiguous().numpy().tobytes()).hexdigest(),
                           'export_vs_selected_stages_bitwise_equal':True,'visible_values_preserved':True})
            if split=='test' and a.save_window:
                a.save_window.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(a.save_window,truth=truth.cpu().numpy(),observed=seen.cpu().numpy(),native_valid=valid,
                                    prediction=pred.cpu().numpy(),starts=starts,mask_seed=np.asarray(a.mask_seed),columns=np.asarray(manifest['columns']))
    assert digest(exported)==before and library_hash(exported.frozen.repair)==library_before
    report={'status':'passed','scope':'restoration only; no training, selection, full-test rerun, checkpoint serialization or original-result writes',
            'target':{'dataset':'PAMAP2','scene':'sr020_bl8','seed':2023,'variant':'H_GLOBAL','paper_name':'PRISM'},
            'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__},
            'selected_main_epoch':99,'selected_calibration_epoch':34,'checkpoint_sha256':EXPECTED,'original_masks_verified':21,
            'windows':manifest['windows'],'retrieval_library_sha256':library_before,'model_state_sha256':before,
            'parameters_registered':sum(x.numel() for x in exported.parameters()),'state_and_retrieval_unchanged_after_inference':True,'checks':checks}
    if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
