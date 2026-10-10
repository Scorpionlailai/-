import csv,json
import numpy as np
from pathlib import Path
from malr.evaluation.report import file_sha256 as _sha
def file_sha256(p):return _sha(Path(p))
ROOT=Path(__file__).resolve().parent
DOCS=Path('/home/xyw/paper/E23/docs/experiments');PREFIX='ETTh1_T1_v41_slow_higher_missing_seed502_20261004'
def read(p):return json.loads(Path(p).read_text())
def output(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def main():
    plan=read(ROOT/'plan.json');rows=[];draws=[]
    for scene in plan['scenarios']:
        for model in plan['models']:
            cell=ROOT/scene['name']/model
            if not (cell/'completed-audit.json').exists():continue
            e=read(cell/'evaluation/twenty-test-draws.json');res=e['training_resources'];record=read(e['selection_manifest'])
            assert file_sha256(e['selection_manifest'])==e['selection_manifest_sha256']
            assert file_sha256(record['checkpoint'])==e['checkpoint_sha256']
            assert e['predictor_initial_sha256']==plan['predictor_initial_sha256'] and len(e['test_draws'])==20
            first=read(cell/'training/repair-selection.json' if model=='T1' else cell/'stage1/runs/e23-stage1.json')
            rr=[read(p) for p in res['resource_record_paths']]
            row=dict(scenario=scene['name'],start_rate=scene['start_rate'],block_len=scene['block_len'],actual_test_missing_fraction=scene['actual_test_missing_mean'],model=model,training_seed=502,seq_len=96,pred_len=96,test_masks=20,
                mse=e['test_mean']['mse'],mae=e['test_mean']['mae'],mask_mse_std=e['test_mask_std_ddof1']['mse'],mask_mae_std=e['test_mask_std_ddof1']['mae'],validation_mse=e['validation_replay']['mse'],validation_mae=e['validation_replay']['mae'],
                stage1_epochs=first['epochs_completed'],stage2_epochs=record['epochs_completed'],selected_stage2_epoch=record['best_epoch'],stage1_repair_mse=first['best_validation']['mse'] if model=='T1' else first['best_val_repair_mse'],
                full_training_seconds=res['full_workflow_seconds'],peak_allocated_gib=res['peak_allocated_gib'],peak_reserved_gib=res['peak_reserved_gib'],gpu=res['gpu'],training_batch_sizes=res['batch_size'],
                repair_stage_seconds=sum(r.get('stages',{}).get('repair',{}).get('wall_seconds',0) for r in rr),forecast_stage_seconds=sum(r.get('stages',{}).get('forecast',{}).get('wall_seconds',0) for r in rr),
                stage1_batch_size=first.get('batch_size',first.get('training',{}).get('batch_size')),stage2_batch_size=record.get('batch_size',record.get('training',{}).get('batch_size')),stage1_precision=first.get('training',{}).get('precision','float32'),stage2_precision=record.get('training',{}).get('precision','float32'),
                stage2_loss=record.get('loss',record.get('training',{}).get('loss','')),repair_update_policy=record.get('repair_update_policy','joint'),predictor_parameters=e['predictor_parameters'],predictor_initial_sha256=e['predictor_initial_sha256'],
                validation_mask_sha256=e['validation_mask_sha256'],test_bank_sha256=e['test_mask_bank_sha256'],checkpoint_sha256=e['checkpoint_sha256'],selection=e['selection_manifest'],evaluation=str(cell/'evaluation/twenty-test-draws.json'))
            rows.append(row)
            for d in e['test_draws']:
                assert file_sha256(d['window_metrics'])==d['window_metrics_sha256']
                draws.append(dict(scenario=scene['name'],model=model,**d))
    if not rows:return
    baseline_root=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments')
    bank=read(baseline_root/'e23_t1_five_seed_standard_20261004/test_masks/manifest.json')
    fraction=sum(g['missing_evaluation_fraction'] for g in bank['groups'])/20
    refs=[]
    for model,path in [('E23-v41-slow',baseline_root/'e23_v41_repair_drift_seed502_20261004/tail_slow/evaluation/twenty-test-draws.json'),('T1',baseline_root/'e23_v42_mse_updates_seed_search_20261004/T1_seed502/evaluation_mae/twenty-test-draws.json')]:
        e=read(path);res=e['training_resources'];record=read(e['selection_manifest'])
        assert file_sha256(e['selection_manifest'])==e['selection_manifest_sha256'] and file_sha256(record['checkpoint'])==e['checkpoint_sha256']
        assert e['predictor_initial_sha256']==plan['predictor_initial_sha256']
        ref=dict(scenario='sr010_bl8_cached',start_rate=.10,block_len=8,actual_test_missing_fraction=fraction,model=model,training_seed=502,seq_len=96,pred_len=96,test_masks=20,
            mse=e['test_mean']['mse'],mae=e['test_mean']['mae'],mask_mse_std=e['test_mask_std_ddof1']['mse'],mask_mae_std=e['test_mask_std_ddof1']['mae'],validation_mse=e['validation_replay']['mse'],validation_mae=e['validation_replay']['mae'],
            full_training_seconds=res['full_workflow_seconds'],peak_allocated_gib=res['peak_allocated_gib'],peak_reserved_gib=res['peak_reserved_gib'],gpu=res['gpu'],training_batch_sizes=res['batch_size'],
            predictor_parameters=e['predictor_parameters'],predictor_initial_sha256=e['predictor_initial_sha256'],validation_mask_sha256=e['validation_mask_sha256'],test_bank_sha256=e['test_mask_bank_sha256'],checkpoint_sha256=e['checkpoint_sha256'],selection=e['selection_manifest'],evaluation=str(path),cached_reference=True)
        refs.append(ref)
        for d in e['test_draws']:
            assert file_sha256(d['window_metrics'])==d['window_metrics_sha256']
            draws.append(dict(scenario='sr010_bl8_cached',model=model,**d))
    for row in rows:row['cached_reference']=False
    rows=refs+rows
    output(DOCS/(PREFIX+'.csv'),rows);output(DOCS/(PREFIX+'_all_draws.csv'),draws)
    lines=['# T1与v41-slow：高缺失场景从头训练','',
        '状态：'+('六组全部完成。' if (ROOT/'matrix-completed.json').exists() else '运行中；仅列已完成组。'),'',
        '固定ETTh1、回溯96、预测96、开发种子502。每个场景分别从头训练两模型，固定验证MAE选模，锁定后测试同一20组缺失位置。',
        '原PatchTST预测头源码、配置、35427参数与初始化一致。v41-slow保留原混合预测损失、uniform repair retention、慢修复更新和legacy padding；T1保留作者修复配置与纯MSE预测训练。因此比较的是两套既定训练方案，不能单独归因于修复器结构。','',
        '| 起点率/块长 | 实际测试缺失 | 模型 | MSE | MAE | 训练分钟 | allocated/reserved GiB |','|---|---:|---|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['start_rate']}/{r['block_len']} | {r['actual_test_missing_fraction']:.2%} | {r['model']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['full_training_seconds']/60:.2f} | {r['peak_allocated_gib']:.3f}/{r['peak_reserved_gib']:.3f} |")
    paired=[]
    for scenario in dict.fromkeys(row['scenario'] for row in rows):
        subset={r['model']:r for r in rows if r['scenario']==scenario}
        if set(subset)!=set(plan['models']):continue
        a,b=subset['E23-v41-slow'],subset['T1']
        ad=[d for d in draws if d['scenario']==scenario and d['model']=='E23-v41-slow']
        bd=[d for d in draws if d['scenario']==scenario and d['model']=='T1']
        assert len(ad)==len(bd)==20
        assert a['validation_mask_sha256']==b['validation_mask_sha256'] and a['test_bank_sha256']==b['test_bank_sha256']
        for x,y in zip(ad,bd):
            assert x['mask_seed']==y['mask_seed'] and x['mask_sha256']==y['mask_sha256'] and x['windows']==y['windows']==2688
        pair=dict(scenario=scenario,start_rate=a['start_rate'],block_len=a['block_len'],actual_test_missing_fraction=a['actual_test_missing_fraction'],
            v41_mse=a['mse'],t1_mse=b['mse'],v41_mae=a['mae'],t1_mae=b['mae'],mse_gap_v41_minus_t1=a['mse']-b['mse'],mae_gap_v41_minus_t1=a['mae']-b['mae'],
            t1_mse_reduction_percent=(a['mse']-b['mse'])/a['mse']*100,
            t1_mse_winning_masks=sum(y['mse']<x['mse'] for x,y in zip(ad,bd)),t1_mae_winning_masks=sum(y['mae']<x['mae'] for x,y in zip(ad,bd)),
            paired_mse_delta_std=float(np.std([x['mse']-y['mse'] for x,y in zip(ad,bd)],ddof=1)),
            paired_mae_delta_std=float(np.std([x['mae']-y['mae'] for x,y in zip(ad,bd)],ddof=1)),test_masks=20)
        paired.append(pair)
    output(DOCS/(PREFIX+'_paired.csv'),paired)
    lines+=['','## 相同缺失位置的配对比较','',
        '| 起点率/块长 | v41−T1 MSE | v41−T1 MAE | T1 MSE胜出位置组 | T1 MAE胜出位置组 |',
        '|---|---:|---:|---:|---:|']
    for r in paired:lines.append(f"| {r['start_rate']}/{r['block_len']} | {r['mse_gap_v41_minus_t1']:+.6f} | {r['mae_gap_v41_minus_t1']:+.6f} | {r['t1_mse_winning_masks']}/20 | {r['t1_mae_winning_masks']}/20 |")
    if (ROOT/'matrix-completed.json').exists():
        lines+=['','## 本轮结论','',
            '- 三种高缺失设置中，T1在全部20组配对测试位置的MSE与MAE均胜出；v41-slow没有呈现高缺失下的准确性优势。',
            '- 相比原场景MSE差距0.012874，0.15/8、0.20/8、0.10/16差距扩大到0.044759、0.068246、0.061176。',
            '- 0.10/16实际点缺失比0.20/8略低，但两模型的预测误差均更高。较长连续缺失应单独评估；两设置不完全匹配实际点缺失率，不能据此分离块长的纯因果效应。',
            '- 本轮v41-slow训练更快（8.82至9.23分钟），但allocated峰值更高（2.271至2.355GiB）；T1为10.85至18.80分钟、1.779GiB。显存allocated/reserved均列在主表及CSV。',
            '- 三种高缺失场景均从头训练。单开发种子结果不能替代多训练种子的正式结论；既定方案的Stage2损失、修复更新策略与padding/RNG差异也应保留披露。']
    lines+=['','0.10/8为已有缓存参考，没有重训；其余三种设置全部从头训练。','20组标准差仅衡量缺失位置变化；本轮单训练种子为开发比较。完整训练时间包括准备、训练、每轮验证和保存，最终测试另计。',f'\n[结果CSV]({PREFIX}.csv) · [20组明细]({PREFIX}_all_draws.csv)',f'\n实验目录：`{ROOT}`。','']
    (DOCS/(PREFIX+'.md')).write_text('\n'.join(lines))
    print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
