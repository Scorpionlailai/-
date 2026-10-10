"""Publish only completed, locked evaluations; retain both checkpoint selectors."""
import csv
import hashlib
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
PREFIX='ETTh1_P4_E23_v43_nonoverlap_seed502_20261004'

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    completion=read(ROOT/'matrix-completed.json');assert completion['status']=='complete'
    plan=read(ROOT/'plan.json');definition=read(ROOT/'matrix-definition.json')
    assert digest(ROOT/'plan.json')==definition['plan_sha256']
    for p,h in definition['protected'].items():assert digest(p)==h,p
    for rel,h in definition['sources'].items():assert digest(Path('/home/xyw/paper/.e23-v43-nonoverlap/E23')/rel)==h,rel
    pairs=[]
    for spec in plan['cells']:
        cell=ROOT/spec['name'];diag={r['stage']:r for r in read(cell/'recovery-diagnostics.json')}
        for metric in ['mae','mse']:
            pairs.append((spec['label'],cell/f'evaluation_{metric}/twenty-test-draws.json',spec,diag,metric))
    refs=[('E23-v41-tail32-slow',ROOT.parent/'e23_v41_repair_drift_seed502_20261004/tail_slow/evaluation/twenty-test-draws.json'),
          ('E23-v41-tail32-frozen',ROOT.parent/'e23_v41_repair_drift_seed502_20261004/tail_frozen/evaluation/twenty-test-draws.json'),
          ('T1',ROOT.parent/'e23_v42_mse_updates_seed_search_20261004/T1_seed502/evaluation_mae/twenty-test-draws.json')]
    previous={r['model']:r for r in csv.DictReader((DOCS/'ETTh1_P4_E23_v41_repair_drift_seed502_20261004.csv').open())}
    for label,path in refs:pairs.append((label,path,None,None,'mae'))
    rows=[];draws=[]
    history={}
    for label,path,spec,diag,metric in pairs:
        e=read(path);r=read(e['selection_manifest']);res=e['training_resources']
        assert e['predictor_initial_sha256']==plan['predictor_initial_sha256'] and e['predictor_parameters']==35427
        assert e['test_mask_bank_sha256']==definition['test_bank_sha256']
        assert digest(e['selection_manifest'])==e['selection_manifest_sha256']
        assert digest(r['checkpoint'])==e['checkpoint_sha256']
        resource_rows=[read(p) for p in res['resource_record_paths']]
        new_resources=[x for p,x in zip(res['resource_record_paths'],resource_rows) if Path(p).is_relative_to(ROOT)]
        old=previous.get(label,{})
        bank={}
        if spec:
            first_path=Path(spec['cached_stage1']) if spec.get('cached_stage1') else ROOT/spec['name']/'stage1/runs/e23-stage1.json'
            first_record=read(first_path)
            snapshot=torch.load(first_record['checkpoint'],map_location='cpu',weights_only=False)['memory_state']['snapshot']
            groups=[g for channel in snapshot['groups'] for g in channel]
            bank=dict(history_bank_groups=len(groups),history_bank_members=sum(len(g['members']) for g in groups),
                      history_observed_identities=len(snapshot['seen']),history_sampling_stride=8)
            history[spec['name']]=snapshot['seen']
        first=diag['stage1'] if diag else None
        final=diag['stage2_'+metric] if diag else None
        row=dict(model=label,dataset='ETTh1',seq_len=96,pred_len=96,missing_block_start_rate=.1,missing_block_length=8,
            checkpoint_selection=metric,training_seed=502,test_masks=20,
            mse=e['test_mean']['mse'],mae=e['test_mean']['mae'],mask_mse_std_ddof1=e['test_mask_std_ddof1']['mse'],mask_mae_std_ddof1=e['test_mask_std_ddof1']['mae'],
            validation_mse=e['validation_replay']['mse'],validation_mae=e['validation_replay']['mae'],selected_stage2_epoch=r['best_epoch'],
            repair_stride=spec['repair_stride'] if spec else (8 if label.startswith('E23') else ''),
            repair_patch_count=(6 if spec['repair_stride']==16 else 12) if spec else '',
            point_feature_width=spec['point_feature_width'] if spec else '',predictor_patch_length=16,predictor_stride=8,predictor_parameters=35427,
            repair_parameters=final['repair_parameters'] if final else '',**bank,
            stage1_repair_mse=first['metrics']['all96']['mse'] if first else old.get('stage1_repair_mse',''),
            stage1_last32_repair_mse=first['metrics']['last32']['mse'] if first else old.get('stage1_last32_repair_mse',''),
            stage1_boundary_repair_mse=first['metrics']['block_boundaries']['mse'] if first else '',
            stage2_repair_mse=final['metrics']['all96']['mse'] if final else old.get('stage2_repair_mse',''),
            stage2_last32_repair_mse=final['metrics']['last32']['mse'] if final else old.get('stage2_last32_repair_mse',''),
            stage2_before64_repair_mse=final['metrics']['before64']['mse'] if final else old.get('stage2_before64_repair_mse',''),
            stage2_boundary_repair_mse=final['metrics']['block_boundaries']['mse'] if final else '',
            full_training_seconds=res['full_workflow_seconds'],peak_allocated_gib=res['peak_allocated_gib'],peak_reserved_gib=res['peak_reserved_gib'],
            gpu=res['gpu'],gpu_uuids=';'.join(sorted({str(x.get('device',{}).get('uuid','unknown')) for x in resource_rows})),
            training_batch_sizes=res['batch_size'],hardware_consistent=res['hardware_consistent'],
            actual_new_training_seconds=sum(x['full_workflow_seconds'] for x in new_resources),
            repair_stage_seconds=sum(x.get('stages',{}).get('repair',{}).get('wall_seconds',0) for x in resource_rows),
            forecast_stage_seconds=sum(x.get('stages',{}).get('forecast',{}).get('wall_seconds',0) for x in resource_rows),
            repair_update_policy=r.get('repair_update_policy','joint'),predictor_learning_rate=r.get('predictor_learning_rate',r.get('learning_rate',.0001)),
            repair_learning_rate=r.get('repair_learning_rate',.0001),stage2_loss=r.get('loss','forecast_mse_only' if label=='T1' else ''),downstream_protocol=r.get('downstream_protocol','legacy'),
            cached_reference=spec is None,stage1_reused=bool(spec and spec.get('cached_stage1')),predictor_initial_sha256=e['predictor_initial_sha256'],
            checkpoint_sha256=e['checkpoint_sha256'],evaluation=str(path),
            matched_validation_gate=completion['validation_gates'].get(spec['name'],{}).get('matched_gate','') if spec else '',
            v41_reference_validation_gate=completion['validation_gates'].get(spec['name'],{}).get('v41_reference_gate','') if spec else '',
            scope='single development seed; mask variation std does not measure training-seed variation')
        rows.append(row)
        assert len(e['test_draws'])==20
        for d in e['test_draws']:
            assert digest(d['window_metrics'])==d['window_metrics_sha256']
            draws.append(dict(model=label,checkpoint_selection=metric,training_seed=502,**d))
    csvwrite(DOCS/(PREFIX+'.csv'),rows)
    primary=[r for r in rows if r['checkpoint_selection']=='mae']
    csvwrite(DOCS/(PREFIX+'_primary.csv'),primary)
    csvwrite(DOCS/(PREFIX+'_all_draws.csv'),draws)
    recovery=[dict(model=spec['label'],stage=d['stage'],region=region,**m) for spec in plan['cells']
              for d in read(ROOT/spec['name']/'recovery-diagnostics.json') for region,m in d['metrics'].items()]
    csvwrite(DOCS/(PREFIX+'_recovery.csv'),recovery)
    identical_history=all(x==history['overlap_slow'] for x in history.values())
    lines=['# E23 v43：非重叠修复分块与逐点特征增强','',
        '2026-10-04 完成。以下为开发 seed502 与同一20组测试缺失位置的比较。预测头保持原 PatchTST D16、35427参数；T1未重新训练。','',
        '## 固定方案','',
        '- O：修复16/8，原tail32点级精修；复用原Stage1，只新训练Stage2。',
        '- N：修复16/16，6个有效块，原tail32点级精修。',
        '- P：N基础上，将每块64维状态展开为16个32维点特征，再进行64维时间精修。',
        '- N/P生成唯一96点序列，再按预测器原16/8重新分块。修复和预测的几何已解耦；有效重叠副本来自同一序列。',
        '- 三组均为Stage1全局缺失MSE+0.1检索损失、Stage2纯预测MSE，Adam，batch128，FP32，100+100轮；预测器LR1e-4、修复器LR1e-5。Stage2采用与T1一致的末值补齐和训练前RNG重置。',
        '- H库保留原16/8观测采样；N/P的T半径与分带按实际时间跨度调整。V相邻块偏移由8变16点，因此几何比较不能仅归因于取消平均。','',
        '## 按验证MAE选模的主结果','',
        '| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | 峰值 allocated / reserved GiB |',
        '|---|---:|---:|---|---:|---|']
    for r in primary:
        lines.append(f"| {r['model']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['validation_mse']:.6f} / {r['validation_mae']:.6f} | {r['full_training_seconds']/60:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    lines+=['','v41两行是已保存的旧训练参考，Stage2含MSE+0.2MAE及修复保留项，padding/RNG规则也不同；不作为本轮结构的匹配对照。O/N/P才是同训练口径的比较。T1保留作者Stage1配置，其batch16/BF16与E23不同。','',
        '**本轮裁定：保留v41-slow开发参考。** N通过相对O的固定验证门槛，但N/P都未超过v41的固定验证指标。测试上N的MSE较O变差；P较N有所改善，仍未超过O的MSE或v41的MSE/MAE。非重叠方案本次减少了时间与显存占用，尚未带来目标预测MSE的改善。原正式五训练种子表保持不变。','',
        '## 验证裁定与修复诊断','',
        '运行前固定：主checkpoint按验证MAE，另存验证MSE最低版本。结构通过匹配对照的条件为验证MSE严格下降、MAE不退步；再单列相对旧v41开发参考的相同门槛。测试不参与裁定。','',
        '```json',json.dumps(completion['validation_gates'],ensure_ascii=False,indent=2),'```','',
        '| 模型 | Stage1修复MSE | Stage1最近32点MSE | Stage2修复MSE | Stage2最近32点MSE |',
        '|---|---:|---:|---:|---:|']
    for r in primary[:3]:
        lines.append(f"| {r['model']} | {r['stage1_repair_mse']:.6f} | {r['stage1_last32_repair_mse']:.6f} | {r['stage2_repair_mse']:.6f} | {r['stage2_last32_repair_mse']:.6f} |")
    newtime=sum(r['actual_new_training_seconds'] for r in primary)
    lines+=['','恢复指标在全局训练标准化尺度，按唯一原缺失坐标计数；边界附近指标与前64点指标见CSV。20个mask标准差描述缺失位置变化，不能代替多训练种子稳定性。seed502属于开发选择。','',
        '## 时间、资源与核对','',
        f'本轮实际新增训练总耗时 {newtime:.3f} 秒。O的完整耗时包含已保存Stage1与新Stage2；N/P均实测两个新阶段。验证重放、诊断与最终测试另列，不计入完整训练。显存为PyTorch allocator峰值。',
        'T1完整时间包含其已保存Stage1资源与v42的Stage2资源；该Stage1拆分口径继承v42记录，未将旧Stage2重复计入。',
        f'三组历史观测坐标身份是否一致：{identical_history}。采样步长保持8；训练后的检索key与聚类成员数可以不同，CSV单列组数和成员数，不宣称训练后的历史库权重相同。',
        f"82项相关测试通过；CUDA检查通过观测/mask保持、隐藏输入不变、6块唯一覆盖、原预测头初始化和源码一致、增强版初始函数一致。最终核对 {len(definition['sources'])} 个冻结源码文件和 {len(definition['protected'])} 个保护文件的SHA一致。",'',
        f'- [全部选模结果CSV]({PREFIX}.csv)',f'- [MAE选模主表]({PREFIX}_primary.csv)',
        f'- [每组测试位置明细]({PREFIX}_all_draws.csv)',f'- [修复诊断CSV]({PREFIX}_recovery.csv)',
        f'- 完整实验：`{ROOT}`；冻结源码、实现patch、训练命令、资源JSON与全部checkpoint均已保存。','']
    (DOCS/(PREFIX+'.md')).write_text('\n'.join(lines))
    audit=dict(status='passed',sources=len(definition['sources']),protected_files=len(definition['protected']),
        evaluated_checkpoint_rows=len(rows),new_training_trajectories=3,new_stage1_runs=2,new_stage2_runs=3,
        test_draw_rows=len(draws),new_test_draw_rows=120,recovery_rows=len(recovery),actual_new_training_seconds=newtime,
        historical_observed_identities_identical=identical_history,
        outputs={str(p):digest(p) for p in DOCS.glob(PREFIX+'*')})
    (ROOT/'final-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(primary=[{k:r[k] for k in ['model','mse','mae','validation_mse','validation_mae','full_training_seconds','peak_allocated_gib','peak_reserved_gib']} for r in primary],audit=audit)))

if __name__=='__main__':main()
