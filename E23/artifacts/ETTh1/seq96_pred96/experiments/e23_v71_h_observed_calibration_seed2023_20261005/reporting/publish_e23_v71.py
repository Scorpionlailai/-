from pathlib import Path
import csv, hashlib, importlib.util, json
import numpy as np
import torch

DOCS=Path('/home/xyw/paper/E23/docs/experiments')
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v71_h_observed_calibration_seed2023_20261005')
PREFIX=DOCS/'ETTh1_E23_v71_H_observed_calibration_seed2023_20261005'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csvwrite(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    plan=read(ROOT/'plan.json');frozen=read(ROOT/'frozen-definition.json')
    assert sha(ROOT/'plan.json')==frozen['plan_sha256']
    for p,h in frozen['sources'].items():assert sha(Path(plan['work'])/p)==h,p
    for p,h in frozen['protected'].items():assert sha(p)==h,p
    for p,h in frozen['controller'].items():assert sha(ROOT/p)==h,p
    for p,h in read(ROOT/'reference-locks.json').items():assert sha(p)==h,p
    spec=importlib.util.spec_from_file_location('v71_controller',ROOT/'execute.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    refs=list(csv.DictReader((DOCS/'ETTh1_E23_v69_T1_other_missing_seed2023_20261005_all_scenarios.csv').open()))
    rows=[];draws=[];pairs=[];selections=[];candidates=[];freeze_counts={};verdicts=[]
    def add(scene, name, score, repair_mse, repair_params, cached):
        manifest=Path(score['selection_manifest']); record=read(manifest)
        assert sha(manifest)==score['selection_manifest_sha256']
        assert sha(record['checkpoint'])==record['checkpoint_sha256']==score['checkpoint_sha256']
        assert score['test_mask_bank_sha256']==scene['test_bank_sha256']
        assert score['validation_mask_sha256']==scene['validation_mask_sha256']
        actual={k:float(np.mean([d[k] for d in score['test_draws']])) for k in ('mse','mae')}
        std={k:float(np.std([d[k] for d in score['test_draws']],ddof=1)) for k in ('mse','mae')}
        assert len(score['test_draws'])==20
        for k in actual:
            assert abs(actual[k]-score['test_mean'][k])<1e-12
            assert abs(std[k]-score['test_mask_std_ddof1'][k])<1e-12
        cost=score['training_resources'];assert cost['resource_status']=='measured_all_stages'
        resources=[read(p) for p in cost['resource_record_paths']]
        stages=[n for r in resources for n in r['stages']]
        assert sorted(stages)==['forecast','repair'],stages
        times={n:sum(r['stages'].get(n,{}).get('wall_seconds',0) for r in resources) for n in ['repair','forecast']}
        preparation=sum(r['preparation_and_reporting_seconds'] for r in resources)
        assert abs(sum(times.values())-cost['training_stages_seconds'])<1e-6
        assert abs(sum(r['full_workflow_seconds'] for r in resources)-cost['full_workflow_seconds'])<1e-6
        assert abs(sum(times.values())+preparation-cost['full_workflow_seconds'])<1e-6
        uuids=[r['device'].get('uuid',r['device']['name']) for r in resources]
        val=record.get('best_validation') or dict(mse=record['best_val_mse'],mae=record['best_val_mae'])
        rows.append(dict(dataset='ETTh1',seq_len=96,pred_len=96,scenario=scene['name'],start_rate=scene['start_rate'],block_len=scene['block_len'],
            actual_test_missing_fraction=scene['actual_test_missing_fraction'],test_masks=20,model=name,
            training_seed=record['seed'],selection_metric='mae',mse=actual['mse'],mae=actual['mae'],
            mask_mse_std_ddof1=std['mse'],mask_mae_std_ddof1=std['mae'],validation_mse=val['mse'],validation_mae=val['mae'],
            full_training_seconds=cost['full_workflow_seconds'],full_training_minutes=cost['full_workflow_seconds']/60,
            repair_stage_seconds=times['repair'],forecast_stage_seconds=times['forecast'],
            preparation_and_reporting_seconds=preparation,
            peak_allocated_gib=cost['peak_allocated_gib'],peak_reserved_gib=cost['peak_reserved_gib'],training_gpu=cost['gpu'],
            training_device_uuids=';'.join(uuids),training_physical_device_consistent=len(set(uuids))==1,
            stage1_batch_size=16 if name=='T1' else 128,stage2_batch_size=128,
            stage1_precision='bfloat16' if name=='T1' else 'float32',stage2_precision='float32',
            stage2_repair_update_policy='joint_author_policy' if name=='T1' else 'frozen_v1',
            stage2_forecast_loss='mse' if name=='T1' else 'mse_plus_0p2_mae',
            historical_calibration='ridge_offset_loo_v1' if name=='E23-v71' else 'none',
            historical_offset_prior=4.0 if name=='E23-v71' else 0.,
            historical_offset_bound_scales=2.0 if name=='E23-v71' else 0.,
            historical_confidence=('observed_leave_one_out' if name=='E23-v71' else 'legacy_visible_error' if name=='E23-v69' else 'not_applicable'),
            stage1_validation_repair_mse=repair_mse,repair_parameters=repair_params,predictor_parameters=35427,
            selected_stage2_epoch=record['best_epoch'],predictor_initial_sha256=record['predictor_initial_sha256'],
            checkpoint_sha256=record['checkpoint_sha256'],test_bank_sha256=scene['test_bank_sha256'],
            validation_mask_sha256=scene['validation_mask_sha256'],selection=str(manifest),
            cached_reference=cached,new_training_in_this_trial=not cached))
        for d in score['test_draws']:draws.append(dict(scenario=scene['name'],model=name,training_seed=record['seed'],**d))
    for scene in plan['scenarios']:
        if not (ROOT/scene['name']/'completed-audit.json').exists():continue
        cell=ROOT/scene['name']/'E23-v71-mae';s1_path=cell/'stage1/runs/e23-stage1.json';s1=read(s1_path)
        a=torch.load(s1['checkpoint'],map_location='cpu',weights_only=False)['model_state']
        assert s1['epochs_completed']==100 and s1['optimizer_updates']==6600 and s1['seed']==2023
        for metric in ['mae','mse']:
            c=ROOT/scene['name']/('E23-v71-'+metric);m=c/'stage2/runs/e23-stage2.json';rec=read(m)
            assert controller.stage2_lock(rec,plan,scene,metric)==read(c/'validation-lock-before-tests.json')
            assert rec['stage1_checkpoint_sha256']==s1['checkpoint_sha256']
            assert rec['memory_sha256']==s1['memory_sha256'] and rec['key_sha256']==s1['key_sha256']
            b=torch.load(rec['checkpoint'],map_location='cpu',weights_only=False)['model_state']
            assert all(torch.equal(v,b['repair.'+k]) for k,v in a.items())
            freeze_counts[scene['name']+'/'+metric]=len(a)
            score=read(c/'evaluation/twenty-test-draws.json')
            selections.append(dict(scenario=scene['name'],selection_metric=metric,training_seed=2023,selected_epoch=rec['best_epoch'],
                validation_mse=rec['best_val_mse'],validation_mae=rec['best_val_mae'],test_mse=score['test_mean']['mse'],test_mae=score['test_mean']['mae'],
                stage1_epochs_completed=100,stage1_optimizer_updates=6600,stage2_epochs_completed=100,stage2_trainable_repair_parameters=0,
                selection=str(m),checkpoint_sha256=rec['checkpoint_sha256'],role='primary' if metric=='mae' else 'secondary, same training trajectory'))
        e=read(cell/'evaluation/twenty-test-draws.json')
        add(scene,'E23-v71',e,s1['best_val_repair_mse'],420973,False)
        match=[r for r in refs if r['scenario']==scene['name']]
        for name,field in [('E23-v69','v69_evaluation'),('T1','t1_evaluation')]:
            old=next(r for r in match if r['model']==name)
            repair_mse=float(old['stage1_validation_repair_mse']) if old['stage1_validation_repair_mse'].strip() else None
            reference=read(scene[field]);add(scene,name,reference,repair_mse,int(old['repair_parameters']),True)
            for ed,rd,delta in zip(e['test_draws'],reference['test_draws'],controller.paired_draws(e,reference)):
                with np.load(ed['window_metrics']) as x,np.load(rd['window_metrics']) as y:
                    assert np.array_equal(x['target_start'],y['target_start'])
                pairs.append(dict(scenario=scene['name'],reference=name,**delta))
        own=next(r for r in rows if r['scenario']==scene['name'] and r['model']=='E23-v71')
        parent=next(r for r in rows if r['scenario']==scene['name'] and r['model']=='E23-v69')
        accepted=own['validation_mse']<parent['validation_mse'] and own['validation_mae']<=parent['validation_mae']
        verdicts.append(dict(scene=scene['name'],validation_upgrade_vs_v69=accepted,
            validation_delta={k:own['validation_'+k]-parent['validation_'+k] for k in ['mse','mae']},
            test_used_for_promotion=False,retained_candidate='E23-v71' if accepted else 'E23-v69'))
        chosen=read(cell/'stage2/runs/e23-stage2.json')
        candidates.append(dict(scenario=scene['name'],selection=own['selection'],checkpoint=chosen['checkpoint'],
            checkpoint_sha256=chosen['checkpoint_sha256'],config=plan['config'],training_seed=2023,
            stage1_selection=str(s1_path),scenario_environment=scene['scenario_manifest'],test_mask_bank=scene['test_bank'],
            scores=e['test_mean'],mask_std=e['test_mask_std_ddof1'],validation_upgrade_vs_v69=accepted,
            main_source_integrated=False))
    if not rows:
        print(json.dumps({'status':'training','completed':0}));return
    csvwrite(PREFIX.with_suffix('.csv'),rows);csvwrite(str(PREFIX)+'_all_draws.csv',draws)
    csvwrite(str(PREFIX)+'_paired.csv',pairs);csvwrite(str(PREFIX)+'_selections.csv',selections)
    lines=['# E23-v71：真实可见点校准H候选与留一置信度实验','',
        f'完成{len(candidates)}/1场景，ETTh1 96→96；主表按固定验证MAE选模，20组共享测试位置。','',
        '| 起点率/块长 | 实际缺失 | 模型 | MSE ± 位置std | MAE ± 位置std | 完整训练分钟 | allocated / reserved GiB |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['start_rate']}/{r['block_len']} | {r['actual_test_missing_fraction']*100:.2f}% | {r['model']} | {r['mse']:.6f} ± {r['mask_mse_std_ddof1']:.6f} | {r['mae']:.6f} ± {r['mask_mae_std_ddof1']:.6f} | {r['full_training_minutes']:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    lines += ['','## 固定验证裁定','']
    for v in verdicts:
        lines.append(f"- {v['scene']}：验证相对v69的MSE变化{v['validation_delta']['mse']:+.6f}，MAE变化{v['validation_delta']['mae']:+.6f}；{'通过升级条件' if v['validation_upgrade_vs_v69'] else '未通过，保留v69'}。测试不决定晋级。")
    lines += ['','## 配对测试','', '| 场景 | 对照 | v71−对照MSE | v71−对照MAE | MSE胜出 | MAE胜出 |','|---|---|---:|---:|---:|---:|']
    for scene in candidates:
        for name in ['E23-v69','T1']:
            p=[r for r in pairs if r['scenario']==scene['scenario'] and r['reference']==name]
            lines.append(f"| {scene['scenario']} | {name} | {np.mean([r['mse_delta'] for r in p]):+.6f} | {np.mean([r['mae_delta'] for r in p]):+.6f} | {sum(r['mse_delta']<0 for r in p)}/20 | {sum(r['mae_delta']<0 for r in p)}/20 |")
    lines += ['','## 改动和范围','',
        '- 仅校准原H可用候选：用真实可见位置的残差和除以(n+4)估计局部水平偏移，限制在±2倍稳健局部尺度；四个虚拟观测将稀疏拟合收缩到零。top4、0.7阈值、原bank成员、mask、完全缺失patch回退保持。没有新增训练参数，修复器420973参数。',
        '- 候选数值在H消息投影前对齐；置信度逐个留出真实可见坐标，以(n+3)重新估计带相同约束的偏移，再评分被留出的值。未使用隐藏真值或未来标签来计算偏移/权重；历史原始候选和对齐后的候选分别留存。',
        '- 留一检验针对偏移拟合；检索候选及原窗口可见统计保持固定，不宣称整个检索流程的独立交叉验证。',
        '- Stage1完全保留原global MSE+0.1检索项+recent_mean_v2监督，没有v70查询或附加对齐损失。',
        '- 每场景seed2023，100轮Stage1+100轮Stage2。完整修复器冻结，原PatchTST35427参数及初始化相同；Stage2原MSE+0.2MAE/uniform项和lr1e-4保持。MAE主选择及MSE次选择在测试前锁定，次选择没有额外训练。',
        '- 同seed同配方v69是直接模块对照；T1作者修复配置seed502、Stage1 batch16/BF16、Stage2纯MSE联合更新保持。基线准确性和资源记录复用，没有重训。',
        '- 20组标准差仅反映缺失位置变化；单训练seed、相同ETTh1未来测试时间段。相同GPU型号/UUID列出，但T1与E23的batch/精度/配方不同。',
        '- 完整训练时间包含两阶段准备、全部训练/验证和保存；额外测试/诊断不计入此训练时间，见各evaluation/diagnostics原始记录。显存是PyTorch allocated/reserved峰值。',
        '- 完整回归1139通过、21项与基线相同的既有失败、1项跳过；具体名称见test-audit.json。没有把旧全套称为全绿。',
        '- 本轮v71相对固定v69候选进行验证裁定，不重写原73%实验中v63验证保留父版本的记录。源码与权重保存于独立目录，未合入主E23。','',
        f'[完整CSV]({PREFIX.name}.csv) · [逐组]({PREFIX.name}_all_draws.csv) · [配对]({PREFIX.name}_paired.csv) · [主/次选模]({PREFIX.name}_selections.csv)','']
    PREFIX.with_suffix('.md').write_text('\n'.join(lines))
    index=dict(work=plan['work'],config=plan['config'],source_archive=str(ROOT/'frozen-source.tar.gz'),
        architecture='v69 plus observed-only H level alignment and LOO reliability',training_seed=2023,repair_parameters=420973,predictor_parameters=35427,
        candidates=candidates,main_source_integrated=False,new_t1_training_runs=0,
        measured_new_training_seconds=sum(r['full_training_seconds'] for r in rows if r['model']=='E23-v71'))
    (DOCS/'E23_v71_h_observed_calibration_candidates_20261005.json').write_text(json.dumps(index,indent=2)+'\n')
    audit=dict(completed_scenarios=[c['scenario'] for c in candidates],main_rows=len(rows),draw_rows=len(draws),
        paired_rows=len(pairs),selection_rows=len(selections),repair_tensors_bitwise_frozen=freeze_counts,
        source_files_verified=len(frozen['sources']),protected_files_verified=len(frozen['protected']),
        verdicts=verdicts,csv_sha256=sha(PREFIX.with_suffix('.csv')),all_scenes_complete=len(candidates)==len(plan['scenarios']),
        reference_locks_verified=len(read(ROOT/'reference-locks.json')))
    (ROOT/'report-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))

if __name__=='__main__':main()
