"""Publish one locked repair trial, preserving validation-based parent selection."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
PREFIX = DOCS/'ETTh1_E23_v76_stage1_task_supervision_seed2023_20261006'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = read(ROOT/'plan.json')
    frozen = read(ROOT/'frozen-definition.json')
    assert sha(ROOT/'plan.json') == frozen['plan_sha256']
    for rel, digest in frozen['sources'].items():
        assert sha(Path(plan['work'])/rel) == digest, rel
    for p, digest in {**frozen['protected'], **read(ROOT/'reference-locks.json')}.items():
        assert sha(p) == digest, p
    for p, digest in frozen['controller'].items():
        assert sha(ROOT/p) == digest, p
    spec = importlib.util.spec_from_file_location('locked_controller', ROOT/'execute.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    scene = plan['scenarios'][0]
    name = 'E23-v76'
    primary = ROOT/scene['name']/(name+'-mae')
    assert read(ROOT/'completed-audit.json')['status'] == 'complete'
    s1_path = primary/'stage1/runs/e23-stage1.json'
    s1 = read(s1_path)
    assert s1['epochs_completed'] == 100 and s1['optimizer_updates'] == 6600
    payload1=torch.load(s1['checkpoint'],map_location='cpu',weights_only=False)
    from_expected=dict(stage1_aux_forecast='forecast_mse_v1',stage1_aux_forecast_weight=.05,stage1_aux_forecast_warmup=10,stage1_aux_predictor_seed=2022,stage1_aux_predictor_discarded=True,stage1_aux_predictor_parameters=35427)
    assert all(s1[k]==v and payload1[k]==v for k,v in from_expected.items())
    assert 'auxiliary_predictor_state' in payload1
    assert s1['stage1_aux_repair_gradient_batches']==5940 and s1['stage1_aux_training_windows']==844800
    state1=payload1['model_state']
    assert len(state1)==214
    assert not any('auxiliary' in k for k in state1)
    freeze_counts, selections = {}, []
    for metric in ['mae', 'mse']:
        cell = ROOT/scene['name']/(name+'-'+metric)
        manifest = cell/'stage2/runs/e23-stage2.json'
        rec = read(manifest)
        assert controller.stage2_lock(rec, plan, scene, metric) == read(cell/'validation-lock-before-tests.json')
        assert rec['stage1_checkpoint_sha256'] == s1['checkpoint_sha256']
        assert (rec['memory_sha256'], rec['key_sha256']) == (s1['memory_sha256'], s1['key_sha256'])
        state2 = torch.load(rec['checkpoint'], map_location='cpu', weights_only=False)['model_state']
        assert all(torch.equal(value, state2['repair.'+key]) for key, value in state1.items())
        assert not any('auxiliary' in k for k in state2)
        freeze_counts[metric] = len(state1)
        score = read(cell/'evaluation/twenty-test-draws.json')
        selections.append(dict(selection_metric=metric, selected_epoch=rec['best_epoch'],
            validation_mse=rec['best_val_mse'], validation_mae=rec['best_val_mae'],
            test_mse=score['test_mean']['mse'], test_mae=score['test_mean']['mae'],
            selection=str(manifest), checkpoint_sha256=rec['checkpoint_sha256'],
            role='primary' if metric == 'mae' else 'secondary from same training trajectory'))
    references = {}
    for filename in ['ETTh1_E23_v72_long_gap_moments_seed2023_20261005.csv',
                     'ETTh1_E23_v69_T1_other_missing_seed2023_20261005.csv']:
        with (DOCS/filename).open() as f:
            for row in csv.DictReader(f):
                if row['scenario'] == scene['name']:
                    references.setdefault(row['model'], row)
    paths = {name: primary/'evaluation/twenty-test-draws.json',
             'E23-v72': Path(scene['v72_evaluation']),
             'E23-v69': Path(scene['v69_evaluation']), 'T1': Path(scene['t1_evaluation'])}
    rows, draws, pairs, source_inventory = [], [], [], {}
    scores = {}
    for model, evaluation in paths.items():
        score = scores[model] = read(evaluation)
        manifest = Path(score['selection_manifest'])
        rec = read(manifest)
        assert sha(manifest) == score['selection_manifest_sha256']
        assert sha(rec['checkpoint']) == rec['checkpoint_sha256'] == score['checkpoint_sha256']
        assert score['test_mask_bank_sha256'] == scene['test_bank_sha256']
        assert score['validation_mask_sha256'] == scene['validation_mask_sha256']
        assert len(score['test_draws']) == 20
        mean = {k: float(np.mean([d[k] for d in score['test_draws']])) for k in ['mse', 'mae']}
        std = {k: float(np.std([d[k] for d in score['test_draws']], ddof=1)) for k in mean}
        assert all(abs(mean[k]-score['test_mean'][k]) < 1e-12 for k in mean)
        assert all(abs(std[k]-score['test_mask_std_ddof1'][k]) < 1e-12 for k in std)
        cost = score['training_resources']
        assert cost['resource_status'] == 'measured_all_stages'
        cost_paths = [Path(p).resolve(strict=True) for p in cost['resource_record_paths']]
        assert len(cost_paths) == len(set(cost_paths)), 'duplicated resource record'
        resources = [read(p) for p in cost_paths]
        assert sorted(n for r in resources for n in r['stages']) == ['forecast', 'repair']
        times = {n: sum(r['stages'].get(n, {}).get('wall_seconds', 0) for r in resources) for n in ['repair', 'forecast']}
        preparation = sum(r['preparation_and_reporting_seconds'] for r in resources)
        seconds = sum(r['full_workflow_seconds'] for r in resources)
        assert abs(seconds-sum(times.values())-preparation) < 1e-6
        assert abs(seconds-cost['full_workflow_seconds']) < 1e-6
        allocated = max(r['peak_allocated_bytes'] for r in resources)/2**30
        reserved = max(r['peak_reserved_bytes'] for r in resources)/2**30
        assert abs(allocated-cost['peak_allocated_gib']) < 1e-12
        assert abs(reserved-cost['peak_reserved_gib']) < 1e-12
        val = rec.get('best_validation') or dict(mse=rec['best_val_mse'], mae=rec['best_val_mae'])
        reference = references.get(model, {})
        if reference:
            assert all(abs(float(reference[k])-mean[k]) < 1e-12 for k in mean)
            assert reference['checkpoint_sha256'] == rec['checkpoint_sha256']
        repair_mse = s1['best_val_repair_mse'] if model == name else float(reference['stage1_validation_repair_mse'])
        rows.append(dict(dataset='ETTh1',seq_len=96,pred_len=96,scenario=scene['name'],
            start_rate=.1,block_len=16,actual_test_missing_fraction=scene['actual_test_missing_fraction'],
            test_masks=20,model=model,training_seed=rec['seed'],selection_metric='mae',
            mse=mean['mse'],mae=mean['mae'],mask_mse_std_ddof1=std['mse'],mask_mae_std_ddof1=std['mae'],
            validation_mse=val['mse'],validation_mae=val['mae'],full_training_seconds=seconds,
            full_training_minutes=seconds/60,repair_stage_seconds=times['repair'],forecast_stage_seconds=times['forecast'],
            preparation_and_reporting_seconds=preparation,peak_allocated_gib=allocated,peak_reserved_gib=reserved,
            training_gpu=cost['gpu'],training_device_uuids=';'.join(r['device']['uuid'] for r in resources),
            stage1_batch_size=16 if model=='T1' else 128,stage2_batch_size=128,
            stage1_precision='bfloat16' if model=='T1' else 'float32',stage2_precision='float32',
            stage2_repair_update_policy=rec.get('repair_update_policy','joint_author_policy'),
            stage2_forecast_loss='mse' if model=='T1' else 'mse_plus_0p2_mae',
            stage1_gap_structure=rec.get('stage1_gap_structure','none'),
            point_context_readout=rec.get('point_context_readout','none'),
            point_context_scope=rec.get('point_context_scope',0),front_level_context=rec.get('front_level_context','none'),front_level_extra_parameters=rec.get('front_level_extra_parameters',0),front_context_features=rec.get('front_context_features',0),front_evidence_features=rec.get('front_evidence_features',rec.get('front_context_features',0)),
            point_context_extra_parameters=rec.get('point_context_extra_parameters',0),
            stage1_validation_repair_mse=repair_mse,
            repair_parameters=plan['repair_parameters'] if model==name else int(reference['repair_parameters']),
            predictor_parameters=35427,stage1_aux_forecast=s1['stage1_aux_forecast'] if model==name else 'none',stage1_aux_weight=.05 if model==name else 0,stage1_aux_warmup=10 if model==name else 0,stage1_aux_predictor_parameters=35427 if model==name else 0,stage1_trainable_parameters=456397 if model==name else (420973 if model in ['E23-v72','E23-v69'] else 'not_measured'),stage1_total_model_parameters=456400 if model==name else int(reference['repair_parameters']),stage1_aux_trainable_parameters=35424 if model==name else 0,stage1_aux_discarded=model==name,auxiliary_padding='native_patch_with_validity' if model==name else 'none',selected_stage2_epoch=rec['best_epoch'],
            predictor_initial_sha256=rec['predictor_initial_sha256'],checkpoint_sha256=rec['checkpoint_sha256'],
            test_bank_sha256=scene['test_bank_sha256'],validation_mask_sha256=scene['validation_mask_sha256'],
            selection=str(manifest),cached_reference=model!=name,new_training_in_this_trial=model==name))
        draws.extend(dict(model=model,**d) for d in score['test_draws'])
        for p in [evaluation,manifest,Path(rec['checkpoint']),*cost_paths]:
            source_inventory[str(p)] = sha(p)
    own, parent, t1 = rows[0],rows[1],rows[3]
    for model in ['E23-v72','E23-v69','T1']:
        for a,b,delta in zip(scores[name]['test_draws'],scores[model]['test_draws'],controller.paired_draws(scores[name],scores[model])):
            with np.load(a['window_metrics']) as x,np.load(b['window_metrics']) as y:
                assert np.array_equal(x['target_start'],y['target_start'])
            pairs.append(dict(reference=model,**delta))
    accepted = own['validation_mse']<parent['validation_mse'] and own['validation_mae']<=parent['validation_mae']
    validation_goal = all(own['validation_'+k]<t1['validation_'+k] for k in ['mse','mae'])
    development_goal = all(own[k]<t1[k] for k in ['mse','mae'])
    for suffix,data in [('.csv',rows),('_all_draws.csv',draws),('_paired.csv',pairs),('_selections.csv',selections)]:
        write_csv(str(PREFIX)+suffix,data)
    lines=['# E23-v76 Stage1训练期未来监督','',
        'ETTh1 96→96，起点率0.10、块长16、实际缺失约81.94%。主表按固定验证MAE选模，原20位置是开发测试。','',
        '| 模型 | MSE ± 位置std | MAE ± 位置std | 完整训练分钟 | allocated / reserved GiB |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['mse']:.6f} ± {r['mask_mse_std_ddof1']:.6f} | {r['mae']:.6f} ± {r['mask_mae_std_ddof1']:.6f} | {r['full_training_minutes']:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    lines+=['','## 验证裁定','',f"相对v72固定验证MSE变化{own['validation_mse']-parent['validation_mse']:+.6f}、MAE变化{own['validation_mae']-parent['validation_mae']:+.6f}；{'通过，保留v76候选' if accepted else '未通过，保留v72'}。测试不用于选checkpoint或晋级。",
        f'对T1固定验证双指标目标：{validation_goal}；当前20位置双指标目标：{development_goal}。','',
        '## 配对测试','', '| 对照 | MSE差 | MAE差 | MSE胜出 | MAE胜出 |','|---|---:|---:|---:|---:|']
    for model in ['E23-v72','E23-v69','T1']:
        p=[r for r in pairs if r['reference']==model]
        lines.append(f"| {model} | {np.mean([r['mse_delta'] for r in p]):+.6f} | {np.mean([r['mae_delta'] for r in p]):+.6f} | {sum(r['mse_delta']<0 for r in p)}/20 | {sum(r['mae_delta']<0 for r in p)}/20 |")
    lines+=['','## 改动与范围','',
        '运行时完全沿用v72修复器420973参数，仅复用既有Stage1 forecast_mse_v1：.05×训练未来MSE，独立原架构辅助头35427参数/seed2022，前10轮detach，第11轮起回传修复器；100轮Stage1总参数456400/实际可训练456397（辅助头3个attention scale固定参数）。辅助头随后丢弃，未来/clean只进入loss，不进入forward。辅助native patch填充与正式zero_invalid不同，未声称数值接口相同。',
        '原35427参数预测器及初始化保持，Stage2完整修复器冻结，原MSE+0.2MAE/uniform/lr1e-4配方及padding保持。两阶段各100轮/6600更新，seed2023，batch128/FP32。MAE主/MSE次来自同一次训练，均在测试前锁定。',
        'v72/v69/T1准确性及资源均为同场景缓存。T1仍为作者修复配置seed502、Stage1 batch16/BF16、Stage2纯MSE联合更新；资源比较须保留batch/精度/更新策略差异。',
        '完整训练时间含准备、所有训练/验证和保存，排除最终测试与额外诊断；显存为PyTorch allocator峰值。次选择不增加训练成本。',
        '开发20位置反复曝光，不是盲测；另20新位置已在正式训练前锁定，首次达到验证及开发目标的候选才做一次确认。单seed、同一未来测试区间的位置标准差不代表训练seed波动或跨数据集泛化。',
        '本轮零产品源码变化，复用已审1173通过/9个继承同名失败/1跳过的基线；9项既有辅助行为与5项controller重新通过。源码、配置和controller在CUDA预检后冻结，旧产物受保护，未合入主E23。','',
        f'[CSV]({PREFIX.name}.csv) · [逐组]({PREFIX.name}_all_draws.csv) · [配对]({PREFIX.name}_paired.csv) · [双选模]({PREFIX.name}_selections.csv)','']
    PREFIX.with_suffix('.md').write_text('\n'.join(lines))
    audit=dict(status='published',rows=dict(main=len(rows),draws=len(draws),paired=len(pairs),selections=len(selections)),
        repair_tensors_bitwise_frozen=freeze_counts,validation_upgrade_vs_parent=accepted,
        retained_candidate=name if accepted else 'E23-v72',validation_goal_vs_T1=validation_goal,
        development_goal_vs_T1=development_goal,test_used_for_selection=False,source_inventory=source_inventory,
        reports={str(Path(str(PREFIX)+s)):sha(str(PREFIX)+s) for s in ['.csv','_all_draws.csv','_paired.csv','_selections.csv','.md']})
    (ROOT/'report-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_inventory','reports']}))


if __name__ == '__main__':
    main()
