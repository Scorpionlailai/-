"""Verify locked records, then publish the paired four-scenario comparison."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
STEM = 'ETTh1_T1_frozen_stage2_E23_v69_four_scenarios_20261007'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified(path, scene, frozen=False):
    value = read(path)
    assert value['test_mask_bank_sha256'] == scene['test_bank_sha256']
    assert value['validation_mask_sha256'] == scene['validation_mask_sha256']
    assert value['metric_space'] == 'train_standardized'
    assert len(value['test_draws']) == 20
    selection = read(value['selection_manifest'])
    assert sha(value['selection_manifest']) == value['selection_manifest_sha256']
    assert sha(selection['checkpoint']) == value['checkpoint_sha256'] == selection['checkpoint_sha256']
    for metric in ['mse', 'mae']:
        values = [draw[metric] for draw in value['test_draws']]
        assert np.isfinite(values).all()
        assert abs(float(np.mean(values))-value['test_mean'][metric]) < 1e-12
        assert abs(float(np.std(values,ddof=1))-value['test_mask_std_ddof1'][metric]) < 1e-12
    if frozen:
        lock = read(Path(path).parent/'validation-lock.json')
        assert not lock['test_metrics_seen']
        assert lock['checkpoint_sha256'] == value['checkpoint_sha256']
        assert selection['epochs_completed'] == 100
        assert selection['training_windows'] == 844800 and selection['optimizer_steps'] == 6600
        assert selection['repair_update_policy'] == 'frozen_v1'
        assert selection['repair_trainable_parameters'] == 0
        assert selection['loss'] == 'forecast_mse_plus_0p2_mae'
        assert selection['predictor_parameters'] == 35427
        assert selection['frozen_state_verified_epochs'] == 100
        assert selection['forecast_padding'] == 'replicate_last_value_v1'
        history = [json.loads(line) for line in (Path(value['selection_manifest']).parent/'epochs.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in history] == list(range(1,101))
        assert {r['frozen_imputer_sha256'] for r in history} == {selection['frozen_imputer_sha256']}
        for key in ['mse','mae']:
            assert abs(lock['validation_replay'][key]-selection['best_validation'][key]) <= 1e-6
        assert selection['best_epoch'] == min(history,key=lambda r:r['validation'][selection['selection_metric']])['epoch']
        for draw in value['test_draws']:
            assert (draw['windows'],draw['target_elements']) == (2688,1806336)
            assert sha(draw['window_metrics']) == draw['window_metrics_sha256']
    return value,selection


def resources(value):
    aggregate = value['training_resources']
    records = [read(path) for path in aggregate['resource_record_paths']]
    assert records and all(r['status']=='completed' for r in records)
    assert all(r['device']['uuid'].replace('GPU-','')=='1315e15c-d657-e61f-bde5-6e0b38e6e54b' for r in records)
    phases = {}
    for record in records:
        for name,stage in record['stages'].items():
            assert name not in phases
            phases[name] = stage
    repair_phase = phases.get('repair') or phases.get('stage1')
    forecast_phase = phases.get('forecast') or phases.get('stage2')
    assert repair_phase and forecast_phase
    forecast_record = next(r for r in records if 'forecast' in r['stages'] or 'stage2' in r['stages'])
    return dict(full_training_cost_seconds=aggregate['full_workflow_seconds'],
                training_stages_seconds=aggregate['training_stages_seconds'],
                stage1_seconds=repair_phase['wall_seconds'],stage2_seconds=forecast_phase['wall_seconds'],
                stage2_command_seconds=forecast_record['full_workflow_seconds'],
                peak_allocated_gib=aggregate['peak_allocated_gib'],
                peak_reserved_gib=aggregate['peak_reserved_gib'],
                stage2_peak_allocated_gib=forecast_phase['peak_allocated_bytes']/1024**3,
                stage2_peak_reserved_gib=forecast_phase['peak_reserved_bytes']/1024**3,
                gpu=aggregate['gpu'],gpu_uuid='1315e15c-d657-e61f-bde5-6e0b38e6e54b',
                resource_status=aggregate['resource_status'],
                resource_paths=';'.join(aggregate['resource_record_paths']))


def main():
    plan = read(ROOT/'plan.json')
    head_audit = read(ROOT/'head-parameter-audit.json')
    assert head_audit['total_parameters'] == 35427
    assert head_audit['trainable_parameters'] == 35424
    assert read(ROOT/'training-complete.json')['status'] == 'completed'
    frozen = read(ROOT/'frozen-definition.json')
    assert sha(ROOT/'plan.json') == frozen['plan_sha256']
    assert all(sha(Path(plan['work'])/p)==h for p,h in frozen['sources'].items())
    assert all(sha(p)==h for p,h in frozen['protected'].items())
    addon = read(ROOT/'plan_sr015_bl8.json')
    addon_frozen = read(ROOT/'frozen-addendum.json')
    assert read(ROOT/'addendum-complete.json')['status'] == 'completed'
    assert sha(ROOT/'plan_sr015_bl8.json') == addon_frozen['plan_sha256']
    assert addon['base_plan_sha256'] == frozen['plan_sha256']
    assert all(sha(p)==h for p,h in addon_frozen['protected'].items())
    plan = {**plan, 'scenarios': sorted(plan['scenarios']+addon['scenarios'],
                                       key=lambda s:(s['block_len'],s['start_rate']))}
    rows,draw_rows,paired,summary = [],[],[],[]
    long_gap_best = None
    for scene in plan['scenarios']:
        e23,sel_e23 = verified(scene['e23_evaluation'],scene)
        new,sel_new = verified(ROOT/scene['name']/'evaluation_mae/twenty-test-draws.json',scene,True)
        new_mse,sel_mse = verified(ROOT/scene['name']/'evaluation_mse/twenty-test-draws.json',scene,True)
        joint,sel_joint = verified(scene['t1_evaluation'],scene)
        assert e23['predictor_initial_sha256'] == new['predictor_initial_sha256'] == plan['predictor_initial_sha256']
        assert sel_e23['loss'] == sel_new['loss']
        assert sha(plan['e23_config']) == sel_e23['e23_config_sha256']
        assert sel_e23['repair_update_policy'] == sel_new['repair_update_policy'] == 'frozen_v1'
        assert sel_e23['learning_rate'] == sel_new['training']['learning_rate']
        assert sel_e23['batch_size'] == sel_new['training']['batch_size'] == 128
        assert sel_e23['epochs_completed'] == sel_new['epochs_completed'] == 100
        for a,b,c in zip(e23['test_draws'],new['test_draws'],joint['test_draws']):
            for key in ['group_id','mask_seed','mask_sha256','windows','target_elements']:
                assert a[key] == b[key] == c[key]
            paired.append(dict(scene=scene['name'],group_id=a['group_id'],mask_seed=a['mask_seed'],
                              e23_mse=a['mse'],t1_frozen_mse=b['mse'],e23_minus_t1_frozen_mse=a['mse']-b['mse'],
                              e23_mae=a['mae'],t1_frozen_mae=b['mae'],e23_minus_t1_frozen_mae=a['mae']-b['mae'],
                              t1_joint_mse=c['mse'],t1_joint_mae=c['mae']))
        comparisons = [('E23-v69-frozen',e23,sel_e23,scene['e23_evaluation']),
                       ('T1-frozen-E23-aligned',new,sel_new,str(ROOT/scene['name']/'evaluation_mae/twenty-test-draws.json')),
                       ('T1-joint-previous',joint,sel_joint,scene['t1_evaluation']),
                       ('T1-frozen-E23-aligned-MSE-selection',new_mse,sel_mse,str(ROOT/scene['name']/'evaluation_mse/twenty-test-draws.json'))]
        if scene['name'] == 'sr010_bl16':
            v77_path = ROOT.parent/'e23_v77_cross_variable_level_seed2023_20261006/sr010_bl16/E23-v77-mae/evaluation/twenty-test-draws.json'
            v77,sel_v77 = verified(v77_path,scene)
            assert sel_v77['loss'] == sel_new['loss']
            assert sel_v77['repair_update_policy'] == 'frozen_v1'
            assert sel_v77['epochs_completed'] == 100
            assert sel_v77['batch_size'] == 128 and sel_v77['learning_rate'] == 1e-4
            assert v77['predictor_initial_sha256'] == new['predictor_initial_sha256']
            comparisons.append(('E23-v77-frozen-supplement',v77,sel_v77,str(v77_path)))
            for a,b in zip(v77['test_draws'],new['test_draws']):
                for key in ['group_id','mask_seed','mask_sha256','windows','target_elements']:
                    assert a[key] == b[key]
            long_gap_best = dict(model='E23-v77',e23=v77['test_mean'],t1_frozen=new['test_mean'])
        for label,value,selection,evaluation_path in comparisons:
            cost = resources(value)
            is_new = label.startswith('T1-frozen')
            parent_stage1 = read(selection.get('repair_selection_manifest',selection.get('stage1_manifest')))
            row = dict(scene=scene['name'],start_rate=scene['start_rate'],block_len=scene['block_len'],
                       actual_test_missing_fraction=scene['actual_test_missing_fraction'],
                       model=label,mse_mean=value['test_mean']['mse'],mae_mean=value['test_mean']['mae'],
                       mse_std_ddof1=value['test_mask_std_ddof1']['mse'],
                       mae_std_ddof1=value['test_mask_std_ddof1']['mae'],
                       test_groups=20,windows_per_group=2688,
                       stage1_training_seed=502 if label.startswith('T1') else 2023,
                       stage1_epochs_completed=parent_stage1['epochs_completed'],
                       stage1_selected_epoch=parent_stage1['best_epoch'],
                       stage2_training_seed=selection.get('training_seed',selection.get('seed')),
                       stage2_epochs=selection['epochs_completed'],stage2_batch_size=128,
                       stage2_learning_rate=1e-4,stage2_weight_decay=0.,
                       forecast_loss=selection.get('loss',selection.get('training',{}).get('loss')),
                       selection_metric=selection.get('selection_metric',selection.get('training',{}).get('selection_metric')),
                       selected_epoch=selection['best_epoch'],
                       repair_update_policy=selection.get('repair_update_policy','joint'),
                       forecast_padding=selection.get('forecast_padding','replicate_last_value_v1'),
                       predictor_initial_sha256=value['predictor_initial_sha256'],
                       predictor_parameters=value['predictor_parameters'],
                       predictor_trainable_parameters=head_audit['trainable_parameters'],stage1_reused=is_new,
                       training_time_status='cached_Stage1_plus_new_Stage2' if is_new else 'recorded_full_training',
                       stage2_command_scope='Stage2 only' if is_new or label.startswith('E23') else 'combined original Stage1+Stage2 command',
                       selection_manifest=value['selection_manifest'],evaluation_manifest=evaluation_path,**cost)
            rows.append(row)
            for draw in value['test_draws']:
                draw_rows.append(dict(scene=scene['name'],model=label,selection_metric=row['selection_metric'],
                                      group_id=draw['group_id'],mask_seed=draw['mask_seed'],mask_sha256=draw['mask_sha256'],
                                      mse=draw['mse'],mae=draw['mae'],windows=draw['windows'],target_elements=draw['target_elements']))
        sub = [p for p in paired if p['scene']==scene['name']]
        summary.append(dict(scene=scene['name'],actual_missing=scene['actual_test_missing_fraction'],
                            e23_mse=e23['test_mean']['mse'],t1_mse=new['test_mean']['mse'],
                            e23_mae=e23['test_mean']['mae'],t1_mae=new['test_mean']['mae'],
                            e23_mse_reduction_percent=(1-e23['test_mean']['mse']/new['test_mean']['mse'])*100,
                            e23_mae_reduction_percent=(1-e23['test_mean']['mae']/new['test_mean']['mae'])*100,
                            e23_mse_mask_wins=sum(p['e23_minus_t1_frozen_mse']<0 for p in sub),
                            e23_mae_mask_wins=sum(p['e23_minus_t1_frozen_mae']<0 for p in sub)))
    write_csv(DOCS/f'{STEM}.csv',rows)
    write_csv(DOCS/f'{STEM}_all_test_draws.csv',draw_rows)
    write_csv(DOCS/f'{STEM}_paired.csv',paired)
    text = ['# T1 冻结修复器与 E23-v69：四个缺失场景','',
            '## 主对照：验证 MAE 选模','',
            '下表均为相同 20 组测试缺失位置的均值。E23 为已有 v69 记录；T1 从对应 Stage1 权重冻结后重新训练 Stage2。', '',
            '| 起点率 / 块长 | 实际缺失率 | E23-v69 MSE | T1 冻结 MSE | E23-v69 MAE | T1 冻结 MAE | E23 MSE 胜出组数 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        scene = next(v for v in plan['scenarios'] if v['name']==s['scene'])
        text.append(f'| {scene["start_rate"]:.2f} / {scene["block_len"]} | {100*s["actual_missing"]:.2f}% | {s["e23_mse"]:.6f} | {s["t1_mse"]:.6f} | {s["e23_mae"]:.6f} | {s["t1_mae"]:.6f} | {s["e23_mse_mask_wins"]}/20 |')
    if long_gap_best:
        text += ['', '### 长缺失当前保留最佳版本（单列）','',
                 f'0.10/16 下，E23-v77 的 MSE/MAE 为 **{long_gap_best["e23"]["mse"]:.6f} / {long_gap_best["e23"]["mae"]:.6f}**；本次冻结 T1 为 **{long_gap_best["t1_frozen"]["mse"]:.6f} / {long_gap_best["t1_frozen"]["mae"]:.6f}**。v77 使用同一 Stage2 预测损失、冻结方式、预测头初始权重、轮数和共享20组测试位置；CSV 单列补充，保持主表v69版本一致。']
    text += ['', '## Stage2 对齐与保留差异','',
             '- 两者修复器均冻结参数和缓冲状态，处于 eval 模式；仅训练原有 PatchTST，结构无改动，35,427 总参数，其中35,424可训练、3个固定注意力缩放标量。',
             '- Stage2 均使用 seed2023、相同预测头初始权重 SHA、同一训练顺序和动态缺失生成、固定验证缺失。',
             '- 损失均为 MSE + 0.2 MAE；Adam、lr1e-4、weight_decay0、batch128、FP32、100轮（每轮66更新，共6600更新），无早停。',
             '- 主对照按验证 MAE 选模；同一轨迹的验证 MSE 最优在 CSV 附表中单列，不按测试结果选模。',
             '- 用户要求 T1 保留原有末尾复制补齐；E23 记录为零补齐。预测头结构相同，补齐输入规则保留这项差异。',
             '- T1 Stage1 保持作者公开配置和已选 seed502，训练上限300轮、patience30、batch16、BF16、隐藏重建MSE；本次复用对应场景选出的 Stage1。E23 Stage1 使用自己的固定方案。',
             '- E23 冻结修复器的修复保留损失对预测头梯度为零；T1 此项不计算，预测目标与更新梯度一致。',
             '- 原 T1 联合训练记录为补充：其预测损失为纯MSE、Stage2 seed502，头初始权重不同。这次变化不能单独归因于冻结。', '',
             '## 实测资源','',
             '| 场景 | 模型 | 两阶段训练成本秒 | Stage1 秒 | Stage2 秒 | 峰值 allocated GiB | 峰值 reserved GiB |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['model'] in ['E23-v69-frozen','T1-frozen-E23-aligned','E23-v77-frozen-supplement']:
            text.append(f'| {row["scene"]} | {row["model"]} | {row["full_training_cost_seconds"]:.2f} | {row["stage1_seconds"]:.2f} | {row["stage2_seconds"]:.2f} | {row["peak_allocated_gib"]:.3f} | {row["peak_reserved_gib"]:.3f} |')
    text += ['',
             'T1 两阶段成本为原 Stage1 实测阶段时间（附原命令非阶段开销）与本次 Stage2 完整命令实测时间相加；Stage1 已复用，这个值是合成训练成本，不是本次从头两阶段的墙钟时间。CSV 同时保留单独 Stage2 命令时间、各训练阶段时间和原始资源记录路径。', '',
             '两者使用同一 RTX5090（UUID 1315e15c-d657-e61f-bde5-6e0b38e6e54b）、CPU线程4；最终20组测试不计入训练时间。显存为 PyTorch allocator 的 allocated/reserved 峰值，保留两个口径。T1 Stage1 batch16遵循作者配置，Stage2与E23统一batch128。', '',
             '## 验证与统计边界','',
             '- 四组训练均完成100轮；每轮修复器全部参数和缓冲状态 SHA 保持一致，未产生修复器梯度。',
             '- 验证回放与选模记录一致；先写验证锁，再评测共享20组缺失；每组2688窗口、1,806,336预测坐标。',
             '- 表中标准差衡量测试缺失位置变化，只有一个训练种子，不是20次独立训练。各组测试目标共享且窗口重叠，不据此宣称20次独立样本显著性。',
             '- 主对照固定为 E23-v69 四场景；长缺失当前保留最佳 v77 另列，明确版本和同一测试位置。',
             f'- 实验计划和所有权重/逐窗口误差/资源凭证：`{ROOT}`。', '',
             f'详细表：[CSV](./{STEM}.csv)；[全部测试组](./{STEM}_all_test_draws.csv)；[配对差异](./{STEM}_paired.csv)。']
    report = DOCS/f'{STEM}.md'
    with report.open('x') as stream:
        stream.write('\n'.join(text)+'\n')
    result = dict(status='verified',scenes=4,epochs_each=100,unit_tests=3,
                  frozen_state_verified=True,validation_replayed=True,shared_test_draws=20,
                  comparisons=summary,long_gap_current_best=long_gap_best,
                  report=str(report),csv=str(DOCS/f'{STEM}.csv'))
    with (ROOT/'verification.json').open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
