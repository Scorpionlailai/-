"""Publish only paired pure-MSE frozen records; keep mixed-loss history separate."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
STEM='ETTh1_E23_v69_T1_pureMSE_frozen_four_scenarios_20261007'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_write(path,rows):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_record(path,scene,metric):
    value=read(path);record=read(value['selection_manifest'])
    assert sha(value['selection_manifest'])==value['selection_manifest_sha256']
    assert sha(record['checkpoint'])==record['checkpoint_sha256']==value['checkpoint_sha256']
    assert record['loss']=='forecast_mse_only'
    assert record['repair_update_policy']=='frozen_v1'
    assert record['epochs_completed']==100
    assert record.get('repair_retention','none')=='none'
    assert record['selection_metric']==metric
    assert record.get('training_seed',record['seed'])==2023
    assert record['predictor_initial_sha256']=='fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561'
    assert value['validation_mask_sha256']==scene['validation_mask_sha256']
    assert value['test_mask_bank_sha256']==scene['test_bank_sha256']
    assert value['metric_space']=='train_standardized' and len(value['test_draws'])==20
    expected=record.get('best_validation',dict(mse=record.get('best_val_mse'),mae=record.get('best_val_mae')))
    for key in ['mse','mae']:
        assert abs(value['validation_replay'][key]-expected[key])<=1e-6
        numbers=[d[key] for d in value['test_draws']]
        assert np.isfinite(numbers).all()
        assert abs(np.mean(numbers)-value['test_mean'][key])<1e-12
        assert abs(np.std(numbers,ddof=1)-value['test_mask_std_ddof1'][key])<1e-12
    for draw in value['test_draws']:
        assert (draw['windows'],draw['target_elements'])==(2688,1806336)
        assert sha(draw['window_metrics'])==draw['window_metrics_sha256']
    return value,record


def resource_details(value):
    aggregate=value['training_resources'];phases={};gpu_uuids=set()
    for path in aggregate['resource_record_paths']:
        r=read(path);assert r['status']=='completed'
        gpu_uuids.add(r['device']['uuid'].replace('GPU-',''))
        for name,stage in r['stages'].items():
            assert name not in phases
            phases[name]=stage
    assert gpu_uuids=={'1315e15c-d657-e61f-bde5-6e0b38e6e54b'}
    return dict(two_stage_recorded_cost_seconds=aggregate['full_workflow_seconds'],
                stage1_recorded_seconds=phases['repair']['wall_seconds'],
                stage2_training_seconds=phases['forecast']['wall_seconds'],
                peak_allocated_gib=aggregate['peak_allocated_gib'],
                peak_reserved_gib=aggregate['peak_reserved_gib'],gpu=aggregate['gpu'],
                resource_paths=';'.join(aggregate['resource_record_paths']),
                resource_scope='cached_stage1_plus_measured_new_stage2')


def main():
    plan=read(ROOT/'plan.json');frozen=read(ROOT/'frozen-definition.json')
    assert read(ROOT/'complete.json')['status']=='completed'
    assert sha(ROOT/'plan.json')==frozen['plan_sha256']
    assert all(sha(p)==h for p,h in frozen['protected'].items())
    assert all(sha(Path(plan['work'])/p)==h for p,h in frozen['sources'].items())
    rows,draws,paired,summaries=[],[],[],[]
    for scene in plan['scenarios']:
        metrics={}
        for metric in ['mae','mse']:
            model_results={}
            for label,directory in [('E23-v69-pureMSE','E23-v69'),('T1-pureMSE-frozen','T1')]:
                path=ROOT/directory/scene['name']/f'evaluation_{metric}/twenty-test-draws.json'
                value,record=verify_record(path,scene,metric);model_results[label]=value
                stage1_path=record.get('stage1_manifest',record.get('repair_selection_manifest'))
                stage1=read(stage1_path)
                row=dict(scene=scene['name'],start_rate=scene['start_rate'],block_len=scene['block_len'],
                    actual_missing_fraction=scene['actual_test_missing_fraction'],model=label,
                    selection_metric=metric,selected_epoch=record['best_epoch'],
                    mse_mean=value['test_mean']['mse'],mae_mean=value['test_mean']['mae'],
                    mse_std_ddof1=value['test_mask_std_ddof1']['mse'],
                    mae_std_ddof1=value['test_mask_std_ddof1']['mae'],test_groups=20,windows_per_group=2688,
                    forecast_loss='pure_mse',repair_update_policy='frozen_v1',
                    stage1_training_seed=502 if directory=='T1' else 2023,
                    stage1_epochs_completed=stage1['epochs_completed'],stage1_reused=True,
                    stage2_training_seed=2023,stage2_epochs=100,stage2_optimizer_steps=6600,
                    stage2_batch_size=128,stage2_learning_rate=1e-4,stage2_weight_decay=0.,
                    predictor_initial_sha256=record['predictor_initial_sha256'],predictor_parameters=35427,
                    predictor_trainable_parameters=35424,
                    forecast_padding='replicate_last_value_v1' if directory=='T1' else 'zero_invalid_v1',
                    evaluation_manifest=str(path),selection_manifest=value['selection_manifest'],
                    **resource_details(value))
                rows.append(row)
                for draw in value['test_draws']:
                    draws.append(dict(scene=scene['name'],model=label,selection_metric=metric,
                                      group_id=draw['group_id'],mask_seed=draw['mask_seed'],
                                      mse=draw['mse'],mae=draw['mae'],mask_sha256=draw['mask_sha256']))
            a=model_results['E23-v69-pureMSE'];b=model_results['T1-pureMSE-frozen']
            for x,y in zip(a['test_draws'],b['test_draws']):
                assert all(x[k]==y[k] for k in ['group_id','mask_seed','mask_sha256','windows','target_elements'])
                paired.append(dict(scene=scene['name'],selection_metric=metric,
                    group_id=x['group_id'],mask_seed=x['mask_seed'],
                    e23_mse=x['mse'],t1_mse=y['mse'],e23_minus_t1_mse=x['mse']-y['mse'],
                    e23_mae=x['mae'],t1_mae=y['mae'],e23_minus_t1_mae=x['mae']-y['mae']))
            metrics[metric]=dict(e23=a['test_mean'],t1=b['test_mean'],
                                both_better=all(a['test_mean'][k]<b['test_mean'][k] for k in ['mse','mae']))
        summaries.append(dict(scene=scene['name'],selections=metrics))
    csv_write(DOCS/f'{STEM}.csv',rows)
    csv_write(DOCS/f'{STEM}_all_draws.csv',draws)
    csv_write(DOCS/f'{STEM}_paired.csv',paired)
    table=['# E23-v69 / T1：纯 MSE、冻结修复器四场景对照','',
           '双方Stage2纯MSE、全部修复器参数和状态冻结、同原预测头与初始化、seed2023、Adam lr1e-4、batch128、FP32、100轮。主表验证MAE选模，MSE次选同轨迹单列CSV。各场景20组共享开发测试缺失。','',
           '| 起点率/块长 | 实际缺失率 | E23 MSE | T1 MSE | E23 MAE | T1 MAE |',
           '|---|---:|---:|---:|---:|---:|']
    for scene,s in zip(plan['scenarios'],summaries):
        result=s['selections']['mae'];a,b=result['e23'],result['t1']
        table.append(f'| {scene["start_rate"]:.2f}/{scene["block_len"]} | {100*scene["actual_test_missing_fraction"]:.2f}% | {a["mse"]:.6f} | {b["mse"]:.6f} | {a["mae"]:.6f} | {b["mae"]:.6f} |')
    table+=['','## 资源','',
            '| 场景 | 模型 | 两阶段记录成本秒 | 新Stage2训练秒 | 峰值allocated / reserved GiB |',
            '|---|---|---:|---:|---:|']
    for row in rows:
        if row['selection_metric']=='mae':
            table.append(f'| {row["scene"]} | {row["model"]} | {row["two_stage_recorded_cost_seconds"]:.2f} | {row["stage2_training_seconds"]:.2f} | {row["peak_allocated_gib"]:.3f} / {row["peak_reserved_gib"]:.3f} |')
    table+=['', '## 解释','',
            '- 双方Stage1均复用已选对应场景权重；表中两阶段成本为Stage1历史实测加新Stage2完整命令实测，不是本次从头两阶段墙钟时间。测试时间不计入训练。',
            '- T1 Stage1作者配置seed502，E23 Stage1固定v69方案seed2023；Stage2双方同seed2023、头SHA、训练顺序和动态缺失。',
            '- 按用户要求T1仍用原末尾复制补齐，E23仍用零补齐。',
            '- 单训练种子，20组标准差为缺失位置波动；这些位置已用于开发，不作为盲测或多训练种子结果。',
            '- 旧混合损失对照保留在原报告。本表只比较双方纯MSE冻结版本；不混用旧T1或旧E23预测结果。',
            f'- 原始记录、checkpoint、逐窗口误差及SHA：{ROOT}', '',
            f'[CSV](./{STEM}.csv) · [全部测试组](./{STEM}_all_draws.csv) · [配对误差](./{STEM}_paired.csv)']
    with (DOCS/f'{STEM}.md').open('x') as f:f.write('\n'.join(table)+'\n')
    result=dict(status='verified',scenarios=summaries,
                all_four_better=all(s['selections']['mae']['both_better'] for s in summaries),
                report=str(DOCS/f'{STEM}.md'),csv=str(DOCS/f'{STEM}.csv'))
    with (ROOT/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
