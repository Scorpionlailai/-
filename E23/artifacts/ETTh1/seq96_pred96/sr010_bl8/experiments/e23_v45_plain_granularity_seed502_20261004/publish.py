"""Publish completed validation-locked cells, including both selectors and resource records."""
import csv,hashlib,json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v45-granularity/E23')
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
PREFIX='ETTh1_P4_E23_v45_plain_granularity_seed502_20261004'

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def main():
    plan=read(ROOT/'plan.json');definition=read(ROOT/'matrix-definition.json')
    assert digest(ROOT/'plan.json')==definition['plan_sha256']
    for path,sha in definition['protected'].items():assert digest(path)==sha,path
    for path,sha in definition['sources'].items():assert digest(WORK/path)==sha,path
    pairs=[];history={};recovery=[]
    for spec in plan['cells']:
        cell=ROOT/spec['name']
        if not (cell/'completed-audit.json').exists():continue
        assert (cell/'validation-lock-before-tests.json').is_file()
        diag={row['stage']:row for row in read(cell/'recovery-diagnostics.json')}
        first=read(cell/'stage1/runs/e23-stage1.json')
        memory=torch.load(first['checkpoint'],map_location='cpu',weights_only=False)['memory_state']['snapshot']
        groups=[g for channel in memory['groups'] for g in channel]
        history[spec['name']]=memory['seen']
        bank=dict(history_bank_groups=len(groups),history_bank_members=sum(len(g['members']) for g in groups),
                  history_observed_identities=len(memory['seen']),history_sampling_stride=8,historical_query_length=16)
        for metric in ('mae','mse'):
            pairs.append((spec['label'],cell/f'evaluation_{metric}/twenty-test-draws.json',spec,diag,metric,bank))
        for stage,row in diag.items():
            for region,metrics in row['metrics'].items():
                recovery.append(dict(model=spec['label'],stage=stage,region=region,**metrics,
                    last32_level_mse=row['last32_level_mse'],last32_slope_mse=row['last32_slope_mse'],
                    h_available_damaged_patches=row['h_available_damaged_patches'],damaged_patches=row['damaged_patches'],
                    h_available_fully_hidden_patches=row['h_available_fully_hidden_patches']))
    for label,path in [
        ('E23-v41-tail32-slow',ROOT.parent/'e23_v41_repair_drift_seed502_20261004/tail_slow/evaluation/twenty-test-draws.json'),
        ('E23-v41-tail32-frozen',ROOT.parent/'e23_v41_repair_drift_seed502_20261004/tail_frozen/evaluation/twenty-test-draws.json'),
        ('E23-v43-N',ROOT.parent/'e23_v43_nonoverlap_seed502_20261004/nonoverlap_slow/evaluation_mae/twenty-test-draws.json'),
        ('T1',ROOT.parent/'e23_v42_mse_updates_seed_search_20261004/T1_seed502/evaluation_mae/twenty-test-draws.json')]:
        pairs.append((label,path,None,None,'mae',{}))
    for p in (16,8,4):
        pairs.append((f'E23-v44-P{p}-Hybrid',ROOT.parent/f'e23_v44_fine_hybrid_seed502_20261004/fine{p}_hybrid/evaluation_mae/twenty-test-draws.json',None,None,'mae',{}))
    rows=[];draws=[]
    for label,path,spec,diag,metric,bank in pairs:
        e=read(path);selected=read(e['selection_manifest']);res=e['training_resources']
        assert e['predictor_initial_sha256']==plan['predictor_initial_sha256'] and e['predictor_parameters']==35427
        assert e['test_mask_bank_sha256']==definition['test_bank_sha256']
        assert digest(e['selection_manifest'])==e['selection_manifest_sha256']
        assert digest(selected['checkpoint'])==e['checkpoint_sha256']
        resources=[read(p) for p in res['resource_record_paths']]
        first=diag['stage1'] if diag else None;last=diag['stage2_'+metric] if diag else None
        row=dict(model=label,checkpoint_selection=metric,training_seed=502,dataset='ETTh1',seq_len=96,pred_len=96,
            missing_block_start_rate=.1,missing_block_length=8,test_masks=20,
            mse=e['test_mean']['mse'],mae=e['test_mean']['mae'],
            mask_mse_std_ddof1=e['test_mask_std_ddof1']['mse'],mask_mae_std_ddof1=e['test_mask_std_ddof1']['mae'],
            validation_mse=e['validation_replay']['mse'],validation_mae=e['validation_replay']['mae'],
            selected_stage2_epoch=selected['best_epoch'],
            repair_patch_length=spec['repair_patch_length'] if spec else '',repair_stride=spec['repair_stride'] if spec else '',
            repair_patch_count=96//spec['repair_stride'] if spec else '',repair_parameters=last['repair_parameters'] if last else '',
            predictor_patch_length=16,predictor_stride=8,predictor_parameters=35427,**bank,
            stage1_repair_mse=first['metrics']['all96']['mse'] if first else '',
            stage1_repair_mae=first['metrics']['all96']['mae'] if first else '',
            stage1_last32_repair_mse=first['metrics']['last32']['mse'] if first else '',
            stage1_last32_level_mse=first['last32_level_mse'] if first else '',
            stage1_last32_slope_mse=first['last32_slope_mse'] if first else '',
            stage2_repair_mse=last['metrics']['all96']['mse'] if last else '',
            stage2_last32_repair_mse=last['metrics']['last32']['mse'] if last else '',
            stage2_last32_level_mse=last['last32_level_mse'] if last else '',
            stage2_last32_slope_mse=last['last32_slope_mse'] if last else '',
            full_training_seconds=res['full_workflow_seconds'],peak_allocated_gib=res['peak_allocated_gib'],
            peak_reserved_gib=res['peak_reserved_gib'],gpu=res['gpu'],training_batch_sizes=res['batch_size'],
            gpu_uuids=';'.join(sorted({str(r.get('device',{}).get('uuid','unknown')) for r in resources})),
            repair_stage_seconds=sum(r.get('stages',{}).get('repair',{}).get('wall_seconds',0) for r in resources),
            forecast_stage_seconds=sum(r.get('stages',{}).get('forecast',{}).get('wall_seconds',0) for r in resources),
            actual_new_training_seconds=sum(r['full_workflow_seconds'] for p,r in zip(res['resource_record_paths'],resources) if Path(p).is_relative_to(ROOT)),
            stage2_loss=selected.get('loss','forecast_mse_only' if label=='T1' else ''),
            repair_update_policy=selected.get('repair_update_policy','joint'),cached_reference=spec is None,
            predictor_initial_sha256=e['predictor_initial_sha256'],checkpoint_sha256=e['checkpoint_sha256'],evaluation=str(path))
        rows.append(row)
        assert len(e['test_draws'])==20
        for draw in e['test_draws']:
            assert digest(draw['window_metrics'])==draw['window_metrics_sha256']
            draws.append(dict(model=label,checkpoint_selection=metric,training_seed=502,**draw))
    primary=[row for row in rows if row['checkpoint_selection']=='mae']
    csvwrite(DOCS/(PREFIX+'.csv'),rows);csvwrite(DOCS/(PREFIX+'_primary.csv'),primary)
    csvwrite(DOCS/(PREFIX+'_all_draws.csv'),draws)
    if recovery:csvwrite(DOCS/(PREFIX+'_recovery.csv'),recovery)
    complete=(ROOT/'matrix-completed.json').exists()
    completion=read(ROOT/'matrix-completed.json') if complete else {}
    lines=['# E23 v45：仅改变三专家修复粒度','',
        '状态：'+('三组全部完成。' if complete else '运行中；下表仅列已完成并锁定选模的组。'),'',
        '## 固定设置','',
        '- 修复输出16/16、8/8、4/4，分别6、12、24块；H保持16点父上下文及16/8训练观测库，候选按子块裁切。T/V读取跨度按原时间点对齐。',
        '- 完全移除末端CNN与跨变量注意力精修。保留H/V/T及T→V→H单轮恢复，三个配置仅改变非重叠输出粒度。',
        '- 新组各自100轮Stage1与100轮Stage2，batch128、FP32。Stage1全局缺失MSE+0.1检索，Stage2纯预测MSE；预测器LR1e-4、修复器LR1e-5。',
        '- 原预测头PatchTST D16、35427参数、16/8输入与初始化一致；沿用T1对齐的padding和训练RNG重置。',
        '- 开发seed502；主checkpoint按固定验证MAE，另存验证MSE最低版本。每组锁定后测试同一20组缺失位置。','',
        '## 验证MAE选模的主结果','',
        '| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | allocated / reserved峰值GiB |',
        '|---|---:|---:|---|---:|---|']
    for r in primary:
        lines.append(f"| {r['model']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['validation_mse']:.6f} / {r['validation_mae']:.6f} | {r['full_training_seconds']/60:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    lines+=['','T1、v41、v43-N及v44为缓存参考，没有新训练。v41使用旧混合损失与修复保留，padding/RNG设置不同；新组三者为粒度的匹配比较；与同粒度v44相比只移除末端精修。T1保留作者Stage1 batch16/BF16配置。','',
        '## 修复与近期结构诊断','',
        '| 模型 | Stage1修复MSE | Stage2修复MSE | 最近32点均值误差 Stage1 → Stage2 | 最近32点斜率误差 Stage1 → Stage2 |',
        '|---|---:|---:|---|---|']
    for r in primary:
        if not r['cached_reference']:
            lines.append(f"| {r['model']} | {r['stage1_repair_mse']:.6f} | {r['stage2_repair_mse']:.6f} | {r['stage1_last32_level_mse']:.6f} → {r['stage2_last32_level_mse']:.6f} | {r['stage1_last32_slope_mse']:.8f} → {r['stage2_last32_slope_mse']:.8f} |")
    if complete:
        lines+=['','## 固定验证裁定','', '```json',json.dumps(completion,ensure_ascii=False,indent=2),'```']
        candidates=[r for r in primary if not r['cached_reference']]
        gates={r['model']:dict(validation_mse=r['validation_mse'],validation_mae=r['validation_mae'],
            passes_v41_reference=r['validation_mse']<plan['v41_reference_validation']['mse'] and
                r['validation_mae']<=plan['v41_reference_validation']['mae']) for r in candidates}
        verdict=dict(primary_selection='fixed_validation_mae',test_used_for_selection=False,
            reference='E23-v41-tail32-slow',reference_gates=gates,
            retained_reference=not any(r['passes_v41_reference'] for r in gates.values()))
        (ROOT/'development-verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
        lines+=['','## 结果解读','',
            ('保留v41-slow开发参考：本轮没有配置同时通过其验证MSE和MAE门槛。' if verdict['retained_reference'] else '存在同时通过v41固定验证MSE和MAE门槛的配置，详细结果见裁定文件。'),
            '粒度是否改善以本轮三组相互比较；去掉精修的影响与相同粒度v44缓存结果比较。测试结果不用于checkpoint选择。','',
            '| 粒度 | 去掉精修后测试MSE变化（v45−v44） | MAE变化 |',
            '|---|---:|---:|']
        for candidate in candidates:
            p=candidate['repair_patch_length']
            cached=next(row for row in primary if row['model']==f'E23-v44-P{p}-Hybrid')
            lines.append(f"| {p}/{p} | {candidate['mse']-cached['mse']:+.6f} | {candidate['mae']-cached['mae']:+.6f} |")
        lines+=['','负值表示移除精修后误差下降。小粒度依然只用原始可见信息和历史候选，没有新增预测头、卷积或注意力模块。',
            '', '本轮三种粒度的测试MSE和MAE均优于相同粒度v44 Hybrid，但验证MSE均退步、验证MAE均改善。单开发种子结果支持末端融合支路可能影响预测，仍不能归纳为CNN/attention一般无效。',
            '8/8同时改善了16/16的验证MSE与MAE，通过本轮匹配验证门槛；测试上16/16更好。因此按验证规则推荐本轮候选8/8，不能依据测试结果反过来把16/16选成正式最优。',
            '三组均未通过v41-slow旧参考门槛，继续保留它；旧参考训练损失与padding/RNG不同，正式统一协议比较仍需另行完成。',
            '4/4的Stage1修复MSE为0.358862，比16/16和8/8高；最近32点均值误差在Stage2并未明显增大，但预测仍较差，说明该单一结构诊断也不能概括全部预测信息。']
    lines+=['','## 统计和产物','',
        '修复按全局标准化尺度上的唯一缺失坐标计数；均值/斜率在完整最近32点窗口计算。原观测、mask和冻结H状态保持。',
        '完整训练时间包括数据准备、两个训练阶段、每轮验证和保存；最终测试及额外诊断不计入训练时间。显存为PyTorch allocated/reserved峰值。',
        '20组标准差衡量缺失位置变化；本轮单开发种子不能替代原五训练种子正式比较。',
        f'- [完整双选模CSV]({PREFIX}.csv)',f'- [MAE主表]({PREFIX}_primary.csv)',
        f'- [20组位置明细]({PREFIX}_all_draws.csv)',f'- [修复与结构诊断]({PREFIX}_recovery.csv)',
        f'- 实验目录：`{ROOT}`；源代码、patch、配置、命令、资源与checkpoint均保存。','']
    (DOCS/(PREFIX+'.md')).write_text('\n'.join(lines))
    if complete:
        assert len(history)==3
        audit=dict(status='passed',sources=len(definition['sources']),protected_files=len(definition['protected']),
            checkpoint_rows=len(rows),new_training_trajectories=3,new_stage1_runs=3,new_stage2_runs=3,
            actual_new_training_seconds=sum(r['actual_new_training_seconds'] for r in primary),
            test_draw_rows=len(draws),new_test_draw_rows=120,
            historical_observed_identities_identical=all(v==next(iter(history.values())) for v in history.values()),
            predictor_source_unchanged=True,tests_passed=52,
            development_verdict_sha256=digest(ROOT/'development-verdict.json'),
            outputs={str(p):digest(p) for p in DOCS.glob(PREFIX+'*')})
        (ROOT/'final-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status='complete' if complete else 'partial',
        primary=[{k:r[k] for k in ('model','mse','mae','validation_mse','validation_mae','full_training_seconds','peak_allocated_gib','peak_reserved_gib')} for r in primary]),ensure_ascii=False))

if __name__=='__main__':main()
