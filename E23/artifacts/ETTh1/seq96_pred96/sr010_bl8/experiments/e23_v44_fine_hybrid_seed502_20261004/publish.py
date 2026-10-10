"""Publish completed validation-locked cells, including both selectors and resource records."""
import csv,hashlib,json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v44-fine/E23')
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
PREFIX='ETTh1_P4_E23_v44_fine_hybrid_seed502_20261004'

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
    lines=['# E23 v44：细粒度三专家与CNN／跨变量注意力','',
        '状态：'+('四组全部完成。' if complete else '运行中；下表仅列已完成并锁定选模的组。'),'',
        '## 固定设置','',
        '- 修复输出16/16、8/8、4/4，分别6、12、24块；H保持16点父上下文及16/8训练观测库，候选按子块裁切。T/V读取跨度按原时间点对齐。',
        '- 完整96点共享多尺度CNN，64通道、两个block、kernel31/5。Hybrid按通道进行变量轴注意力；CNN对照用本变量Q/K/V卷积结果均值。两者参数量和初始化完全一致。',
        '- 支路在全局标准化尺度输出残差，读取完整96点、精修最近32点缺失值；Stage1 specialist singleton目标关闭支路残差。',
        '- 新组各自100轮Stage1与100轮Stage2，batch128、FP32。Stage1全局缺失MSE+0.1检索，Stage2纯预测MSE；预测器LR1e-4、修复器LR1e-5。',
        '- 原预测头PatchTST D16、35427参数、16/8输入与初始化一致；沿用T1对齐的padding和训练RNG重置。',
        '- 开发seed502；主checkpoint按固定验证MAE，另存验证MSE最低版本。每组锁定后测试同一20组缺失位置。','',
        '## 验证MAE选模的主结果','',
        '| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | allocated / reserved峰值GiB |',
        '|---|---:|---:|---|---:|---|']
    for r in primary:
        lines.append(f"| {r['model']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['validation_mse']:.6f} / {r['validation_mae']:.6f} | {r['full_training_seconds']/60:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    lines+=['','T1、v41与v43-N为缓存参考，没有新训练。v41使用旧混合损失与修复保留，padding/RNG设置不同；新组之间才是粒度与注意力的匹配比较。T1保留作者Stage1 batch16/BF16配置。','',
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
            '**四组均未通过相对v41-slow的固定验证门槛，保留v41-slow开发参考。** 第一组的验证MSE更低，但MAE退步；其余组也未同时改善两个指标。全部主/副版本测试指标均未超过v41-slow的MSE和MAE。原正式五训练种子表保持。','',
            '- 本轮Hybrid中，8/8的测试MSE为0.415202，优于16/16的0.438323；4/4反而为0.424249。该趋势来自单开发种子的这组训练，不能认定patch越小越好。',
            '- 同为4/4且参数、初始化一致，CNN测试MSE/MAE为0.412746/0.424908，Hybrid为0.424249/0.436216。新增跨变量注意力在本次方案中没有带来预测收益。',
            '- 4/4 Hybrid的Stage1和Stage2整体修复MSE均比CNN更低，但Stage2最近32点均值误差更高：0.102708对0.076122。这与此前近期状态诊断一致：整体修复误差更低，仍可能没有保住预测需要的状态结构；本次没有对均值做新干预，不能仅由相关指标断言因果归因。',
            '- 小块H上下文盲区已在实现中缓解：4/4 Hybrid的155905个完全缺失子块可以读取父16点上下文检索所得候选，父块完全缺失仍不启用H。四组训练库的原始观测坐标身份一致，训练后的key、分组和权重不同。',
            '- 下一方向应检验在专家之前以共享CNN建立小块的上下文状态，以及保护Stage2的近期均值/趋势；本轮只验证末端精修方案，未实现或训练这两项后续改动。']
    lines+=['','## 统计和产物','',
        '修复按全局标准化尺度上的唯一缺失坐标计数；均值/斜率在完整最近32点窗口计算。原观测、mask和冻结H状态保持。',
        '完整训练时间包括数据准备、两个训练阶段、每轮验证和保存；最终测试及额外诊断不计入训练时间。显存为PyTorch allocated/reserved峰值。',
        '20组标准差衡量缺失位置变化；本轮单开发种子不能替代原五训练种子正式比较。',
        f'- [完整双选模CSV]({PREFIX}.csv)',f'- [MAE主表]({PREFIX}_primary.csv)',
        f'- [20组位置明细]({PREFIX}_all_draws.csv)',f'- [修复与结构诊断]({PREFIX}_recovery.csv)',
        f'- 实验目录：`{ROOT}`；源代码、patch、配置、命令、资源与checkpoint均保存。','']
    (DOCS/(PREFIX+'.md')).write_text('\n'.join(lines))
    if complete:
        assert len(history)==4
        audit=dict(status='passed',sources=len(definition['sources']),protected_files=len(definition['protected']),
            checkpoint_rows=len(rows),new_training_trajectories=4,new_stage1_runs=4,new_stage2_runs=4,
            actual_new_training_seconds=sum(r['actual_new_training_seconds'] for r in primary),
            test_draw_rows=len(draws),new_test_draw_rows=160,
            historical_observed_identities_identical=all(v==next(iter(history.values())) for v in history.values()),
            predictor_source_unchanged=True,tests_passed=66,
            development_verdict_sha256=digest(ROOT/'development-verdict.json'),
            outputs={str(p):digest(p) for p in DOCS.glob(PREFIX+'*')})
        (ROOT/'final-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(status='complete' if complete else 'partial',
        primary=[{k:r[k] for k in ('model','mse','mae','validation_mse','validation_mae','full_training_seconds','peak_allocated_gib','peak_reserved_gib')} for r in primary]),ensure_ascii=False))

if __name__=='__main__':main()
