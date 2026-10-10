"""Publish current completed metrics every five minutes, without polling GPU logs."""
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
NEXT = ROOT.parent / 'e23_global_dense_stage1_iterations_20261007'
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
STEM = 'E23_stage1_live_status_20261007'


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def publish():
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    rows = []
    states = []
    active = set()
    for directory in [ROOT, NEXT]:
        if (directory / 'queue-state.json').exists():
            state = read(directory / 'queue-state.json')
            states.append(dict(queue=directory.name, **state))
            if state['status'] == 'running':
                active.add(state['active']['label'])
        for entry in read(directory / 'queue-plan.json')['models']:
            root = Path(entry['root'])
            plan = read(root / 'plan.json')
            verified = (root / 'verification.json').exists()
            for scene in plan['scenarios']:
                cell = root / scene['name'] / entry['label']
                repair = cell / 'stage1/runs/e23-stage1.json'
                evaluation = cell / 'evaluation_mae/twenty-test-draws.json'
                row = dict(updated_shanghai=now, model=entry['label'], scene=scene['name'],
                    actual_missing_fraction=scene['actual_test_missing_fraction'],
                    status='verified' if verified else 'tests_complete' if evaluation.exists() else
                        'stage2' if entry['label'] in active and (cell / 'stage2.log').exists() else
                        'stage1' if entry['label'] in active and (cell / 'stage1.log').exists() else 'queued',
                    repair_val_mse='', repair_val_mae='', forecast_test_mse='', forecast_test_mae='',
                    training_seconds='', peak_allocated_gib='', peak_reserved_gib='',
                    stage2_loss='pure_mse', repair_update_policy='frozen_v1',
                    stage1_manifest=str(repair) if repair.exists() else '',
                    evaluation_manifest=str(evaluation) if evaluation.exists() else '')
                if repair.exists():
                    r = read(repair)['validation_repair_metrics']
                    row.update(repair_val_mse=r['global_mse'], repair_val_mae=r['global_mae'])
                if evaluation.exists():
                    value = read(evaluation)
                    resources = value['training_resources']
                    row.update(forecast_test_mse=value['test_mean']['mse'],
                        forecast_test_mae=value['test_mean']['mae'],
                        training_seconds=resources['full_workflow_seconds'],
                        peak_allocated_gib=resources['peak_allocated_gib'],
                        peak_reserved_gib=resources['peak_reserved_gib'])
                rows.append(row)
    table = ['# E23 Stage1 持续迭代进度', '', f'更新时间（上海）：{now}', '',
        '所有改动限定Stage1。Stage2纯MSE、修复器参数/缓冲/记忆/检索冻结、原预测头及原训练配置。每5分钟自动更新此表；训练日志由各队列每300秒检查。', '',
        '| 版本 | 场景 | 状态 | 修复验证MSE / MAE | 预测测试MSE / MAE | 两阶段训练秒 | allocated / reserved GiB |',
        '|---|---|---|---:|---:|---:|---:|']
    def number(value):
        return '—' if value == '' else f'{value:.6f}'
    for r in rows:
        table.append(f'| {r["model"]} | {r["scene"]} | {r["status"]} | '
            f'{number(r["repair_val_mse"])} / {number(r["repair_val_mae"])} | '
            f'{number(r["forecast_test_mse"])} / {number(r["forecast_test_mae"])} | '
            f'{number(r["training_seconds"])} | '
            f'{number(r["peak_allocated_gib"])} / {number(r["peak_reserved_gib"])} |')
    table += ['', '修复为固定验证集全局标准化缺失坐标；预测为验证MAE选定权重、旧20组开发测试的平均值。未完成结果留空，预测MSE次选另见各版本正式CSV。训练时间包含两阶段完整命令，不含测试。', '',
              f'[CSV](./{STEM}.csv)', '', '## 队列状态', '']
    for state in states:
        table.append(f'- {state["queue"]}: {state["status"]}')
    if not (NEXT / 'queue-state.json').exists():
        table.append('- 全局逐点修复候选已准备，等待前一队列结果。')
    atomic_text(DOCS / f'{STEM}.md', '\n'.join(table) + '\n')
    temporary = DOCS / f'{STEM}.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(DOCS / f'{STEM}.csv')
    print(json.dumps(dict(updated_shanghai=now, active=sorted(active), rows=len(rows))), flush=True)
    return states


if __name__ == '__main__':
    while True:
        try:
            states = publish()
        except (json.JSONDecodeError, FileNotFoundError):
            # Queue state changes are short writes; do not lose the publisher
            # if its scheduled read coincides with a controller update.
            time.sleep(1)
            continue
        first = read(ROOT / 'queue-state.json')['status']
        if first != 'running' and ((NEXT / 'queue-state.json').exists() and
                read(NEXT / 'queue-state.json')['status'] != 'running' or
                (NEXT / 'deferred-status.json').exists()):
            break
        time.sleep(300)
