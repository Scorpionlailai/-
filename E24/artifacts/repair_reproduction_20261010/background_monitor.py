"""Keep half-hour progress records available even if the UI disconnects."""
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
lock = (root / 'monitor.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
start = datetime.datetime.fromisoformat('2026-10-10T10:02:39+00:00').timestamp()
now = time.time()
next_due = start + (int((now - start) // 1800) + 1) * 1800
while not (root / 'MONITOR_STOP').exists():
    try:
        result = subprocess.run([sys.executable, str(root / 'session_snapshot.py')], capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        snapshot = json.loads(result.stdout)
        if time.time() >= next_due:
            snapshot['scheduled_due_utc'] = datetime.datetime.fromtimestamp(next_due, datetime.timezone.utc).isoformat()
            with (root / 'automatic_half_hour_reports.jsonl').open('a') as stream:
                stream.write(json.dumps(snapshot) + '\n')
            local_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
            lines = [f'# 实验监控记录（北京时间 {local_time}）', '',
                     f"修复评估：{snapshot['completed_repairs']}/32；预测评估：{snapshot['completed_forecasts']}/32。",
                     f"队列状态：{snapshot['queue_state']}；失败记录：{snapshot['failed_runs']}。", '']
            for job in snapshot['active']:
                lines.append(f"- GPU {job['gpu']}：{job['model']} / {job['dataset']} / {job['scene']}；阶段 {job['phase']}；轮数 {job['epoch']}；已测试掩码 {job['completed_masks']}。")
            lines += ['', '此文件为后台记录，不能自动向会话发送消息。']
            temporary = root / 'latest_monitor_report.md.tmp'
            temporary.write_text('\n'.join(lines) + '\n')
            temporary.replace(root / 'latest_monitor_report.md')
            while next_due <= time.time():
                next_due += 1800
    except Exception as error:
        with (root / 'monitor_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), error=str(error))) + '\n')
    time.sleep(30)
