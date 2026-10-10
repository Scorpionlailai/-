"""Read-only 5-minute forecast monitor and final comparison export."""
from pathlib import Path
import json,time,re,csv,datetime
R=Path(__file__).resolve().parent
job=json.loads((R/'prediction_job.json').read_text());base=Path(job['output']).parent

def check():
 proc=Path('/proc')/str(job['pid']);alive=proc.exists() and (proc/'stat').read_text().split()[2]!='Z'
 log=Path(job['log']).read_text();matches=re.findall(r'stage=forecast epoch=(\d+)/100 val_mse=([\d.]+) val_mae=([\d.]+)',log)
 complete=(R/'prediction_completed.json').exists();status='completed' if complete else ('training_or_evaluating' if alive else 'failed_or_interrupted')
 result=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,epochs_completed=int(matches[-1][0]) if matches else 0,last_validation_mse=float(matches[-1][1]) if matches else None,last_validation_mae=float(matches[-1][2]) if matches else None,log_tail=log.splitlines()[-3:])
 (R/'prediction_monitor_status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 if complete:
  files=[('Weather-v79-fixed4',Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_weather_20261007/seed_2023/E24-v79/evaluation_mae/twenty-test-draws.json')),('T1',Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/T1/evaluation/mae/twenty-test-draws.json')),('E23-v103',base/'evaluation_mae/twenty-test-draws.json')]
  rows=[];hashes=None
  for label,path in files:
   d=json.loads(path.read_text());current=[r['mask_sha256'] for r in d['draws']]
   if hashes is None:hashes=current
   assert current==hashes and d['repair_frozen_verified']
   row=dict(model=label,selection='validation_mae',test_masks=len(current),result_path=str(path),test_role='existing development masks, single training seed')
   for key in ('forecast_mse','forecast_mae','repair96_mse','repair96_mae','repair32_mse','repair32_mae'):
    source=key
    if key.startswith('forecast') and key not in d['summary']:source=key.replace('forecast_','')
    row[key]=d['summary'][source]['mean'];row[key+'_mask_std']=d['summary'][source]['mask_std_ddof1']
   rows.append(row)
  with (R/'forecast_comparison.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return status
if __name__=='__main__':
 while True:
  state=check()
  if state in ('completed','failed_or_interrupted'):break
  time.sleep(300)
