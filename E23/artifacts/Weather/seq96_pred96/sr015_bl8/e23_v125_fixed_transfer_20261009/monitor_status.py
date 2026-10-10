from pathlib import Path
import json
from datetime import datetime,timezone
r=Path(__file__).resolve().parent
s=json.loads((r/'status.json').read_text())
out={'time_utc':datetime.now(timezone.utc).isoformat(),'status':s['status'],'phase':s.get('phase'),'progress':[]}
for width in [64,256]:
 b=r/f'calibration{width}';p=b/'train_refinement.log'
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-1500));lines=f.read().decode().splitlines()
 launch=json.loads((b/'launch.json').read_text())
 x={'width':width,'alive':(Path('/proc')/str(launch['pid'])).exists(),'last_line':lines[-1] if lines else ''}
 hp=b/f'E23-v125-C{width}/sr015_bl8/stage1/runs/e23-stage1-history.json'
 if hp.exists():
  h=json.loads(hp.read_text())['epochs'];best=min(h,key=lambda a:a['metrics']['global_mae'])
  x.update(completed_epochs=len(h),best_val_mae=best['metrics']['global_mae'],paired_val_mse=best['metrics']['global_mse'])
 if x.get('completed_epochs')==100 and not x['alive']:x['last_line']='training finished; paired MAE/MSE saved'
 out['progress'].append(x)
if s.get('phase')!='parallel_calibration_training':
 out['phase_last_line']=s.get('last_line',s.get('error',''))
 q=r/'ensemble/evaluate_repair.log'
 if q.exists():out['completed_test_draws']=sum('\"split\": \"test\"' in line for line in q.read_text().splitlines())
print(json.dumps(out))
