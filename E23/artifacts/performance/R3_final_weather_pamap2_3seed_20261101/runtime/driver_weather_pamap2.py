"""User scope change: adopt live workers; schedule only Weather and PAMAP2."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from driver import write,done,launch,ROOT
from report import report
ALLOWED={'Weather','PAMAP2'}

def live(pid):
 try:
  stat=Path(f'/proc/{pid}/stat').read_text().split()
  return stat[2]!='Z'
 except FileNotFoundError:return False

def refresh_report():
 report()
 rows=json.loads((ROOT/'reports/per_seed.json').read_text());current=[r for r in rows if r['dataset'] in ALLOWED];archived=[r for r in rows if r['dataset'] not in ALLOWED]
 summaries=json.loads((ROOT/'reports/three_seed_summary.json').read_text());summary_count=sum(m['dataset'] in ALLOWED for m in summaries)
 p=ROOT/'reports/report.md';text=p.read_text();text=text.replace('R3 三数据集正式训练进度与结果','R3 本机 Weather / PAMAP2 正式训练进度与结果')
 lines=text.splitlines()
 lines=[f'本机当前范围：已完成正式模型 {len(current)}/24；凑齐三个seed的场景 {summary_count}/8。Exchange-Rate已按用户要求移出本机队列，历史完成模型{len(archived)}个保留，不计入当前进度。' if line.startswith('已完成正式模型 ') else line for line in lines]
 p.write_text('\n'.join(lines)+'\n')

def main():
 lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 handoff=json.loads((ROOT/'scope_handoff.json').read_text());running={};failed=[]
 for gpu,j in handoff['active'].items():
  if j['dataset'] in ALLOWED and live(j['pid']):running[int(gpu)]={'pid':j['pid'],'job':{k:j[k] for k in ['dataset','scene','seed']},'process':None,'log':None,'adopted':True}
 jobs=[{'dataset':d,'scene':s,'seed':seed} for s in ['sr015_bl8','sr010_bl8','sr020_bl8','sr010_bl16'] for seed in [2023,2024,2025] for d in ['Weather','PAMAP2']]
 queue=[j for j in jobs if not done(j) and not any(x['job']==j for x in running.values())]
 refresh_report()
 while queue or running:
  for gpu,x in list(running.items()):
   p=x['process'];finished=p.poll() is not None if p is not None else not live(x['pid'])
   if not finished:continue
   if x['log'] is not None:x['log'].close()
   if not done(x['job']):failed.append({**x['job'],'pid':x['pid'],'gpu':gpu,'exit':None if p is None else p.returncode})
   del running[gpu];refresh_report()
  queue=[j for j in queue if not done(j)]
  for gpu in [0,1]:
   if gpu in running or not queue:continue
   j=queue.pop(0);p,log,_=launch(j,gpu);running[gpu]={'pid':p.pid,'job':j,'process':p,'log':log,'adopted':False}
  completed=sum(done(j) for j in jobs)
  archived=sum(1 for p in (ROOT/'jobs').glob('Exchange-Rate_*_run.json'))
  write(Path('status.json'),{'status':'running' if running else ('complete_with_failures' if failed else 'complete'),'updated_unix':time.time(),'scope':'Weather/PAMAP2 only; Exchange-Rate excluded by user','completed_models':completed,'planned_models':24,'Exchange_historical_completed':archived,'active':{str(g):{'pid':x['pid'],**x['job'],'adopted_without_restart':x['adopted']} for g,x in running.items()},'pending_ready':len(queue),'pending_blocked':0,'failures':failed})
  if queue or running:time.sleep(5)
 refresh_report()
if __name__=='__main__':
 try:main()
 except BaseException as e:write(Path('driver_scope_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
