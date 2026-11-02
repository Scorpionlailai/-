from pathlib import Path
import json,csv,statistics,argparse
R=Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text());SC=P['scene'];DS=P['dataset']
def group(v):return 'sources' if v in ['E1','E2','E3','E4','E5'] else 'context' if v=='E6' else 'point'
def writecsv(path,rows):
 if not rows:return
 path.parent.mkdir(exist_ok=True,parents=True);keys=list(dict.fromkeys(k for row in rows for k in row));temp=path.with_suffix('.tmp')
 with temp.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 temp.replace(path)
def build(suffix='current'):
 per=[];sources={};rsc={}
 for v in ['E0']+P['variants']:
  for s in P['seeds']:
   folder=(Path(P['reuse_E0'])/'outputs/evaluation'/DS/SC/f'seed_{s}'/'H_GLOBAL') if v=='E0' else R/group(v)/'outputs/evaluation'/DS/SC/f'seed_{s}'/v
   file=folder/'results.json'
   if not file.exists():continue
   d=json.loads(file.read_text());assert d['test_masks']==20 and not d.get('smoke_only',False)
   sources[v,s]=d
   resource=folder/'resources.json'
   if resource.exists():rsc[v,s]=json.loads(resource.read_text())
   for stage,field in [('coarse','precal_test'),('final','test')]:
    for region in ['full96','recent32']:
     m=d[field][region];per.append(dict(variant=v,seed=s,stage=stage,region=region,mse=m['mse_mean'],mae=m['mae_mean'],test_masks=20,result_file=str(file)))
 baseline={(x['seed'],x['stage'],x['region']):x for x in per if x['variant']=='E0'}
 for row in per:
  b=baseline[row['seed'],row['stage'],row['region']]
  row.update(delta_mse_vs_E0=row['mse']-b['mse'],delta_mae_vs_E0=row['mae']-b['mae'],mse_higher_than_E0_pct=100*(row['mse']/b['mse']-1))
 stats=[]
 for v in ['E0']+P['variants']:
  for stage in ['coarse','final']:
   for region in ['full96','recent32']:
    rows=[x for x in per if x['variant']==v and x['stage']==stage and x['region']==region]
    if not rows:continue
    out=dict(variant=v,stage=stage,region=region,n_training_seeds=len(rows))
    for key in ['mse','mae','delta_mse_vs_E0','delta_mae_vs_E0']:
     values=[x[key] for x in rows];out[key+'_mean']=statistics.mean(values);out[key+'_std_ddof1']=statistics.stdev(values) if len(values)>1 else None
    out['E0_better_mse_seeds']=sum(x['delta_mse_vs_E0']>0 for x in rows)
    out['T1_archived_mse']=P['T1_reference_mse'] if region=='full96' else None
    out['T1_reference_scope']='single original T1 training seed2023; descriptive only'
    stats.append(out)
 writecsv(R/'summary'/f'per_seed_{suffix}.csv',per);writecsv(R/'summary'/f'single_seed_{suffix}.csv',stats)
 res=[]
 for (v,s),d in sources.items():
  c=d['info'];data=dict(variant=v,seed=s,coarse_seconds=c['coarse']['training_seconds'],calibration_seconds=c['calibration_seconds'])
  data.update({k:rsc.get((v,s),{}).get(k) for k in ['effective_parameters','registered_parameters','p50_batch_ms','gpu','physical_gpu','peak_allocated_GiB','peak_reserved_GiB','exclusive_gpu_verified']});res.append(data)
 writecsv(R/'summary'/f'resources_{suffix}.csv',res)
 return len(sources)
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--suffix',default='current');a=p.parse_args();print('Models evaluated',build(a.suffix))
