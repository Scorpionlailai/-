"""Regenerate draft tables using finite, hash-verified evidence snapshots."""
from pathlib import Path
import csv,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1];SNAP=ROOT/'data/snapshots';OUT=ROOT/'tables/generated'
DATASETS=['ETTh1','Weather','Exchange-Rate','BeijingAirQuality','PhysioNet2012','PAMAP2']
SCENES=['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16']
VARIANTS=['E0','E1','E2','E3','E4','E5','E6','E8']

def read(name):return list(csv.DictReader((SNAP/name).open(encoding='utf-8-sig')))
def verify_snapshots():
    for name,item in json.loads((ROOT/'data/source_manifest.json').read_text()).items():
        if hashlib.sha256((SNAP/name).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Changed source snapshot: '+name)

def validate_formal(rows):
    pairs=[(r['dataset'],r['scene']) for r in rows]
    if len(pairs)!=24 or len(set(pairs))!=24 or set(pairs)!={(d,s) for d in DATASETS for s in SCENES}:raise ValueError('Formal cohort incomplete/duplicate')
    for r in rows:
        if r['training_seeds']!=3 or r['test_masks']!=20:raise ValueError('Formal seed/mask scope differs')
        if not all(math.isfinite(r[x]) and r[x]>=0 for x in ['mse','mae','mse_sd','mae_sd']):raise ValueError('Invalid formal metric')

def formal_results():
    rows=[]
    for d in read('formal_four_datasets.csv'):
        rows.append(dict(dataset=d['dataset'],scene=d['scene'],mse=float(d['mse_mean']),mae=float(d['mae_mean']),mse_sd=float(d['mse_std_ddof1']),mae_sd=float(d['mae_std_ddof1']),training_seeds=int(d['n_training_seeds']),test_masks=int(d['test_masks_per_seed']),cohort='formal'))
    for d in read('formal_weather_pamap2.csv'):
        rows.append(dict(dataset=d['dataset'],scene=d['scene'],mse=float(d['test_mse_mean']),mae=float(d['test_mae_mean']),mse_sd=float(d['test_mse_seed_sd']),mae_sd=float(d['test_mae_seed_sd']),training_seeds=int(d['training_seeds']),test_masks=int(d['test_masks_per_model']),cohort='formal'))
    validate_formal(rows)
    return sorted(rows,key=lambda r:(DATASETS.index(r['dataset']),SCENES.index(r['scene'])))

def core_results():
    rows=[]
    for d in read('analysis_core.csv'):
        r={k:(v if k in ['dataset','scene','stage'] else float(v)) for k,v in d.items()};rows.append(r)
        if r['n_training_seeds']!=3 or r['test_masks_each']!=20:raise ValueError('Core analysis seed scope differs')
    if len(rows)!=4:raise ValueError('Incomplete core analysis')
    return rows

def component_results():
    rows=read('analysis_components.csv')
    if len(rows)!=32:raise ValueError('Incomplete components')
    for dataset in ['ETTh1','PAMAP2']:
        selected=[r for r in rows if r['dataset']==dataset]
        if {(r['variant'],r['stage']) for r in selected}!={(v,s) for v in VARIANTS for s in ['coarse','final']}:raise ValueError('Incomplete component cells')
    return rows

def tex_table(name,header,rows,spec):
    OUT.mkdir(parents=True,exist_ok=True)
    lines=[r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']
    lines.extend(' & '.join(map(str,row))+r' \\' for row in rows)
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/(name+'.tex')).write_text('\n'.join(lines)+'\n')

def number(mean,sd=None):
    m=float(mean)
    return f'${m:.6f}$' if sd is None or str(sd)=='' else f'${m:.6f}\\,\\pm\\,{float(sd):.6f}$'

def generate():
    verify_snapshots();formal=formal_results();core=core_results();components=component_results()
    rows=[]
    short={'BeijingAirQuality':'Beijing AQ','PhysioNet2012':'PhysioNet','Exchange-Rate':'Exchange'}
    for r in formal:
        rate,block={'sr010_bl8':('.10',8),'sr015_bl8':('.15',8),'sr020_bl8':('.20',8),'sr010_bl16':('.10',16)}[r['scene']]
        rows.append([short.get(r['dataset'],r['dataset']),rate,block,number(r['mse'],r['mse_sd']),number(r['mae'],r['mae_sd'])])
    tex_table('formal_results',r'\Lang{数据集}{Dataset} & $p_s$ & $b$ & MSE $\downarrow$ & MAE $\downarrow$',rows,'llcrr')
    rows=[]
    descriptions={'E0':r'\Model{}','E1':r'\Lang{去时间证据}{w/o temporal evidence}','E2':r'\Lang{去跨变量证据}{w/o variable evidence}','E3':r'\Lang{去历史证据}{w/o historical evidence}','E4':r'\Lang{来源内随机}{Source-random}','E5':r'\Lang{统一检索}{Unified retrieval}','E6':r'\Lang{无候选容量匹配}{Context-only (matched)}','E8':r'\Lang{Patch共享读取}{Patch-shared read}'}
    for v in VARIANTS:
        a=next(r for r in components if r['dataset']=='ETTh1' and r['stage']=='final' and r['variant']==v)
        b=next(r for r in components if r['dataset']=='PAMAP2' and r['stage']=='final' and r['variant']==v)
        rows.append([descriptions[v],number(a['mse_mean'],a['mse_std_ddof1']),number(a['mae_mean'],a['mae_std_ddof1']),number(b['mse_mean']),number(b['mae_mean'])])
    tex_table('components',r'\Lang{变体}{Variant} & ETTh1 MSE & ETTh1 MAE & PAMAP2 MSE & PAMAP2 MAE',rows,'lrrrr')
    rows=[]
    for dataset in ['ETTh1','PAMAP2']:
        c=next(r for r in core if r['dataset']==dataset and r['stage']=='coarse');f=next(r for r in core if r['dataset']==dataset and r['stage']=='final')
        for v,arm in [(r'\Model{}','e0'),(r'\Lang{上下文对照}{Context-only}','e6')]:rows.append([dataset,v,number(c[arm+'_mse_mean'],c[arm+'_mse_std_ddof1']),number(f[arm+'_mse_mean'],f[arm+'_mse_std_ddof1']),number(f[arm+'_mae_mean'],f[arm+'_mae_std_ddof1'])])
    tex_table('paired_core',r'\Lang{数据集}{Dataset} & \Lang{模型}{Model} & \Lang{粗MSE}{\shortstack{Coarse\\MSE}} & \Lang{精MSE}{\shortstack{Refined\\MSE}} & \Lang{精MAE}{\shortstack{Refined\\MAE}}',rows,'llrrr')
    rows=[]
    for d in read('beijing_backbone_refinement.csv'):
        if d['region']=='full96':rows.append([{'R3':r'\Model{}','E6':r'\Lang{上下文对照}{Context-only}'}.get(d['model'],d['model']),number(d['coarse_mse']),number(d['calibrated_mse']),number(d['coarse_mae']),number(d['calibrated_mae'])])
    tex_table('backbone_refinement',r'\Lang{骨干}{Backbone} & \Lang{粗MSE}{\shortstack{Coarse\\MSE}} & \Lang{精MSE}{\shortstack{Refined\\MSE}} & \Lang{粗MAE}{\shortstack{Coarse\\MAE}} & \Lang{精MAE}{\shortstack{Refined\\MAE}}',rows,'lrrrr')
    rows=[]
    for d in read('analysis_efficiency.csv'):
        if d['variant'] in ['E0','E6']:
            rows.append([d['dataset'],r'\Model{}' if d['variant']=='E0' else r'\Lang{上下文对照}{Context-only}',f"{int(d['effective_parameters']):,}",f"{float(d['p50_ms']):.2f}",f"{float(d['peak_allocated_GiB']):.3f}"])
    tex_table('efficiency',r'\Lang{数据集}{Dataset} & \Lang{模型}{Model} & \Lang{参数}{Params} & p50 (ms) & \Lang{显存}{VRAM} (GiB)',rows,'llrrr')
    rows=[]
    for d in read('analysis_sensitivity.csv'):rows.append([d['heads'],number(d['validation_mse']),number(d['validation_mae']),number(d['final_test_mse']),number(d['final_test_mae'])])
    tex_table('sensitivity',r'\Lang{头数}{Heads} & Val MSE & Val MAE & Test MSE & Test MAE',rows,'crrrr')
    macros=[]
    for r in core:
        if r['stage']=='final':
            tag='ETT' if r['dataset']=='ETTh1' else 'PAM'
            for metric in ['mse','mae']:macros.append(r'\newcommand{\Gain'+tag+metric.upper()+'}{'+f"{r[metric+'_reduction_pct']:.2f}"+r'\%}')
    (OUT/'numbers.tex').write_text('\n'.join(macros)+'\n')
    (OUT/'metadata.json').write_text(json.dumps(dict(formal_rows=24,formal_pamap_mask_bank='20261014',analysis_pamap_mask_bank='20261018',core_three_seeds=True,component_pamap_seeds=1,full_baseline_matrix_complete=False),indent=2)+'\n')
    print('Generated 6 tables + scalar macros from hash-verified CSV snapshots')
if __name__=='__main__':generate()
