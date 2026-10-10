"""Acquire original planned datasets; no preprocessing, masks or training."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv, datetime, gzip, hashlib, json, os, shutil, subprocess, tarfile, zipfile
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; DL=DATA/'downloads'/'planned_20261010'; RAW=DATA/'raw'; META=DATA/'metadata'
for d in [DL,RAW,META]:d.mkdir(parents=True,exist_ok=True)
TASKS=[
 ('Exchange-Rate','https://github.com/laiguokun/multivariate-time-series-data', [('exchange_rate.txt.gz','https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz')]),
 ('Beijing_AirQuality','https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data', [('beijing_air_quality.zip','https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip')]),
 ('METR-LA','https://github.com/TorchSpatiotemporal/tsl/blob/main/tsl/datasets/metr_la.py', [('metr_la.zip','https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/metr_la.zip')]),
 ('PAMAP2','https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring', [('pamap2.zip','https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip')]),
 ('PhysioNet2012','https://physionet.org/content/challenge-2012/1.0.0/', [(f'set-{s}.tar.gz',f'https://physionet.org/files/challenge-2012/1.0.0/set-{s}.tar.gz') for s in 'abc']+[(f'Outcomes-{s}.txt',f'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-{s}.txt') for s in 'abc']),
]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def timestamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def download(url,p):
 if p.exists():return
 part=p.with_name(p.name+'.part')
 cmd=['curl','--fail','--location','--continue-at','-','--retry','3','--retry-all-errors','--connect-timeout','30','--speed-limit','1','--speed-time','180','--silent','--show-error','--output',str(part),url]
 subprocess.run(cmd,check=True)
 assert part.stat().st_size>0
 part.replace(p)
def destination(base,name):
 p=base/name
 if not p.resolve().is_relative_to(base.resolve()):raise ValueError('Unsafe archive path')
 return p
def unzip(p,d):
 with zipfile.ZipFile(p) as z:
  bad=z.testzip()
  if bad:raise ValueError('ZIP CRC failure: '+bad)
  for info in z.infolist():
   dest=destination(d,info.filename)
   if info.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
   if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Unexpected symlink')
   dest.parent.mkdir(parents=True,exist_ok=True)
   with z.open(info) as src,dest.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
def untar(p,d):
 with tarfile.open(p,'r:gz') as t:
  for info in t:
   dest=destination(d,info.name)
   if info.isdir():dest.mkdir(parents=True,exist_ok=True)
   elif info.isfile():
    dest.parent.mkdir(parents=True,exist_ok=True)
    with t.extractfile(info) as src,dest.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
   else:raise ValueError('Unexpected non-file tar entry')
def acquire(task):
 name,page,assets=task;d=RAW/name;d.mkdir(exist_ok=True);folder=DL/name;folder.mkdir(exist_ok=True)
 status=META/(name+'_download_status.json')
 write(status,{'state':'downloading','dataset':name,'started_utc':timestamp(),'source_page':page})
 records=[]
 try:
  for filename,url in assets:
   p=folder/filename;print('Downloading '+name+'/'+filename,flush=True);download(url,p)
   records.append({'path':str(p.relative_to(DATA)),'url':url,'bytes':p.stat().st_size,'sha256':sha(p)})
   if filename.endswith('.zip'):unzip(p,d)
   elif filename.endswith('.tar.gz'):untar(p,d)
   elif filename.endswith('.txt.gz'):
    with gzip.open(p,'rb') as src,(d/'exchange_rate.txt').open('wb') as out:shutil.copyfileobj(src,out)
   else:shutil.copy2(p,d/filename)
  # UCI bundles can contain the original dataset archive as a nested ZIP.
  for p in list(d.rglob('*.zip')):unzip(p,p.parent)
  checks={}
  if name=='Exchange-Rate':
   with (d/'exchange_rate.txt').open() as f:
    rows=[x for x in csv.reader(f) if x]
   assert len(rows)>1000 and all(len(x)==8 for x in rows)
   for row in rows:
    for x in row:float(x)
   checks={'rows':len(rows),'variables':8,'calendar_dates':'source has no date column; no dates invented'}
  elif name=='Beijing_AirQuality':
   sites=list(d.rglob('PRSA_Data_*.csv'));assert len(sites)==12
   rows=0
   for p in sites:
    with p.open() as f:
     reader=csv.reader(f);header=next(reader);assert {'PM2.5','PM10','SO2','NO2','CO','O3','station'}.issubset(header)
     count=sum(1 for _ in reader);assert count==35064;rows+=count
   checks={'station_files':12,'rows_per_station':35064,'total_station_rows':rows,'native_missing_preserved':True}
  elif name=='PAMAP2':
   files=list(d.rglob('*.dat'));protocol=[p for p in files if p.parent.name=='Protocol'];assert len(protocol)==9
   for p in files:
    with p.open() as f:first=next(x for x in f if x.strip())
    assert len(first.split())==54
   checks={'protocol_subjects':len(protocol),'recording_files':len(files),'raw_columns':54,'labels_and_invalid_orientation_retained_in_raw_only':True,'native_missing_preserved':True}
  elif name=='PhysioNet2012':
   groups={}
   for group in 'abc':
    files=list((d/f'set-{group}').glob('*.txt'));assert len(files)==4000,(group,len(files))
    with files[0].open() as f:assert next(csv.reader(f))==['Time','Parameter','Value']
    groups[group]=len(files)
   checks={'patients_per_group':groups,'patients_total':sum(groups.values()),'no_patient_concatenation':True,'outcomes_not_used_as_repair_inputs':True}
  elif name=='METR-LA':
   h5=list(d.rglob('*.h5'));assert len(h5)==1
   with h5[0].open('rb') as f:assert f.read(8)==b'\x89HDF\r\n\x1a\n'
   assert list(d.rglob('*distances*.csv')) and list(d.rglob('*sensor*ids*.txt'))
   checks={'archive_crc_verified':True,'hdf5_magic_verified':True,'provider_version':'TorchSpatiotemporal/ProcessedDatasets v1.0.0','zero_and_nan_values_preserved':True,'table_shape_validation':'pending HDF5 reader availability'}
  rawfiles=[]
  for p in sorted(d.rglob('*')):
   if p.is_file():rawfiles.append({'path':str(p.relative_to(DATA)),'bytes':p.stat().st_size,'sha256':sha(p)})
  result={'state':'complete','dataset':name,'source_page':page,'completed_utc':timestamp(),'archives':records,'raw_files':rawfiles,'checks':checks,'preprocessing_applied':False,'training_started':False}
  write(status,result);print('VERIFIED '+name,flush=True);return result
 except Exception as e:
  result={'state':'failed','dataset':name,'source_page':page,'error':str(e),'downloaded_assets':records}
  write(status,result);print('FAILED '+name+': '+str(e),flush=True);return result
results=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for f in as_completed([pool.submit(acquire,t) for t in TASKS]):
  results.append(f.result());write(META/'planned_datasets_20261010.json',{'updated_utc':timestamp(),'new_datasets':results,'state':'running'})
existing=[]
for name,filename in [('ETTh1','ETTh1.csv'),('Weather','weather.csv')]:
 p=RAW/filename
 with p.open() as f:
  reader=csv.reader(f);header=next(reader);rows=sum(1 for _ in reader)
 assert header[0]=='date'
 existing.append({'dataset':name,'path':str(p.relative_to(DATA)),'bytes':p.stat().st_size,'sha256':sha(p),'rows':rows,'variables':len(header)-1,'state':'existing_verified'})
write(META/'planned_datasets_20261010.json',{'updated_utc':timestamp(),'state':'complete' if all(x['state']=='complete' for x in results) else 'incomplete','existing_datasets':existing,'new_datasets':results})
print('DOWNLOAD_SUMMARY '+json.dumps([(x['dataset'],x['state']) for x in results]),flush=True)
if any(x['state']!='complete' for x in results):raise SystemExit(1)
