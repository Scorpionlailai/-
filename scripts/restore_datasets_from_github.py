"""Fetch official dataset archive mirrors from GitHub Releases, verify and extract."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def valid(path,record):
 return path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
def fetch(part):
 p=DATA/'downloads/github_parts_20261010'/part['name'];p.parent.mkdir(parents=True,exist_ok=True)
 if valid(p,part):return p
 tmp=p.with_name(p.name+'.part')
 subprocess.run(['curl','--fail','--location','--retry','3','--retry-all-errors','--connect-timeout','30','--silent','--show-error','--output',str(tmp),part['url']],check=True)
 if not valid(tmp,part):raise RuntimeError('Release asset checksum mismatch: '+part['name'])
 tmp.replace(p);return p

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--verify-only',action='store_true',help='Verify existing complete archives; do not download or extract')
 args=parser.parse_args()
 m=json.loads((DATA/'metadata/github_datasets_20261010.json').read_text())
 missing=[]
 for a in m['archives']:
  dst=DATA/a['path']
  if valid(dst,a):print('VERIFIED '+a['dataset']+'/'+dst.name,flush=True);continue
  if args.verify_only:missing.append(a['path']);continue
  print('Downloading '+a['dataset']+'/'+dst.name,flush=True)
  with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(fetch,a['parts']))
  dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.github-part')
  with tmp.open('wb') as out:
   for p in parts:
    with p.open('rb') as src:
     for block in iter(lambda:src.read(1024*1024),b''):out.write(block)
  if not valid(tmp,a):raise RuntimeError('Reassembled archive checksum mismatch: '+str(dst))
  tmp.replace(dst);print('VERIFIED '+a['dataset']+'/'+dst.name,flush=True)
 if missing:raise RuntimeError('Missing or invalid archives: '+', '.join(missing))
 if not args.verify_only:
  subprocess.run([sys.executable,str(ROOT/'scripts/download_planned_datasets_20261010.py')],check=True)
  print('All datasets restored. No training started.',flush=True)
if __name__=='__main__':main()
