"""Publish verified archive mirrors from official sources on a GitHub runner."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'data/metadata/github_datasets_20261010.json'
manifest=json.loads(M.read_text());repo=manifest['repository'];tag=manifest['release_tag']
def gh(*args):return subprocess.check_output(['gh',*args],text=True)
release_id=json.loads(gh('release','view',tag,'--repo',repo,'--json','databaseId'))['databaseId']
release=json.loads(gh('api',f'repos/{repo}/releases/{release_id}'))
assets={a['name']:a for a in release['assets']}
def present(part):
 a=assets.get(part['name'],{})
 return a.get('state')=='uploaded' and a.get('size')==part['bytes'] and a.get('digest')=='sha256:'+part['sha256']
folder=Path('/tmp/paper-dataset-release');folder.mkdir(exist_ok=True)
def upload(path):
 subprocess.run(['gh','release','upload',tag,str(path),'--repo',repo,'--clobber'],check=True)
def process(a):
 missing=[p for p in a['parts'] if not present(p)]
 if not missing:return 'Reused '+a['dataset']+'/'+Path(a['path']).name
 dst=folder/(a['dataset']+'__'+Path(a['path']).name)
 subprocess.run(['curl','--fail','--location','--retry','3','--retry-all-errors','--connect-timeout','30','--silent','--show-error','--output',str(dst),a['original_url']],check=True)
 h=hashlib.sha256()
 with dst.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 assert dst.stat().st_size==a['bytes'] and h.hexdigest()==a['sha256'],'Official archive differs from verified local snapshot'
 with dst.open('rb') as f:
  for p in a['parts']:
   b=f.read(p['bytes']);assert hashlib.sha256(b).hexdigest()==p['sha256']
   if present(p):continue
   part=folder/p['name']
   if part!=dst:part.write_bytes(b)
   upload(part)
   if part!=dst:part.unlink()
 dst.unlink()
 return 'Uploaded '+a['dataset']+'/'+Path(a['path']).name
with ThreadPoolExecutor(max_workers=4) as pool:
 for f in as_completed([pool.submit(process,a) for a in manifest['archives']]):print(f.result(),flush=True)
upload(M)
release=json.loads(gh('api',f'repos/{repo}/releases/{release_id}'))
assets={a['name']:a for a in release['assets']}
for a in manifest['archives']:
 for p in a['parts']:assert present(p),(p['name'],'remote SHA256/size mismatch')
assert assets[M.name]['size']==M.stat().st_size
subprocess.run(['gh','release','edit',tag,'--repo',repo,'--draft=false','--latest=false'],check=True)
print('VERIFIED_PUBLISHED '+manifest['release_url'],flush=True)
