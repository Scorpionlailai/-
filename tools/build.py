"""Rebuild previews and an Overleaf-ready source ZIP; reject broken references."""
from pathlib import Path
import subprocess,shutil,os,re,json,hashlib,zipfile,sys
ROOT=Path(__file__).resolve().parents[1]

def run(cmd,**kwargs):return subprocess.run(cmd,cwd=ROOT,check=True,**kwargs)
def build():
    run([sys.executable,str(ROOT/'tools/generate_tables.py')])
    engine=os.environ.get('TECTONIC') or shutil.which('tectonic') or str(Path.home()/'.local/bin/tectonic')
    reports={}
    for language,entry,stem in [('en','main.tex','main'),('zh','main_zh.tex','main_zh')]:
        out=ROOT/'build'/language;out.mkdir(parents=True,exist_ok=True)
        log=ROOT/'build'/f'{language}_build.log'
        with log.open('w') as f:run([engine,'-o',str(out),'--keep-logs',entry],stdout=f,stderr=subprocess.STDOUT)
        text=(out/(stem+'.log')).read_text(errors='replace')
        errors=[p for p in [r'Undefined control sequence',r'Citation [`\'\{].+undefined',r'Reference [`\'\{].+undefined',r'There were undefined',r'Overfull \\hbox',r'Font shape .+undefined',r'Missing character:'] if re.search(p,text)]
        if errors:raise RuntimeError((language,'Compile diagnostics',errors))
        preview=ROOT/'preview'/f'main_{language}.pdf';preview.parent.mkdir(exist_ok=True);shutil.copy2(out/(stem+'.pdf'),preview)
        if shutil.which('pdfinfo'):
            info=subprocess.run(['pdfinfo',str(preview)],text=True,capture_output=True,check=True).stdout
            pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
        else:pages=None
        reports[language]=dict(entrypoint=entry,pdf=str(preview.relative_to(ROOT)),pages=pages,sha256=hashlib.sha256(preview.read_bytes()).hexdigest(),undefined_citations_or_references=False,overfull_boxes=False)
    dist=ROOT/'dist';dist.mkdir(exist_ok=True);archive=dist/'evirefine_cvpr_bilingual_v0.1_overleaf.zip'
    files=[ROOT/'main.tex',ROOT/'main_en.tex',ROOT/'main_zh.tex',ROOT/'cvpr.sty',ROOT/'ieeenat_fullname.bst',ROOT/'references.bib']
    for directory in ['sections','shared','tables/generated','figures']:files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
    downloads=ROOT/'downloads';downloads.mkdir(exist_ok=True);shutil.copy2(archive,downloads/archive.name)
    (ROOT/'preview/build_manifest.json').write_text(json.dumps(dict(engine=subprocess.run([engine,'--version'],capture_output=True,text=True,check=True).stdout.strip(),previews=reports,zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_files=len(files),format_note='CVPR official review style unchanged; XeTeX uses Times-compatible TeX Gyre Termes; Chinese companion is ctexart'),indent=2)+'\n')
    print(json.dumps(reports,indent=2));print('OVERLEAF_ZIP',archive)
if __name__=='__main__':build()
