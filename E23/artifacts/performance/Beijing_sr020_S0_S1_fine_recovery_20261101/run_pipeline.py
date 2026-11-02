from pathlib import Path
import argparse,os,subprocess,sys,json
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();v=json.loads((R/'plan.json').read_text())['allocation'][str(a.gpu)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
for extra in [['--smoke'],[]]:subprocess.run([sys.executable,str(R/'run_experiment.py'),'--variant',v,*extra],env=env,check=True)
subprocess.run([sys.executable,str(R/'compare.py'),'--suffix',v],env=env,check=True)
