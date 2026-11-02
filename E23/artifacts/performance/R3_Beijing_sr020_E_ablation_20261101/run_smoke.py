from pathlib import Path
import json,subprocess,sys,os,argparse
R=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();P=json.loads((R/'plan.json').read_text())
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
for v in P['allocation'][str(a.gpu)]:
 g='sources' if v in ['E1','E2','E3','E4','E5'] else 'context' if v=='E6' else 'point'
 subprocess.run([sys.executable,str(R/'check_runtime.py'),'--group',g,'--variant',v],env=env,check=True)
 subprocess.run([sys.executable,str(R/g/'implementation/train.py'),'--dataset',P['dataset'],'--scene',P['scene'],'--seed','2023','--variant',v,'--data-root',P['data_root'],'--output-root',str(R/g/'smoke_outputs'),'--stage','both','--device','cuda:0','--smoke-batches','1'],env=env,check=True)
 subprocess.run([sys.executable,str(R/'check_runtime.py'),'--group',g,'--variant',v,'--mode','export'],env=env,check=True)
print('ALL_ASSIGNED_SMOKES_PASSED',flush=True)
