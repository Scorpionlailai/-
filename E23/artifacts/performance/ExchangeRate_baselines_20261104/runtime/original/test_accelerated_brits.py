import sys,copy,json,torch
from pathlib import Path
q=Path(__file__).resolve().parent
sys.path[:0]=['/home/xyw/paper/E24/src','/home/xyw/paper/E24',str(q)]
from adapters.models import build_author
from adapters.fast_brits import compile_brits
p=json.loads((q/'profiles.json').read_text())['models']['BRITS']
torch.manual_seed(2023);torch.set_num_threads(4)
eager=build_author('BRITS',p,None,torch.device('cpu'))
fast=compile_brits(copy.deepcopy(eager))
values=torch.randn(3,96,8);mask=(torch.rand(3,96,8)>.6).float()
a=eager(values,mask=mask);b=fast(values,mask=mask)
assert torch.allclose(a,b,atol=2e-6,rtol=2e-6)
x=eager.original_loss(values,mask);y=fast.original_loss(values,mask)
assert torch.allclose(x,y,atol=2e-5,rtol=2e-6)
x.backward();y.backward()
assert len([p for p in eager.parameters() if p.grad is not None])==len([p for p in fast.parameters() if p.grad is not None])
for a,b in zip(eager.parameters(),fast.parameters()):
 if a.grad is not None and b.grad is not None:assert torch.allclose(a.grad,b.grad,atol=3e-5,rtol=1e-4)
print('PASS: scripted vs original BRITS outputs, loss and parameter gradients')
