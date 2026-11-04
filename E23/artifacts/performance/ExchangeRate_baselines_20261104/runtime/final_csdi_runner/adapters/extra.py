from pathlib import Path
import copy
import torch
from torch import nn
class CSDIAdapter(nn.Module):
    def __init__(self,cfg,device):
        super().__init__();root=Path(cfg['root'])
        ns={'__name__':'exchange_csdi_diff'}
        text=(root/'diff_models.py').read_text().replace('from linear_attention_transformer import LinearAttentionTransformer','')
        exec(compile(text,str(root/'diff_models.py'),'exec'),ns)
        main={'diff_CSDI':ns['diff_CSDI'],'__name__':'exchange_csdi_main'}
        text=(root/'main_model.py').read_text().replace('from diff_models import diff_CSDI','')
        exec(compile(text,str(root/'main_model.py'),'exec'),main)
        self.core=main['CSDI_base'](8,copy.deepcopy(cfg['model']),device)
        self.sampling_seed=2026101000
        self.sampling_count=5
    def inputs(self,values,mask):
        data=torch.where(mask.bool(),values,torch.zeros_like(values)).transpose(1,2)
        cond=mask.float().transpose(1,2)
        tp=torch.arange(96,device=values.device).float()[None].expand(len(values),-1)
        side=self.core.get_side_info(tp,cond)
        return data,cond,side
    def original_loss(self,values,mask,clean):
        data,cond,side=self.inputs(values,mask)
        return self.core.calc_loss(clean.transpose(1,2),cond,torch.ones_like(cond),side,1)
    @torch.no_grad()
    def diffusion_validation_loss(self,values,mask,clean):
        data,cond,side=self.inputs(values,mask)
        return self.core.calc_loss_valid(clean.transpose(1,2),cond,torch.ones_like(cond),side,0)
    @torch.no_grad()
    def forward(self,values,mask):
        data,cond,side=self.inputs(values,mask);b,k,l=data.shape
        devices=[values.device.index or 0] if values.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.sampling_seed)
            samples=[]
            # Independent initial and per-step noises; approved count1/5.
            expanded_data=data.repeat_interleave(self.sampling_count,dim=0)
            expanded_cond=cond.repeat_interleave(self.sampling_count,dim=0)
            expanded_side=side.repeat_interleave(self.sampling_count,dim=0)
            for group in range(1):
                current=torch.randn_like(expanded_data)
                for t in range(self.core.num_steps-1,-1,-1):
                    diff_input=torch.cat([(expanded_cond*expanded_data).unsqueeze(1),((1-expanded_cond)*current).unsqueeze(1)],dim=1)
                    predicted=self.core.diffmodel(diff_input,expanded_side,torch.tensor([t],device=values.device))
                    coeff1=1/self.core.alpha_hat[t]**.5
                    coeff2=(1-self.core.alpha_hat[t])/(1-self.core.alpha[t])**.5
                    current=coeff1*(current-coeff2*predicted)
                    if t>0:
                        sigma=((1-self.core.alpha[t-1])/(1-self.core.alpha[t])*self.core.beta[t])**.5
                        current=current+sigma*torch.randn_like(current)
                samples.append(current.reshape(b,self.sampling_count,k,l))
            return torch.cat(samples,dim=1).median(dim=1).values.transpose(1,2)
