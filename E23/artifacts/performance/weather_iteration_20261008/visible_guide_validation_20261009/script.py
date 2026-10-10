"""Validation-only visible interpolation checks; no optimization or test targets."""
from pathlib import Path
from time import perf_counter
import json
import sys
import torch

BASE = Path(__file__).resolve().parent
CODE = Path('/home/xyw/.cache/codex-worktrees/e23-v107-weather-context-gate-20261009')
sys.path[:0] = [str(CODE/'src'), str(CODE)]


def main():
    started=perf_counter(); torch.set_num_threads(4)
    cached=torch.load(BASE/'v97_cpu_screen_inputs.pt',map_location='cpu',weights_only=False)
    totals={name:dict(square=0.,absolute=0.,tail_square=0.,tail_absolute=0.,count=0,tail_count=0)
            for name in ('linear_visible','nearest_visible','visible_mean')}
    for explicit,latents,points,mean,std,target,mask in cached['validation_rows']:
        valid=explicit.sequence_validity; observed=mask & valid; hidden=~observed & valid
        visible=torch.where(observed,explicit.observed_sequence*std[:,None]+mean[:,None],0)
        fallback=explicit.normalized_sequence*std[:,None]+mean[:,None]
        index=torch.arange(96)[None,:,None]
        left=torch.cummax(torch.where(observed,index,-1),1).values
        right=torch.cummin(torch.where(observed,index,96).flip(1),1).values.flip(1)
        lv=visible.gather(1,left.clamp_min(0)); rv=visible.gather(1,right.clamp_max(95))
        both=(left>=0)&(right<96)&(right>left)
        linear=torch.where(both,lv+(rv-lv)*(index-left)/(right-left).clamp_min(1),torch.where(left>=0,lv,rv))
        nearest=torch.where((index-left<=right-index)&(left>=0)|(right>=96),lv,rv)
        average=visible.sum(1,keepdim=True)/observed.sum(1,keepdim=True).clamp_min(1)
        for name,guide in [('linear_visible',linear),('nearest_visible',nearest),('visible_mean',average)]:
            repaired=torch.where(observed.any(1,keepdim=True)&hidden,guide,fallback)
            truth=target*std[:,None]+mean[:,None]
            error=(repaired-truth).double(); part=error[hidden]; tail=error[:,-32:][hidden[:,-32:]]
            t=totals[name];t['square']+=part.square().sum().item();t['absolute']+=part.abs().sum().item()
            t['tail_square']+=tail.square().sum().item();t['tail_absolute']+=tail.abs().sum().item()
            t['count']+=part.numel();t['tail_count']+=tail.numel()
    results={name:dict(full_mse=t['square']/t['count'],full_mae=t['absolute']/t['count'],
                       tail_mse=t['tail_square']/t['tail_count'],tail_mae=t['tail_absolute']/t['tail_count'])
             for name,t in totals.items()}
    record=dict(scope='full fixed validation only; no train/future/test targets; guide alone replaces valid hidden coordinates with source; frozen coarse fallback without visible target-variable source',
                results=results,seconds=perf_counter()-started,windows=5175)
    destination=BASE/'visible_guide_validation_20261009';destination.mkdir(exist_ok=False)
    (destination/'results.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))


if __name__=='__main__': main()
