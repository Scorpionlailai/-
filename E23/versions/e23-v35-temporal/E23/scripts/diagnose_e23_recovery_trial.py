"""Original validation diagnostics for H and actual new-module parameters."""
import argparse,json
from pathlib import Path
from time import perf_counter
import torch
from scripts.diagnose_e23_v34 import diagnose
from scripts.diagnose_e23_v30_stack import load_protocol
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection,load_stage2_selection,build_repair_model,build_forecast_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cell',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    a=p.parse_args();output=a.cell/'recovery-diagnostics.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(4);device=torch.device(a.device)
    rows=diagnose(a.config,a.cell,device)
    protocol=load_protocol();settings=load_e23_settings(a.config,protocol=protocol)
    for stage in ('stage1','stage2'):
        started=perf_counter()
        selection=a.cell/stage/'runs'/f'e23-{stage}.json'
        payload=(load_stage1_selection if stage=='stage1' else load_stage2_selection)(selection,protocol,settings,device=device)
        model=(build_repair_model if stage=='stage1' else build_forecast_model)(protocol,settings,device=device)
        model.load_state_dict(payload['model_state']);repair=model if stage=='stage1' else model.repair
        point=repair.temporal.point_context
        rows[stage]['point_context_projection_norm']=float(point.projection.weight.norm()) if point is not None else None
        rows[stage]['point_context_parameters']=sum(p.numel() for p in point.parameters()) if point is not None else 0
        rows[stage]['diagnostic_seconds']+=perf_counter()-started
        del model,repair,payload;torch.cuda.empty_cache()
    output.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))


if __name__=='__main__':main()
