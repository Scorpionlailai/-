"""E24 entry: original v79 training with mandatory frozen, pure-MSE Stage2."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]

def inherited_arguments(argv):
    args = list(argv)
    manifest_flags = [i for i,x in enumerate(args) if x == '--dataset-manifest']
    if manifest_flags:
        if len(manifest_flags) != 1 or manifest_flags[0]+1 >= len(args):
            raise SystemExit('specify one --dataset-manifest path')
        i = manifest_flags[0]
        manifest_path = Path(args[i+1]).resolve(strict=True)
        from .data import read_manifest
        manifest = read_manifest(manifest_path)
        os.environ['E24_DATASET_MANIFEST'] = str(manifest_path)
        del args[i:i+2]
        if not any(x == '--protocol-config' or x.startswith('--protocol-config=') for x in args):
            args.extend(['--protocol-config', manifest['config_path']])
        if not any(x == '--mask-artifacts' or x.startswith('--mask-artifacts=') for x in args):
            args.extend(['--mask-artifacts', str(manifest_path.parent)])
    if not args or args[0] in ('-h', '--help'):
        return args
    if args[0] not in ('train-repair', 'train-forecast', 'evaluate'):
        raise SystemExit('E24 permits train-repair, train-forecast, evaluate; other tasks require a new protocol version.')
    args = ['--e23-config' if x == '--e24-config' else '--e23-config='+x.split('=',1)[1] if x.startswith('--e24-config=') else x for x in args]
    def values(flag):
        result=[]
        for i,x in enumerate(args):
            if x == flag:
                if i+1 >= len(args): raise SystemExit(f'missing value for {flag}')
                result.append(args[i+1])
            elif x.startswith(flag+'='): result.append(x.split('=',1)[1])
        return result
    def default(flag,value):
        if not values(flag): args.extend([flag,str(value)])
    def require(flag,value):
        found=values(flag)
        if any(x != str(value) for x in found): raise SystemExit(f'E24 protocol requires {flag} {value}')
        default(flag,value)
    protocol_path=ROOT/'configs/etth1_seq96_pred96.yaml'
    parent_flags=['--selection'] if args[0]=='evaluate' else ['--stage1-manifest'] if args[0]=='train-forecast' else []
    for flag in parent_flags:
        paths=values(flag)
        if paths and not values('--protocol-config'):
            # Saved parent records bind the exact protocol path as well as its hash.
            protocol_path=Path(json.loads(Path(paths[-1]).read_text())['protocol_config_path'])
    default('--protocol-config',protocol_path)
    default('--e23-config',ROOT/'configs/e24_v79.yaml')
    default('--migration-catalog',ROOT/'migration/paper_catalog.jsonl')
    default('--mask-artifacts',ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr020_bl8/experiments/two_stage_v5_dynamic_temporal_context')
    default('--training-seed',2023)
    default('--device','cuda:1')
    if args[0] == 'train-repair':
        require('--auxiliary-forecast','none')
        require('--repair-batch-size',128)
    elif args[0] == 'train-forecast':
        for flag,value in [('--repair-update-policy','frozen_v1'),('--forecast-loss','mse'),
                ('--repair-retention','none'),('--forecast-mixer','none'),
                ('--forecast-input','explicit_v1'),('--selection-metric','mae'),
                ('--forecast-padding','zero_invalid_v1')]: require(flag,value)
        if '--extra-mse-checkpoint' not in args: args.append('--extra-mse-checkpoint')
        if '--common-stage2' in args: raise SystemExit('E24 v79 uses the recorded original head setup; --common-stage2 is not part of this recipe.')
    return args

def main(argv=None):
    args=inherited_arguments(sys.argv[1:] if argv is None else argv)
    from . import data as scenario_data
    sys.modules["integrations.e24.data"] = scenario_data
    from integrations.e23.run import main as inherited_main
    return inherited_main(args)

if __name__ == '__main__':
    raise SystemExit(main())
