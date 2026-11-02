"""Optional PAMAP2 preparation using the exact existing record-local protocol."""
import argparse
from pathlib import Path
import bootstrap
from integrations.pamap2_baselines.prepare import prepare
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.source,a.output)
