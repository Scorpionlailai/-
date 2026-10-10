"""Generate exactly two extra P4 test draws, keeping the original validation."""

import argparse
import json
from pathlib import Path

from integrations.forecast_compare.protocol import load_p4_protocol
from integrations.forecast_compare.test_mask_bank import create_test_mask_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/stage1_seq96_dynamic_temporal_repair.yaml'))
    parser.add_argument('--masks', type=Path, default=Path('../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002'))
    args = parser.parse_args()
    protocol = load_p4_protocol(args.config, args.masks)
    manifest = create_test_mask_bank(protocol, args.output)
    record = json.loads(manifest.read_text())
    print(json.dumps({'manifest': str(manifest), 'groups': record['groups']}, indent=2))


if __name__ == '__main__':
    main()
