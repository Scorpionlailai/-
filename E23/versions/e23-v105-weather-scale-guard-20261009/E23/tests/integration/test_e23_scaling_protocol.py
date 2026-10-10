"""The sealed Weather loader must work without discarding evaluation tails."""
import importlib.util
import json
from pathlib import Path

from integrations.forecast_compare.protocol import iter_p4_batches


def test_sealed_weather_protocol_retains_all_validation_windows():
    spec=importlib.util.spec_from_file_location('readonly_weather_loader','/home/xyw/paper/E24/integrations/e24/data.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json')
    identity=json.loads(manifest.read_text())
    protocol=module.load_dataset_protocol(manifest,config_path=Path(identity['config_path']),artifacts=manifest.parent)
    sizes=[b.normalized_observed.shape[0] for b in iter_p4_batches(protocol,'validation',batch_size=128)]
    assert sum(sizes)==5175
    assert sizes[-1]==55
    first=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0))
    assert first.normalized_observed.shape==(128,96,21)
