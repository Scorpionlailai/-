"""Forecast testing must reuse repaired inputs and refuse missing caches."""
import tempfile
from pathlib import Path
import unittest
import torch
from integrations.repair_baselines.data import SharedData
from integrations.repair_baselines.cache import RepairCache
from integrations.repair_baselines.run import evaluate_bank, build_head

ROOT = Path('/home/xyw/paper/E24/artifacts/repair_reproduction_20261010')

class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.forbid = False
    def forward(self, values, observed, starts):
        self.calls += 1
        if self.forbid:
            raise AssertionError('forecast testing unexpectedly invoked repair model')
        return torch.where(observed, values, torch.zeros_like(values))

class PredictionCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / 'stage1').mkdir()
        self.data = SharedData('ETTh1', 'sr010_bl8', ROOT / 'shared_assets')
        self.data.starts = {s: starts[:2] for s, starts in self.data.starts.items()}
        self.model = Probe().eval()
        torch.save(self.model.state_dict(), self.output / 'stage1/selected.pt')
        self.cache = RepairCache(self.model, self.data, self.output, torch.device('cpu'), 2)
        self.head = build_head(self.data, torch.device('cpu')).eval()
    def test_missing_prediction_cache_fails_before_any_repair_call(self):
        self.model.forbid = True
        with self.assertRaises(FileNotFoundError):
            evaluate_bank(self.model, self.data, self.output, torch.device('cpu'), 2,
                          head=self.head, label='forecast_mae', cache=self.cache)
        self.assertEqual(self.model.calls, 0)
    def test_twenty_existing_caches_are_reused_for_both_heads_without_repair(self):
        for seed in self.data.test_seeds:
            self.cache.load('test', seed)
        self.model.calls = 0
        self.model.forbid = True
        for label in ('forecast_mae', 'forecast_mse'):
            result = evaluate_bank(self.model, self.data, self.output, torch.device('cpu'), 2,
                                   head=self.head, label=label, cache=self.cache)
            self.assertEqual(len(result['groups']), 20)
            self.assertTrue(all(g['windows'] == 2 for g in result['groups']))
        self.assertEqual(self.model.calls, 0)

if __name__ == '__main__':
    unittest.main()
