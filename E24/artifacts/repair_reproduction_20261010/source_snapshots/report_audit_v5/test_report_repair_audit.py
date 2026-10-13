"""Review readiness must distinguish FP32 score drift from changed repairers."""
import json
import shutil
import tempfile
from pathlib import Path
import unittest
from integrations.repair_baselines.report import render

ROOT = Path('/home/xyw/paper/E24/artifacts/repair_reproduction_20261010')

class ReviewAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / 'final_saved_output_audit.json', self.root / 'final_saved_output_audit.json')
        for source in ROOT.glob('*/*/*/seed_2023'):
            target = self.root / source.relative_to(ROOT)
            for name in ('plan.json', 'status.json', 'stage1/selection.json', 'stage2/selection.json',
                         'stage2/cache_audit.json', 'evaluation/repair_summary.json',
                         'evaluation/forecast_mae_summary.json', 'evaluation/forecast_mse_summary.json'):
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, destination)
            for name in ('stage1/selected.pt', 'stage2/selected_mae.pt', 'stage2/selected_mse.pt'):
                (target / name).symlink_to(source / name)
    def review(self):
        render(self.root)
        return json.loads((self.root / 'review_status.json').read_text())
    def change(self, relative, update):
        path = self.root / relative
        record = json.loads(path.read_text())
        update(record)
        path.write_text(json.dumps(record))
    def test_all_actual_runs_pass_while_preserving_exact_difference(self):
        status = self.review()
        self.assertTrue(status['ready_for_user_review'])
        self.assertTrue(any(not row['repair_metrics_bitwise_identical'] for row in status['freeze_metric_audit']))
        self.assertTrue(all(row['repair_unchanged_after_stage2'] for row in status['freeze_metric_audit']))
    def test_real_metric_change_blocks_review(self):
        self.change('Weather/sr010_bl8/TimesNet/seed_2023/evaluation/forecast_mae_summary.json',
                    lambda r: r['groups'][0]['full96'].update(mse=r['groups'][0]['full96']['mse'] + 1e-3))
        self.assertFalse(self.review()['ready_for_user_review'])
    def test_different_mask_blocks_review(self):
        self.change('Weather/sr010_bl8/TimesNet/seed_2023/evaluation/forecast_mae_summary.json',
                    lambda r: r['groups'][0].update(mask_sha256='different-mask'))
        self.assertFalse(self.review()['ready_for_user_review'])
    def test_different_repair_parent_blocks_review(self):
        self.change('Weather/sr010_bl8/TimesNet/seed_2023/stage2/selection.json',
                    lambda r: r.update(parent_stage1_sha256='different-parent'))
        self.assertFalse(self.review()['ready_for_user_review'])
    def test_missing_saved_output_audit_blocks_review(self):
        (self.root / 'final_saved_output_audit.json').unlink()
        self.assertFalse(self.review()['ready_for_user_review'])

if __name__ == '__main__':
    unittest.main()
