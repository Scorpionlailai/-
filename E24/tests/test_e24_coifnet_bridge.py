"""Check the masking boundary that can silently discard upstream repairs."""
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from integrations.e24_coifnet.model import repaired_coifnet_output
from integrations.forecast_compare.coifnet_p4 import _load_coifnet_class


class BridgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2023)
        root = Path(__file__).resolve().parents[2] / "CoIFNet"
        config = OmegaConf.load(root / "conf/model/CoIFNet.yaml")
        options = OmegaConf.create(dict(model=OmegaConf.to_container(config), dataset=dict(channels=21), device="cpu"))
        self.core = _load_coifnet_class(root)(options, seq_len=96, pred_len=192).eval()
        self.mask = (torch.rand(2, 96, 21) > .6).float()
        self.visible = (torch.randn(2, 96, 21) + 2) * self.mask
        self.time = torch.stack((torch.zeros(2, 96), torch.ones(2, 96)), -1).long()

    def test_repairs_reach_forecast_without_changing_observed_statistics(self):
        first = self.visible + (1-self.mask) * 3
        second = self.visible + (1-self.mask) * 7
        with torch.no_grad():
            a = repaired_coifnet_output(self.core, first, self.visible, self.mask, self.time)
            mean, std = self.core.revin_layer.mean.clone(), self.core.revin_layer.stdev.clone()
            b = repaired_coifnet_output(self.core, second, self.visible, self.mask, self.time)
        self.assertFalse(torch.allclose(a[:, 96:], b[:, 96:]))
        self.assertTrue(torch.equal(mean, self.core.revin_layer.mean))
        self.assertTrue(torch.equal(std, self.core.revin_layer.stdev))

    def test_neutral_missing_features_match_original_author_path(self):
        with torch.no_grad():
            expected = self.core.impute(self.visible, self.mask, feat=self.time)
            neutral = torch.where(self.mask.bool(), self.visible, self.core.revin_layer.mean.expand_as(self.visible))
            actual = repaired_coifnet_output(self.core, neutral, self.visible, self.mask, self.time)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
