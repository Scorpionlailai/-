import unittest
import tempfile
import json
from pathlib import Path
import torch
from quality6 import assert_legal_attention

class AttentionLegality(unittest.TestCase):
    def test_masks_and_normalization(self):
        valid=torch.tensor([[[[True,False]]]])
        weights=torch.tensor([[[[[[.4,.6,0.],[1.,0.,0.]]]]]])
        assert_legal_attention(weights,valid)
        bad=weights.clone();bad[...,0,0]=.3;bad[...,0,2]=.1
        with self.assertRaises(AssertionError):assert_legal_attention(bad,valid)
        bad=weights.clone();bad[...,0,1]=float('nan')
        with self.assertRaises(AssertionError):assert_legal_attention(bad,valid)
        with self.assertRaises(AssertionError):assert_legal_attention(weights*.5,valid)

class ResumeEvidence(unittest.TestCase):
    def test_existing_forward_evidence_survives_empty_reused_phase(self):
        from run6 import record_quality_execution
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'proof.json'
            evidence=dict(passed=True,quality_monitor=dict(shared_router_quality_zero=35),evidence_scope='observed phase forwards')
            p.write_text(json.dumps(evidence));before=p.read_bytes()
            record_quality_execution(p,{},None,None,None)
            self.assertEqual(p.read_bytes(),before)

if __name__=='__main__':unittest.main()
