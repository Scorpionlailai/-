import unittest
import tempfile
import gc
import weakref
from types import SimpleNamespace
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

class MonitorLifetime(unittest.TestCase):
    def test_expert_wrapper_does_not_retain_old_input_tensor(self):
        from quality6 import monitor_quality,NAMES
        def quality_module():
            m=torch.nn.Module();m.quality=torch.nn.ModuleDict({n:torch.nn.Identity() for n in NAMES});return m
        point=quality_module();point.neutral_quality=True;point.patch_shared=False
        core=SimpleNamespace(iterative=SimpleNamespace(neutral_quality=True,parallel=False),point_candidate_readout=point,
            point_router=quality_module(),router=SimpleNamespace(expert_delta=lambda *a:None,shared_score=torch.nn.Identity()))
        for name in NAMES:setattr(core,name,SimpleNamespace(from_cache=lambda x:None,quality_projector=torch.nn.Identity()))
        model=SimpleNamespace(disabled=frozenset(),core=core)
        with monitor_quality(model):
            t=torch.ones(2,requires_grad=True);reference=weakref.ref(t)
            core.historical.from_cache(t);del t;gc.collect()
            self.assertIsNone(reference(),'monitor retained previous forward input/autograd graph')


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
