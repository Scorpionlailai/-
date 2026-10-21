import unittest
from types import SimpleNamespace
import numpy as np
import torch
try:
    from acceptance_utils import tree_hash, stream_call, finish_calibration
except ImportError:
    tree_hash=stream_call=finish_calibration=None

class AcceptanceUtilsTests(unittest.TestCase):
    def test_canonical_hash_checks_optimizer_values_not_serialization(self):
        self.assertIsNotNone(tree_hash)
        x={'step':torch.tensor(1.),'state':{0:{'exp_avg':torch.tensor([1.,2.])}}}
        y={'state':{0:{'exp_avg':torch.tensor([1.,2.])}},'step':torch.tensor(1.)}
        self.assertEqual(tree_hash(x),tree_hash(y))
        y['state'][0]['exp_avg'][1]=3.;self.assertNotEqual(tree_hash(x),tree_hash(y))
    def test_branch_stream_owns_python_numpy_torch_and_restores_outer_stream(self):
        self.assertIsNotNone(stream_call)
        import random
        from integrations.v125_ablation.common import rng_state,restore_rng
        random.seed(9);np.random.seed(9);torch.manual_seed(9);stream=rng_state();outer=tree_hash(rng_state())
        first=stream_call(stream,lambda:(random.random(),float(np.random.random()),float(torch.rand(()))))
        self.assertEqual(tree_hash(rng_state()),outer)
        second=stream_call(stream,lambda:(random.random(),float(np.random.random()),float(torch.rand(()))))
        self.assertNotEqual(first,second);self.assertEqual(tree_hash(rng_state()),outer)
    def test_shared_path_replays_outer_scaling_order_and_observed_replacement(self):
        self.assertIsNotNone(finish_calibration)
        from integrations.v125_ablation.model.coordinates import repair_scale
        std=torch.tensor([[.31234567,.87654321]]);mean=torch.tensor([[.1234567,-.2345678]])
        z=torch.arange(192,dtype=torch.float32).reshape(1,96,2)/173
        obs=torch.zeros_like(z,dtype=torch.bool);obs[:,0]=True
        norm=SimpleNamespace(std=std,mean=mean,student=z*.7)
        e=SimpleNamespace(normalized_sequence=z,sequence_observed=obs,sequence_validity=torch.ones_like(obs))
        visible=torch.where(obs,norm.student*std[:,None]+mean[:,None],0.)
        # Independent reference: the original sealed core.forward post-calibration statements.
        scaled=z*(repair_scale(std)/std)[:,None]
        scaled=torch.where(obs,torch.where(obs,norm.student,0.),scaled)
        expected=torch.where(obs,visible,scaled*std[:,None]+mean[:,None])
        actual=finish_calibration(e,norm,visible,obs,'window_scale_v2')
        self.assertTrue(torch.equal(actual,expected))
        self.assertFalse(torch.equal(actual,torch.where(obs,visible,z*repair_scale(std)[:,None]+mean[:,None])))
if __name__=='__main__':unittest.main()
