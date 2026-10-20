"""Breaks caught: training frozen prefixes, cross-wiring sources, pooling masks as seeds."""
import copy
import unittest
import torch
try:
    from policy import frozen_hash,source_matches,initialize_branch,calibration_loss,paired_summary
except ImportError:
    frozen_hash=source_matches=initialize_branch=calibration_loss=paired_summary=None

class PolicyTests(unittest.TestCase):
    def test_frozen_hash_detects_keys_and_memory_but_allows_active_branch_updates(self):
        self.assertIsNotNone(frozen_hash,'calibration freeze guard is not implemented')
        s={'core.encoder.weight':torch.tensor([1.]),'memory_blob':torch.tensor([1],dtype=torch.uint8),
           'core.recent_mean_projector.members.0.weight':torch.tensor([2.]),
           'core.recent_mean_projector.members.1.weight':torch.tensor([3.])}
        base=frozen_hash(s,0);x=copy.deepcopy(s);x['core.recent_mean_projector.members.0.weight']+=1
        self.assertEqual(base,frozen_hash(x,0))
        for key in ('memory_blob','core.encoder.weight','core.recent_mean_projector.members.1.weight'):
            x=copy.deepcopy(s);x[key]+=1;self.assertNotEqual(base,frozen_hash(x,0))
    def test_reuse_rejects_wrong_coarse_library_protocol_or_training_config(self):
        self.assertIsNotNone(source_matches,'strict reuse binding is not implemented')
        b=dict(coarse_sha256='c',library_sha256='l',protocol_sha256='p',training_config_sha256='t',arm='A0',model_seed=2023)
        self.assertTrue(source_matches(b,dict(b)))
        for key in b:
            changed=dict(b);changed[key]='wrong';self.assertFalse(source_matches(b,changed),key)
    def test_calibration_loss_excludes_native_missing_and_uses_original_recent_weight(self):
        self.assertIsNotNone(calibration_loss,'original calibration loss is not implemented')
        y=torch.zeros((1,96,1));p=y.clone();observed=torch.ones_like(y,dtype=torch.bool);valid=torch.ones_like(observed)
        observed[:,0]=False;observed[:,64]=False;observed[:,65]=False
        p[:,0]=2.;p[:,64]=4.;p[:,65]=100.;valid[:,65]=False
        self.assertEqual(float(calibration_loss(p,y,observed,valid)),4.)
    def test_next_branch_clears_previous_branch_gradients_when_freezing(self):
        self.assertIsNotNone(initialize_branch)
        model=torch.nn.Module();model.core=torch.nn.Module();model.core.recent_mean_projector=torch.nn.Module()
        model.core.recent_mean_projector.members=torch.nn.ModuleList([torch.nn.Linear(2,2),torch.nn.Linear(2,2)])
        model.coarse_projector=torch.nn.Linear(2,2)
        initialize_branch(model,0,2024)
        model.core.recent_mean_projector.members[0](torch.ones(1,2)).sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.core.recent_mean_projector.members[0].parameters()))
        initialize_branch(model,1,2024)
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad),
                        'old active branch gradients must not survive as frozen gradients')
    def test_mask_means_are_averaged_before_training_sample_std(self):
        self.assertIsNotNone(paired_summary,'hierarchical summarization is not implemented')
        rows=[dict(arm='A0',model_seed=2023,values=[1,3]),dict(arm='A0',model_seed=2024,values=[4,8]),
              dict(arm='A0',model_seed=2025,values=[8,16]),dict(arm='A1',model_seed=2023,values=[2,4]),
              dict(arm='A1',model_seed=2024,values=[5,9]),dict(arm='A1',model_seed=2025,values=[9,17])]
        out=paired_summary(rows)
        self.assertEqual(out['A0']['training_means'],[2.,6.,12.]);self.assertEqual(out['A0']['n_training_seeds'],3)
        self.assertAlmostEqual(out['A0']['mean'],20/3);self.assertAlmostEqual(out['A0']['training_std_ddof1'],5.033222956847166)
        self.assertEqual(out['paired_A1_minus_A0'],[1.,1.,1.])
if __name__=='__main__':unittest.main()
