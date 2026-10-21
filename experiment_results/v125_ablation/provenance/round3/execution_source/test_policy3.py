import unittest
try:
    from policy3 import require_barrier,paired_training_summary
except ImportError:
    require_barrier=paired_training_summary=None

class ThirdRoundTests(unittest.TestCase):
    def test_test_replay_refuses_one_missing_checkpoint_even_if_seed2023_is_ready(self):
        self.assertIsNotNone(require_barrier,'global checkpoint barrier is missing')
        complete={f'A4_{seed}_{phase}':'hash' for seed in (2023,2024,2025) for phase in ('coarse','calibration64','calibration256')}
        self.assertTrue(require_barrier(complete))
        del complete['A4_2025_calibration256']
        with self.assertRaises(ValueError):require_barrier(complete)
    def test_A4_minus_A0_sign_and_three_training_means_not60_replicates(self):
        self.assertIsNotNone(paired_training_summary,'paired third-round aggregation is missing')
        rows=[]
        for seed,a0,a4 in [(2023,[1.,3.],[2.,4.]),(2024,[4.,8.],[2.,6.]),(2025,[8.,16.],[10.,18.])]:
            rows.extend([dict(variant='A0',training_seed=seed,values=a0),dict(variant='A4',training_seed=seed,values=a4)])
        out=paired_training_summary(rows)
        self.assertEqual(out['paired_A4_minus_A0'],[1.,-2.,2.])
        self.assertEqual(out['A0']['n_training_seeds'],3)
        self.assertAlmostEqual(out['A0']['mean'],20/3)
        self.assertAlmostEqual(out['A0']['training_std_ddof1'],5.033222956847166)
        self.assertEqual(out['A4']['training_means'],[3.,4.,14.])
if __name__=='__main__':unittest.main()
