import unittest
try:
    from comparison5 import build_comparison
except ImportError:
    build_comparison=None


def fixture():
    rows=[]
    for arm,error in [('A0',2.),('A1',4.),('A2',3.),('A3',1.),('A4',2.5)]:
        for seed in (2023,2024,2025):
            for split in ('validation','test'):
                for phase in ('coarse','ensemble'):
                    for section in ('full96','recent32'):
                        rows.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant=arm,training_seed=seed,
                            split=split,phase=phase,section=section,mse=error,mae=error/2,positions=20 if split=='test' else 1,
                            checkpoint_sha256='source-'+arm+str(seed),library_sha256='library-'+arm+str(seed)))
    return rows


class CombinedSourceTests(unittest.TestCase):
    def test_all_four_contrasts_and_metrics_remain_present_with_correct_sign(self):
        self.assertIsNotNone(build_comparison)
        rows=build_comparison(fixture())
        self.assertEqual(len(rows),64)
        self.assertEqual({x['variant'] for x in rows},{'A1','A2','A3','A4'})
        row=next(x for x in rows if x['variant']=='A3' and x['metric']=='mse')
        self.assertEqual(row['paired_differences'],[-1.,-1.,-1.])
        self.assertEqual(row['difference_training_std_ddof1'],0.)
        self.assertEqual(row['n_training_seeds'],3)
        self.assertEqual(row['relative_difference_percent_of_A0'],-50.)

    def test_missing_seed_and_duplicate_identity_refused(self):
        self.assertIsNotNone(build_comparison)
        rows=fixture()
        with self.assertRaises(ValueError):build_comparison(rows[:-1])
        with self.assertRaises(ValueError):build_comparison(rows+[rows[0]])

    def test_test_positions_are_already_averaged_within_seed(self):
        self.assertIsNotNone(build_comparison)
        rows=fixture();next(x for x in rows if x['split']=='test')['positions']=19
        with self.assertRaises(ValueError):build_comparison(rows)

if __name__=='__main__':unittest.main()
