import unittest
try:
    from wait_and_publish import a4_is_busy
except ImportError:
    a4_is_busy=None
class WaitGateTests(unittest.TestCase):
    def test_live_A4_supervisor_or_worker_prevents_acceptance_launch(self):
        self.assertIsNotNone(a4_is_busy)
        self.assertTrue(a4_is_busy([dict(pid=1,args='python /runs/v125_a4_seed_recheck_20261021/tools/supervisor3.py')]))
        self.assertTrue(a4_is_busy([dict(pid=2,args='python /runs/v125_a4_seed_recheck_20261021/tools/run3.py --phase capacity')]))
        self.assertFalse(a4_is_busy([dict(pid=3,args='python /other/worker.py')]))
        self.assertFalse(a4_is_busy([dict(pid=5,argv=['bash','-lc','cat /runs/v125_a4_seed_recheck_20261021/tools/run3.py'])]))
        self.assertFalse(a4_is_busy([dict(pid=4,args='python /runs/v125_a4_seed_recheck_20261021/tools/run3.py',state='Z')]))
if __name__=='__main__':unittest.main()
