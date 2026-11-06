"""Explicit single-factor profile wiring; independent of mapper runs."""
import copy
import unittest
from run import flow_witness_profile

class FlowWitnessProfileTests(unittest.TestCase):
    def test_changes_only_the_explicit_factor(self):
        base={'name':'delivery','mapping_seconds':600,'environment':{'EGM_TOTAL_TIME_ONLY':'1','EGM_INCREMENTAL_DELAY':'1','EGM_FLOW_WITNESS_DELTA':'inherited'}}
        before=copy.deepcopy(base)
        legacy=flow_witness_profile(base,'legacy');delta=flow_witness_profile(base,'delta')
        self.assertEqual(base,before)
        self.assertEqual(legacy['environment']['EGM_FLOW_WITNESS_DELTA'],'0')
        self.assertEqual(delta['environment']['EGM_FLOW_WITNESS_DELTA'],'1')
        self.assertEqual(legacy['mapping_seconds'],600);self.assertEqual(delta['mapping_seconds'],600)
        for profile in [legacy,delta]:
            del profile['flow_witness_policy'];del profile['environment']['EGM_FLOW_WITNESS_DELTA']
        self.assertEqual(legacy,delta)
    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):flow_witness_profile({'environment':{}},'automatic')

if __name__=='__main__':unittest.main()
