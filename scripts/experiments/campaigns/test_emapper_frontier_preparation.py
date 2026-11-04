import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).with_name('run_emapper_frontier_preparation.py')
spec=importlib.util.spec_from_file_location('cut_join_driver',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FrontierPreparationPlanTests(unittest.TestCase):
    def test_numeric_cache_enabled_in_both_arms_without_other_policy_changes(self):
        i=m.COMMON.index('--batch-frontier-cache')
        self.assertEqual(m.COMMON[i+1],'on')
        self.assertNotIn('--frontier-preparation',m.COMMON)
        self.assertEqual(m.COMMON[m.COMMON.index('--cut-join')+1],'legacy')
        self.assertEqual(m.COMMON[m.COMMON.index('--cut-truth')+1],'legacy')

    def test_full20_and_predeclared_repeats_are_not_best_case_selection(self):
        names=m.REPRESENTATIVES+['sin','voter']+[f'other{i}' for i in range(16)]
        jobs=m.plan(names)
        self.assertEqual(len(jobs),56)
        self.assertEqual(len(set(j['key'] for j in jobs)),56)
        self.assertEqual(sum(j['stage']=='representative' for j in jobs),4)
        self.assertEqual(sum(j['repetition']==0 for j in jobs),40)
        self.assertEqual(sum(j['stage']=='repeat' for j in jobs),16)
        for name in names:
            self.assertEqual({j['arm'] for j in jobs if j['design']==name and j['repetition']==0},set(m.ARMS))
        self.assertEqual([j['arm'] for j in jobs[:2]],['off','on'])
        self.assertEqual([j['arm'] for j in jobs[2:4]],['on','off'])

if __name__=='__main__':unittest.main()
