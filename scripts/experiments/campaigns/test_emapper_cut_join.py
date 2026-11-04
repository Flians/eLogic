import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).with_name('run_emapper_cut_join.py')
spec=importlib.util.spec_from_file_location('cut_join_driver',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CutJoinPlanTests(unittest.TestCase):
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
        self.assertEqual([j['arm'] for j in jobs[:2]],['legacy','owned'])
        self.assertEqual([j['arm'] for j in jobs[2:4]],['owned','legacy'])

if __name__=='__main__':unittest.main()
