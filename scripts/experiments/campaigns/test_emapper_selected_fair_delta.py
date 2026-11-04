"""Freeze the evaluation scope before any expensive job is launched."""
import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).with_name('run_emapper_selected_fair_delta.py')
S=importlib.util.spec_from_file_location('selected_fair_driver',P)
M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

class PlanTests(unittest.TestCase):
    def test_full20_and_predeclared_repetitions_have_no_missing_or_selected_arms(self):
        names=list(M.read(M.PROTOCOL)['inputs'])
        jobs=M.plan(names)
        self.assertEqual(len(jobs),112)
        self.assertEqual(len({j['key']for j in jobs}),112)
        for rep in range(3):
            expected=set(names if rep==0 else M.TARGETS)
            actual=[j for j in jobs if j['repetition']==rep]
            self.assertEqual({j['design']for j in actual},expected)
            for design in expected:
                self.assertEqual({j['arm']for j in actual if j['design']==design},set(M.ARMS))
    def test_profiles_isolate_evidence_matching_and_cover_delta(self):
        self.assertEqual(M.ARMS['reference'],['--sdc-window-inputs','0','--match-schedule','legacy'])
        self.assertEqual(M.ARMS['evidence'],['--sdc-window-inputs','0','--selected-provenance','--match-schedule','legacy'])
        self.assertEqual(M.ARMS['fair_match'],['--sdc-window-inputs','0','--selected-provenance','--match-schedule','fair'])
        self.assertEqual(M.ARMS['cover_delta'],M.ARMS['fair_match']+['--recovery-cache','delta'])
        self.assertIn('--seed-native',M.COMMON)
        self.assertNotIn('--sdc-monotone',M.COMMON)
        self.assertNotIn('--sdc-monotone',[a for args in M.ARMS.values()for a in args])

if __name__=='__main__':unittest.main()
