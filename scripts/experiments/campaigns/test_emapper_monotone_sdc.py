"""Controller-only tests, not real EPFL/CEC evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('monotone_driver',Path(__file__).with_name('run_emapper_monotone_sdc.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PlanTests(unittest.TestCase):
    def test_all_designs_and_declared_repeats_once_each(self):
        designs=['sin','max','square','voter']+[f'x{i}' for i in range(16)]
        p=m.plan(designs)
        self.assertEqual(len(p),56);self.assertEqual(len({x['key'] for x in p}),56)
        for name in designs:
            self.assertEqual(sum(x['design']==name for x in p),6 if name in m.TARGETS else 2)
        self.assertEqual(p[0]['arm'],'legacy')
        self.assertEqual(next(x for x in p if x['repetition']==1)['arm'],'monotone8')
        self.assertNotIn('--sdc-monotone',m.ARMS['legacy'])
        self.assertIn('--sdc-monotone',m.ARMS['monotone8'])
        self.assertNotIn('--snapshot-after',m.COMMON)

    def test_missing_and_unmeasured_failure_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(planned=[dict(key='a',design='sin',arm='legacy',repetition=0),dict(key='b',design='sin',arm='monotone8',repetition=0)],
                   jobs={'a':dict(status='unmeasured_exit',error='native exit without report')},expected=2)
            out=Path(tmp);m.summarize(out,c);r=json.loads((out/'summary.json').read_text())
            self.assertEqual(r['completed'],1);self.assertEqual(r['missing'],1)
            self.assertEqual(r['final_CEC'],0);self.assertEqual(len(r['failures']),1)

if __name__=='__main__':unittest.main()
