import copy,json,unittest
from run import low_overhead_profile

class LowOverhead(unittest.TestCase):
    def test_only_two_serialization_policies_change(self):
        base=dict(environment=dict(EGM_K='6',EGM_TOTAL_TIME_ONLY='1',EGM_SELECTED_PHYSICAL_BORROW='0'),mapping_seconds=600,objective='delay-area')
        saved=copy.deepcopy(base);a=low_overhead_profile(base,'fragmented','copy');b=low_overhead_profile(base,'line','move')
        self.assertEqual(base,saved)
        for p in [a,b]:
            self.assertEqual(p['mapping_seconds'],600);self.assertEqual(p['objective'],'delay-area');self.assertEqual(p['environment']['EGM_SELECTED_PHYSICAL_BORROW'],'0')
        differences={k for k in a['environment'] if a['environment'][k]!=b['environment'][k]}
        self.assertEqual(differences,{'EGM_TRACE_LINE_WRITE','EGM_EVIDENCE_ARRAY_MOVE'})
        self.assertEqual(low_overhead_profile(base,'line','copy')['environment']['EGM_EVIDENCE_ARRAY_MOVE'],'0')
        for args in [('bad','copy'),('line','bad')]:
            with self.assertRaises(ValueError):low_overhead_profile(base,*args)

if __name__=='__main__':unittest.main()
