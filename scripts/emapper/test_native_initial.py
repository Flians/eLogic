import copy,unittest
from run import initial_cover_profile
class NativeInitialProfileTests(unittest.TestCase):
 def test_only_explicit_factor_changes_and_no_budget_rewrite(self):
  b={'mapping_seconds':600,'environment':{'EGM_EDITABLE_SOURCE':'1','EGM_FUNCTIONAL_CURSOR':'1'}};before=copy.deepcopy(b)
  a=initial_cover_profile(b,'source');z=initial_cover_profile(b,'native-delay');self.assertEqual(b,before)
  self.assertEqual(z['environment']['EGM_NATIVE_INITIAL_DELAY'],'1');self.assertEqual(a['environment']['EGM_NATIVE_INITIAL_DELAY'],'0');self.assertEqual(z['mapping_seconds'],600)
  for p in [a,z]:p.pop('initial_cover_policy');p['environment'].pop('EGM_NATIVE_INITIAL_DELAY')
  self.assertEqual(a,z)
 def test_requires_the_existing_editable_full_pipeline(self):
  with self.assertRaises(ValueError):initial_cover_profile({'environment':{}},'native-delay')
 def test_unknown_mode_rejected(self):
  with self.assertRaises(ValueError):initial_cover_profile({'environment':{}},'fast')

class NativeInitialCliTests(unittest.TestCase):
 def test_native_mode_rejects_external_seed_before_mapping(self):
  import os,subprocess,sys,tempfile
  from pathlib import Path
  root=Path(__file__).resolve().parents[2]
  with tempfile.TemporaryDirectory(prefix='native-initial-reject-') as t:
   out=Path(t)/'case'
   p=subprocess.run([sys.executable,str(root/'scripts/emapper/run.py'),'missing.aig','--output',str(out),'--initial-cover','native-delay','--ablation','source-editable','--seed-verilog','missing.v'],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertIn('cannot use an external mapper or seed',p.stderr);self.assertFalse(out.exists())
