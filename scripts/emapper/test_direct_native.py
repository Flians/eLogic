"""Independent native entry: never construct/project a separate source netlist."""
import json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import frontend
from run import initial_cover_profile,select_output
class DirectNativeTests(unittest.TestCase):
 def test_profile_preserves_domain_and_budget_without_source_projection(self):
  base={'mapping_seconds':600,'environment':{'EGM_EDITABLE_SOURCE':'1','EGM_FUNCTIONAL_CURSOR':'1','EGM_CELL_RULE_ROUTE':'hybrid'}}
  p=initial_cover_profile(base,'direct');self.assertEqual(p['environment']['EGM_DIRECT_NATIVE'],'1');self.assertEqual(p['environment']['EGM_EDITABLE_SOURCE'],'0');self.assertEqual(p['environment']['EGM_NATIVE_INITIAL_DELAY'],'0');self.assertEqual(p['mapping_seconds'],600);self.assertEqual(p['environment']['EGM_FUNCTIONAL_CURSOR'],'1');self.assertEqual(base['environment']['EGM_EDITABLE_SOURCE'],'1')
 def scenario(self,emit):
  with tempfile.TemporaryDirectory(prefix='direct-native-receipt-')as t:
   d=Path(t);lib=d/'lib';subject=d/'subject';lib.write_text('lib');subject.write_text('aig');native=d/'native';out=d/'raw.v'
   code='import os,json,pathlib\n'
   if emit:code+='''root=pathlib.Path(os.environ['EGM_INCUMBENT_DIR'])/'00000001';root.mkdir()
data=b'module mapped; endmodule\\n';(root/'mapped.v').write_bytes(data)
v=0xcbf29ce484222325
for b in data:v=((v^b)*0x100000001b3)&((1<<64)-1)
(root/'manifest.json').write_text(json.dumps(dict(kind='complete-generated-cover',sequence=1,area=1.,delay=2.,instances=1,matched_cuts=1,bytes=len(data),fnv64=f'{v:016x}')))
'''
   native.write_text('#!'+sys.executable+'\n'+code);native.chmod(0o755)
   start=time.monotonic();env={'EGM_DIRECT_NATIVE':'1','EGM_EDITABLE_SOURCE':'0','EGM_NATIVE_INITIAL_DELAY':'0','EGM_NATIVE_OPTIMIZER':str(native),'EGM_MAPPING_STARTED_MONOTONIC':str(start),'EGM_MAPPING_DEADLINE_MONOTONIC':str(start+5)}
   with patch.dict(os.environ,env),patch.object(sys,'argv',['frontend',str(lib),str(subject),str(out)]),patch.object(frontend,'build_source_cover',side_effect=AssertionError('source constructor called')):frontend.main()
   info=json.loads(out.with_suffix('.complete.json').read_text());self.assertFalse((d/'source.bundle.json').exists());self.assertFalse(out.with_suffix('.source.v').exists())
   if emit:
    self.assertTrue(out.exists());self.assertTrue(info['optimizer_succeeded']);first=json.loads((d/'first-native-cover.json').read_text());self.assertEqual(first['area'],1.);self.assertGreaterEqual(first['mapping_elapsed_seconds'],0.);self.assertEqual(first['kind'],'complete-generated-cover');effective,selected=select_output(out,info,lib,subject,native);self.assertEqual(selected['kind'],'optimized')
   else:
    self.assertFalse(out.exists());self.assertFalse(info['optimizer_succeeded'])
    with self.assertRaisesRegex(RuntimeError,'no verified complete cover'):select_output(out,info,lib,subject,native)
 def test_frontend_never_constructs_source_and_records_first_verified_cover(self):self.scenario(True)
 def test_no_native_cover_fails_without_synthetic_fallback(self):self.scenario(False)
 def test_scan_accepts_first_finite_cover_without_external_reference(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);root=p/'incumbents';q=root/'00000001';q.mkdir(parents=True);data=b'net';(q/'mapped.v').write_bytes(data);(q/'manifest.json').write_text(json.dumps(dict(kind='complete-generated-cover',sequence=1,area=1.,delay=2.,bytes=3,fnv64=frontend.fnv(data))));best=frontend.scan(root,p/'receipt.json',time.monotonic()+1,{},None);self.assertEqual(best['sequence'],1)
if __name__=='__main__':unittest.main()
