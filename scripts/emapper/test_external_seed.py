"""Real frontend regression: text-f64 seed delay must not compete with native-f32 delay."""
import json,os,struct,sys,tempfile,time,unittest,shutil
from pathlib import Path
from unittest.mock import patch
import frontend
from physical import contribution
ROOT=Path(__file__).resolve().parents[2]
MEASURE=ROOT/'target/release/emapper_measure'
f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]

@unittest.skipUnless(MEASURE.exists() and shutil.which('yosys'),'build emapper_measure and install Yosys')
class ExternalSeedTests(unittest.TestCase):
 def scenario(self,seed_cell='INV_B',candidate_cell='INV_C',declared_delay='.1',improve=True):
  with tempfile.TemporaryDirectory(prefix='emapper-seed-regression-')as tmp:
   d=Path(tmp);lib=d/'test.genlib'
   lib.write_text('GATE INV_B .3 Y=!A;\nPIN A INV 1 999 .1 0 .1 0\nGATE INV_C .2 Y=!A;\nPIN A INV 1 999 .1 0 .1 0\n')
   (d/'cells.v').write_text('module INV_B(input A,output Y);assign Y=~A;endmodule\nmodule INV_C(input A,output Y);assign Y=~A;endmodule\n')
   seed=d/'seed.v';seed.write_text(f'module mapped(input a,output y);{seed_cell} old(.A(a),.Y(y));endmodule\n')
   candidate=f'module mapped(input a,output y);{candidate_cell} improved(.A(a),.Y(y));endmodule\n'
   subject=d/'input.aig';subject.write_bytes(b'aig 1 1 0 1 0\n3\ni0 a\no0 y\nc\n')
   kernel=d/'kernel';kernel.write_text('#!'+sys.executable+'\n'+'''import os,json,pathlib
from hashlib import sha256
root=pathlib.Path(os.environ['EGM_INCUMBENT_DIR'])/'00000001';root.mkdir()
text='''+repr(candidate)+''';data=text.encode();(root/'mapped.v').write_bytes(data)
v=0xcbf29ce484222325
for b in data:v=((v^b)*0x100000001b3)&((1<<64)-1)
m=dict(kind='complete-generated-cover',sequence=1,area='''+repr(f32(.2 if candidate_cell=='INV_C' else .3))+''',delay='''+repr(f32(.1))+''',instances=1,matched_cuts=1,bytes=len(data),fnv64=f'{v:016x}')
(root/'manifest.json').write_text(json.dumps(m))
''');kernel.chmod(0o755)
   out=d/'raw.v';env=dict(EGM_NATIVE_OPTIMIZER=str(kernel),EGM_MEASURE_BINARY=str(MEASURE),EGM_YOSYS=shutil.which('yosys'),EGM_SEED_VERILOG=str(seed),EGM_SEED_AREA='999',EGM_SEED_DELAY=declared_delay,EGM_MAPPING_DEADLINE_MONOTONIC=str(time.monotonic()+10))
   with patch.dict(os.environ,env),patch.object(sys,'argv',['frontend',str(lib),str(subject),str(out)]),patch.object(frontend,'build_source_cover',return_value=(seed.read_text(),dict(area=f32(.3),delay=f32(.1),instances=1))):frontend.main()
   if improve:self.assertEqual(out.read_text(),candidate,'same canonical delay + lower area must beat external seed')
   else:
    self.assertIn(seed_cell,out.read_text());self.assertNotIn(candidate_cell,out.read_text())
   m=json.loads(out.with_suffix('.complete.json').read_text());self.assertEqual(m['source']['delay'],f32(.1));self.assertEqual(m['optimizer_succeeded'],improve)
   self.assertNotEqual(m['source']['area'],999)

 def test_same_delay_area_recovery_uses_canonical_seed_metrics(self):
  self.scenario()

 def test_text_rounding_cannot_authorize_same_delay_area_regression(self):
  self.scenario('INV_C','INV_B','.100000002',False)

 def test_exact_same_delay_priority_style_regression_is_not_a_search_gain(self):
  v=contribution(dict(area=60.2999998703599,delay=2045.5799980163574),dict(area=66.4499998614192,delay=2045.5799980163574))
  self.assertEqual(v['outcome'],'regressed');self.assertEqual(v['delay_reduction'],0.)

if __name__=='__main__':unittest.main()
