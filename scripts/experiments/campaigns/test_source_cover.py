import json,tempfile,time,unittest,re,struct
from pathlib import Path
from source_cover import build_source_cover
class SourceCoverTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.models=Path(self.tmp.name)/'lib.json';self.cells=[dict(name='AND2',vars=['A','B'],out_pin='Y',tt=[8],area=1.,pin_delay={'A':2.,'B':3.}),dict(name='INV',vars=['A'],out_pin='Y',tt=[1],area=.2,pin_delay={'A':1.})];self.models.write_text(json.dumps(self.cells))
 def tearDown(self):self.tmp.cleanup()
 def data(self):return b'aig 3 2 0 5 1\n6\n7\n2\n3\n1\n'+bytes([2,1])+b'i0 a[0]\ni1 b\no0 y[0]\no1 yn\no2 pass\no3 ainv\no4 one\nc\nfixture\n'
 def run_network(self,text,values):
  norm=lambda x:x.strip().lstrip('\\');signals=dict(values);by={c['name']:c for c in self.cells}
  def bit(x):return x=="1'b1" if x.startswith("1'b")else signals[norm(x)]
  for line in text.splitlines():
   m=re.fullmatch(r'(AND2|NAND2|INV) (\w+)\((.*)\);',line)
   if m:
    c=by[m[1]];pins=dict((p,norm(n))for p,n in re.findall(r'\.(\w+)\(([^)]+)\)',m[3]));index=sum(int(bit(pins[p]))<<i for i,p in enumerate(c['vars']));signals[pins[c['out_pin']]]=bool(c['tt'][0]&(1<<index))
   a=re.fullmatch(r'assign (.+?) = (.+);',line)
   if a:signals[norm(a[1])]=bit(a[2])
  return signals
 def test_named_phase_shared_and_constant(self):
  text,s=build_source_cover(self.data(),self.models,time.monotonic()+10);self.assertEqual(s['instances'],3);self.assertEqual(s['shared_source_inverters'],2)
  for a in [False,True]:
   for b in [False,True]:
    v=self.run_network(text,{'a[0]':a,'b':b});expect=[a is False and b,not(a is False and b),a,not a,True];self.assertEqual([v[k]for k in ['y[0]','yn','pass','ainv','one']],expect)
 def test_f32_metric_roundtrip(self):
  text,s=build_source_cover(self.data(),self.models,time.monotonic()+10)
  f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
  self.assertEqual(s['area'],f32(1.)+2*f32(.2))
 def test_both_source_phases_with_nand(self):
  self.cells.append(dict(name='NAND2',vars=['A','B'],out_pin='Y',tt=[7],area=.8,pin_delay={'A':1.,'B':1.}));self.models.write_text(json.dumps(self.cells));text,s=build_source_cover(self.data(),self.models,time.monotonic()+10,phase_aware=True);self.assertIn('NAND2',s['source_cells']);self.assertEqual(s['instances'],3)
  for a in [False,True]:
   for b in [False,True]:
    v=self.run_network(text,{'a[0]':a,'b':b});self.assertEqual([v[k]for k in ['y[0]','yn','pass','ainv','one']],[not a and b,not(not a and b),a,not a,True])
 def test_expired_deadline_rejects(self):
  with self.assertRaises(TimeoutError):build_source_cover(self.data(),self.models,time.monotonic()-1)
 def test_missing_primitive_rejects(self):
  self.models.write_text('[]')
  with self.assertRaises(ValueError):build_source_cover(self.data(),self.models,time.monotonic()+10)
if __name__=='__main__':unittest.main()
