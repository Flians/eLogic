"""The editable plan records the same constructive physical cover as Verilog."""
import json,time,unittest
from pathlib import Path
from source_cover import build_source_cover
LIB=Path(__file__).resolve().parents[2]/'emapper/libraries/asap7/library.json'
class SourceBundleTests(unittest.TestCase):
 def test_pi_and_constants_have_an_explicit_empty_instance_list(self):
  bundle={};text,stats=build_source_cover(b'aig 1 1 0 3 0\n0\n1\n2\ni0 a[0]\no0 z\no1 o\no2 y\nc\n',LIB,time.monotonic()+5,True,bundle)
  self.assertEqual(bundle['instances'],[]);self.assertEqual(bundle['roots'],[0,1,2]);self.assertEqual(bundle['area'],0.);self.assertEqual(stats['instances'],0)
 def test_shared_negative_phase_is_recorded_once(self):
  bundle={};text,stats=build_source_cover(b'aig 1 1 0 3 0\n2\n3\n3\ni0 a[0]\no0 p\no1 n\no2 again\nc\n',LIB,time.monotonic()+5,True,bundle)
  self.assertEqual(len(bundle['instances']),1);self.assertEqual(bundle['instances'][0]['inputs'],[2]);self.assertEqual(bundle['instances'][0]['output'],3);self.assertEqual(bundle['area'],stats['area']);self.assertEqual(bundle['delay'],stats['delay']);self.assertEqual(bundle['input_names'],['a[0]'])
