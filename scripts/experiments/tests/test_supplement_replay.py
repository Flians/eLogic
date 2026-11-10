import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_epmapping_supplement import select,quant,candidate_truth,functions,expression_truth,restore_abc_output_aliases
from run_epmapping_supplement import audit_abc
class ReplayTests(unittest.TestCase):
 def candidate(self,**kw):
  c=dict(function_valid=1,standalone_area=.19,standalone_delay=25.,reuse_area=.15,live_area=.20,live_delay=25.,required=30.,removed=.20,old_delay=24.,level=2,candidate=0)
  c.update(kw);return c
 def test_retention_prevents_false_improvement(self):
  c=self.candidate()
  self.assertIs(select([c],'reuse'),c)
  self.assertIsNone(select([c],'live'))
 def test_phase_reuse_recovers_feasible_choice(self):
  c=self.candidate(standalone_delay=31.,live_area=.15)
  self.assertIsNone(select([c],'standalone'))
  self.assertIs(select([c],'live'),c)
 def test_delta_ranking_across_different_cuts(self):
  a=self.candidate(live_area=.19,removed=.30,candidate=0)
  b=self.candidate(live_area=.10,removed=.20,candidate=1)
  self.assertIs(select([a,b],'live'),a)
 def test_quantized_tie_stability(self):
  a=self.candidate(live_area=.15000001)
  b=self.candidate(live_area=.15,candidate=1)
  self.assertIs(select([a,b],'live'),a)
  self.assertEqual(quant(-.00005,1e4),-1)
 def test_boolean_independence(self):
  with tempfile.TemporaryDirectory() as d:
   lib=Path(d)/'test.genlib'
   lib.write_text('GATE XOR2x2_ASAP7_75t_R 0.16 Y=(A * !B) + (!A * B);\n PIN A UNKNOWN 1 999 1 0 1 0\n PIN B UNKNOWN 1 999 2 0 2 0\n')
   cells=functions(lib)
  c=dict(leaves=2,cell_root=2,nodes=[dict(kind=0,input=0,out_neg=0),dict(kind=0,input=1,out_neg=0),dict(kind=2,cell='XOR2x2_ASAP7_75t_R',pins=['A','B'],fanins=[0,1],neg=0,out_neg=0)])
  self.assertEqual(candidate_truth(c,cells),6)
  self.assertEqual(expression_truth('(XOR2x2_ASAP7_75t_R a (~ b))',2,cells),9)
  c['nodes'][2]['neg']=1
  self.assertEqual(candidate_truth(c,cells),9)
 def test_interface_buffer_cost_is_not_a_logic_gain(self):
  with tempfile.TemporaryDirectory() as d:
   lib=Path(d)/'test.genlib';net=Path(d)/'test.v'
   lib.write_text('GATE BUF 0.07 Y=A;\n PIN A UNKNOWN 1 999 19.78 0 19.78 0\n')
   net.write_text('module top(a,y,z);\ninput a;\noutput y,z;\nBUF g0(.A(a),.Y(y));\nBUF g1(.A(y),.Y(z));\nendmodule\n')
   result=audit_abc(net,lib)
   self.assertAlmostEqual(result['area'],.07)
   self.assertAlmostEqual(result['delay'],19.78)
 def test_alias_restoration_preserves_internal_cells(self):
  with tempfile.TemporaryDirectory() as d:
   lib=Path(d)/'test.genlib';net=Path(d)/'restored.v'
   lib.write_text('GATE BUF 0.07 Y=A;\n PIN A UNKNOWN 1 999 19.78 0 19.78 0\n')
   original='module top(a,y,z);\ninput a;\noutput y,z;\nwire n;\nBUF core0(.A(a),.Y(n));\nBUF core1(.A(n),.Y(y));\nBUF alias0(.A(a),.Y(z));\nendmodule\n'
   text,aliases=restore_abc_output_aliases(original,lib)
   self.assertEqual(len(aliases),1)
   self.assertIn('BUF core0',text);self.assertIn('BUF core1',text)
   self.assertIn('assign z = a;',text)
   net.write_text(text);self.assertAlmostEqual(audit_abc(net,lib)['area'],.14)
 def test_native_float32_timing_is_separate_from_shared_double_model(self):
  with tempfile.TemporaryDirectory() as d:
   lib=Path(d)/'test.genlib';net=Path(d)/'test.v'
   lib.write_text('GATE B1 0.07 Y=A;\n PIN A UNKNOWN 1 999 0.10 0 0.10 0\nGATE B2 0.07 Y=A;\n PIN A UNKNOWN 1 999 0.20 0 0.20 0\n')
   net.write_text('module top(a,y);\ninput a;\noutput y;\nwire n;\nB1 g0(.A(a),.Y(n));\nB2 g1(.A(n),.Y(y));\nendmodule\n')
   self.assertAlmostEqual(audit_abc(net,lib)['delay'],.3)
   self.assertEqual(audit_abc(net,lib,float32=True)['delay'],.30000001192092896)
if __name__=='__main__':
 unittest.main()
