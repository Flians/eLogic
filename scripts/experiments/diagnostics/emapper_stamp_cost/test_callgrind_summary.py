import pathlib,tempfile,unittest
from callgrind_summary import parse
class Parser(unittest.TestCase):
 def parsed(self,text):
  with tempfile.TemporaryDirectory()as d:
   p=pathlib.Path(d)/'part';p.write_text(text);return parse(p)
 def test_call_edges_do_not_inflate_self_total_and_forward_names_resolve(self):
  own,edges,calls,total=self.parsed('events: Ir sysCount sysTime sysCpuTime\nsummary: 5 1 8 5\nfn=(1) a\n0 2\ncfn=(2) b\ncalls=1 0\n0 3 1 8 5\nfn=(2)\n0 3 1 8 5\n')
  self.assertEqual(own[('a','Ir')],2);self.assertEqual(own[('b','Ir')],3);self.assertEqual(edges[('a','b','Ir')],3);self.assertEqual(calls[('a','b')],1);self.assertEqual(total['Ir'],5)
 def test_equal_demangled_names_accumulate_instead_of_overwriting(self):
  own,_,_,_=self.parsed('events: Ir\nsummary: 5\nfn=(1) same\n0 2\nfn=(2) same\n0 3\n');self.assertEqual(own[('same','Ir')],5)
 def test_incomplete_cost_dump_is_rejected(self):
  with self.assertRaises(AssertionError):self.parsed('events: Ir\nsummary: 6\nfn=(1) x\n0 5\n')
if __name__=='__main__':unittest.main()
