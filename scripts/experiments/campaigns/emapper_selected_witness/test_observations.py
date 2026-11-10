import pathlib,tempfile,unittest
from observations import read
class ObservationTests(unittest.TestCase):
 def test_only_killed_last_fragment_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'log';p.write_bytes(b'{"x":1}\n{"x":');r,t=read(p);self.assertEqual(r,[{'x':1}]);self.assertEqual(t['offset'],8);self.assertEqual(p.read_bytes(),b'{"x":1}\n{"x":')
   for raw in [b'{bad}\n{"x":1}\n',b'{"x":1}\n{bad}\n']:
    p.write_bytes(raw)
    with self.assertRaises(ValueError):read(p)
 def test_complete_last_record_needs_no_newline(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'log';p.write_text('{"x":1}');self.assertEqual(read(p),([{'x':1}],None))
if __name__=='__main__':unittest.main()
