"""Exact internal definitions, physical topology and deadline regressions."""
import json,tempfile,time,unittest
from pathlib import Path
from editable_seed import build_editable_seed

class EditableSeedTests(unittest.TestCase):
 def fixture(self,root=2):
  d=tempfile.TemporaryDirectory();p=Path(d.name)
  # Seed output is an INV of PI. It deliberately needs a PO equivalence check:
  # this builder must neither assert overall equivalence nor silently union it.
  library=[dict(name='INV',vars=['A'],out_pin='Y',expr={'Not':{'Var':'A'}})]
  (p/'library.json').write_text(json.dumps(library))
  doc={'modules':{'mapped':{'ports':{'a':{'direction':'input','bits':[2]},'y':{'direction':'output','bits':[3]}},'cells':{'u':{'type':'INV','connections':{'A':[2],'Y':[3]}}}}}}
  (p/'seed.json').write_text(json.dumps(doc))
  aig=f'aig 1 1 0 1 0\n{root}\ni0 a\no0 y\nc\n'.encode()
  return d,p,doc,aig
 def test_negative_phase_definition_does_not_assume_po_alias(self):
  d,p,doc,aig=self.fixture()
  with d:
   b=build_editable_seed(aig,p/'library.json',p/'seed.json',dict(area=1.,delay=1.),time.monotonic()+2)
   self.assertEqual(b['roots'],[2]);self.assertEqual(b['seed_roots'],[3]);self.assertEqual(b['instances'][0]['inputs'],[2]);self.assertEqual(b['instances'][0]['output'],3)
   self.assertNotIn('CEC',b)
 def test_duplicate_physical_driver_rejected(self):
  d,p,doc,aig=self.fixture()
  with d:
   doc['modules']['mapped']['cells']['v']=doc['modules']['mapped']['cells']['u'];(p/'seed.json').write_text(json.dumps(doc))
   with self.assertRaisesRegex(ValueError,'duplicate'):build_editable_seed(aig,p/'library.json',p/'seed.json',dict(area=1.,delay=1.),time.monotonic()+2)
 def test_unknown_input_cycle_rejected(self):
  d,p,doc,aig=self.fixture()
  with d:
   doc['modules']['mapped']['cells']['u']['connections']['A']=[3];(p/'seed.json').write_text(json.dumps(doc))
   with self.assertRaisesRegex(ValueError,'cyclic'):build_editable_seed(aig,p/'library.json',p/'seed.json',dict(area=1.,delay=1.),time.monotonic()+2)
 def test_expired_clock_cannot_generate_definition(self):
  d,p,doc,aig=self.fixture()
  with d:
   with self.assertRaises(TimeoutError):build_editable_seed(aig,p/'library.json',p/'seed.json',dict(area=1.,delay=1.),time.monotonic()-1)
if __name__=='__main__':unittest.main()
