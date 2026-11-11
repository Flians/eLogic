"""A missing or late clock row never proves a missed pre-deadline delivery."""
import json,tempfile,unittest
from pathlib import Path
from delivery import inspect
from frontend import fnv
class DeliveryEvidenceTests(unittest.TestCase):
 def fixture(self,clock):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);p=Path(tmp.name);folder=p/'incumbents/00000002';folder.mkdir(parents=True)
  data=b'module mapped; endmodule';(folder/'mapped.v').write_bytes(data)
  (folder/'manifest.json').write_text(json.dumps(dict(sequence=2,area=9.,delay=10.,bytes=len(data),fnv64=fnv(data))))
  (p/'raw.incumbent.receipt.json').write_text(json.dumps(dict(sequence=1,area=10.,delay=10.,deadline_monotonic=100.)))
  (p/'native.publication-timing.jsonl').write_text(json.dumps(dict(sequence=2,commit_after_monotonic=clock))+'\n'if clock is not None else'')
  (p/'frontend.publication-timing.jsonl').write_text('');return p
 def test_missing_native_row_is_unknown_not_a_proven_miss(self):
  r=inspect(self.fixture(None));self.assertEqual(r['time_unknown_sequences'],[2]);self.assertEqual(r['predeadline_better_committed_than_selected'],[])
 def test_boundary_and_late_rows_cannot_be_called_predeadline(self):
  for clock in [100.,101.]:
   with self.subTest(clock=clock):self.assertEqual(inspect(self.fixture(clock))['predeadline_better_committed_than_selected'],[])
 def test_valid_missed_transaction_is_diagnostic_and_never_rewrites_receipt(self):
  p=self.fixture(99.);before=(p/'raw.incumbent.receipt.json').read_bytes();r=inspect(p);miss=r['predeadline_better_committed_than_selected'][0]
  self.assertTrue(miss['full_byte_integrity']);self.assertFalse(miss['independent_CEC']);self.assertEqual(before,(p/'raw.incumbent.receipt.json').read_bytes())
if __name__=='__main__':unittest.main()
