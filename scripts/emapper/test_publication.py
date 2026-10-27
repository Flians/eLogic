"""P0: final lexicographic objective, transactional receipt, bounded repeated IO."""
import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from frontend import ScanState,scan,better_da,fnv
class PublicationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.receipt=self.root/'receipt.json';self.state=ScanState();self.source=dict(area=10.,delay=1000000.);self.deadline=time.monotonic()+10
 def manifest(self,sequence=1,area=11.,delay=999999.99,kind='complete-generated-cover'):
  p=self.root/f'{sequence:08}';p.mkdir();data=b'module mapped; endmodule\n';(p/'mapped.v').write_bytes(data);m=dict(kind=kind,sequence=sequence,area=area,delay=delay,instances=1,matched_cuts=0,bytes=len(data),fnv64=fnv(data));(p/'manifest.json').write_text(json.dumps(m));return p
 def scan(self,best=None):return scan(self.root,self.receipt,self.deadline,{},self.source,best,self.state)
 def test_exact_delay_primary_regardless_of_scale_or_area(self):
  self.assertTrue(better_da(dict(area=100.,delay=999999.99),self.source));self.assertFalse(better_da(dict(area=1.,delay=1000000.01),self.source));self.assertFalse(better_da(dict(area=0.,delay=float('nan')),self.source))
 def test_nonimprovement_rejected_without_reading_netlist(self):
  p=self.manifest(area=9.,delay=1000001.);(p/'mapped.v').unlink();self.assertIsNone(self.scan());self.assertEqual(self.state.counts['netlists_checked'],0);self.assertEqual(self.state.counts['nonimproving'],1)
 def test_immutable_sequence_is_checked_once(self):
  self.manifest();best=self.scan();self.assertIsNotNone(best)
  for _ in range(10):self.assertEqual(self.scan(best),best)
  self.assertEqual(self.state.counts['netlists_checked'],1);self.assertEqual(self.state.counts['manifests_seen'],1)
 def test_partial_io_is_retryable(self):
  p=self.manifest();data=(p/'mapped.v').read_bytes();(p/'mapped.v').unlink();self.assertIsNone(self.scan());self.assertNotIn(1,self.state.processed);(p/'mapped.v').write_bytes(data);self.assertIsNotNone(self.scan())
 def test_corrupt_snapshot_cannot_be_receipted(self):
  p=self.manifest();(p/'mapped.v').write_text('corrupt');self.assertIsNone(self.scan());self.assertFalse(self.receipt.exists());self.assertEqual(self.state.counts['integrity_rejected'],1)
 def test_late_verification_does_not_change_receipt(self):
  self.manifest()
  with patch('frontend.time.monotonic',side_effect=[self.deadline-1,self.deadline]):self.assertIsNone(self.scan())
  self.assertFalse(self.receipt.exists())
 def test_source_origin_survives_receipt(self):
  self.manifest(kind='complete-source-cover');self.assertEqual(self.scan()['kind'],'complete-source-cover')
