"""P0: final lexicographic objective, transactional receipt, bounded repeated IO."""
import hashlib,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from frontend import ScanState,scan,better_da,fnv,valid_integrity,verify_receipt,sha
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

 def test_latest_complete_cover_dominates_old_files_before_materialization(self):
  self.manifest(1,area=10.,delay=999999.99);self.manifest(2,area=11.,delay=999999.98)
  self.assertEqual(self.scan()['sequence'],2);self.assertEqual(self.state.counts['netlists_checked'],1);self.assertEqual(self.state.counts['nonimproving'],1)
 def test_bad_latest_integrity_falls_back_to_an_earlier_valid_cover(self):
  self.manifest(1);p=self.manifest(2,delay=999998.);(p/'mapped.v').write_text('corrupt')
  self.assertEqual(self.scan()['sequence'],1);self.assertEqual(self.state.counts['integrity_rejected'],1)
 def test_nonmonotone_metadata_still_selects_the_older_better_cover(self):
  self.manifest(1,delay=999998.);self.manifest(2,delay=999999.)
  self.assertEqual(self.scan()['sequence'],1);self.assertEqual(self.state.counts['netlists_checked'],2)

 def test_sha256_protocol_can_receipt_without_slow_python_fnv(self):
  p=self.manifest();m=json.loads((p/'manifest.json').read_text());m.update(integrity_algorithm='sha256',sha256=hashlib.sha256((p/'mapped.v').read_bytes()).hexdigest());(p/'manifest.json').write_text(json.dumps(m))
  clock=[self.deadline-1]
  def slow_fnv(data):
   clock[0]=self.deadline+1
   return fnv(data)
  with patch('frontend.time.monotonic',side_effect=lambda:clock[0]),patch('frontend.fnv',side_effect=slow_fnv)as legacy:
   self.assertIsNotNone(self.scan(),'complete SHA-verified cover must not wait for redundant Python FNV')
   legacy.assert_not_called()

 def test_sha256_protocol_is_fail_closed_and_checks_full_bytes(self):
  data=b'abcdef';m=dict(bytes=len(data),integrity_algorithm='sha256',sha256=hashlib.sha256(data).hexdigest(),fnv64=fnv(data))
  self.assertTrue(valid_integrity(m,data))
  self.assertFalse(valid_integrity(m,b'abcdeg'))
  for bad in [dict(m,sha256=None),dict(m,sha256=''),dict(m,bytes=len(data)+1),dict(m,integrity_algorithm='unknown')]:
   with self.subTest(manifest=bad):self.assertFalse(valid_integrity(bad,data))
  self.assertTrue(valid_integrity(dict(bytes=len(data),fnv64=fnv(data)),data))

 def test_sha_receipt_reverification_never_reenters_python_fnv(self):
  p=self.manifest();m=json.loads((p/'manifest.json').read_text());m.update(integrity_algorithm='sha256',sha256=sha(p/'mapped.v'));(p/'manifest.json').write_text(json.dumps(m))
  paths=[self.root/n for n in ['library','subject','native']]
  for path in paths:path.write_text(path.name)
  provenance=dict(zip(['library_sha256','prepared_sha256','native_sha256'],map(sha,paths)))
  with patch('frontend.fnv',side_effect=AssertionError('redundant legacy digest')):
   result=scan(self.root,self.receipt,self.deadline,provenance,self.source,state=self.state)
   self.assertEqual(verify_receipt(self.receipt,*paths),result)
  original=self.receipt.read_bytes();(p/'mapped.v').write_bytes(b'x'*m['bytes'])
  with self.assertRaises(AssertionError):verify_receipt(self.receipt,*paths)
  self.assertEqual(self.receipt.read_bytes(),original)
