"""Atomic directory notifications are hints, never output authority."""
import tempfile,time,unittest
from pathlib import Path
from publication_receiver import Receiver,TimingTrace
from frontend import ScanState,scan,fnv
import json

class ReceiverTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def commit(self,n,data=b'module mapped; endmodule'):
  p=self.root/f'{n:08}.pending';p.mkdir();(p/'mapped.v').write_bytes(data)
  (p/'manifest.json').write_text(json.dumps(dict(kind='complete-generated-cover',sequence=n,area=1.,delay=float(10-n),instances=1,matched_cuts=1,bytes=len(data),fnv64=fnv(data))))
  q=p.with_name(f'{n:08}');p.rename(q);return q
 def test_subscribed_before_commit_and_latest_first_with_same_scoring(self):
  with Receiver(self.root,'events')as r:
   self.assertEqual(r.folders(),[])
   self.commit(1);self.commit(2);r.wait(.2)
   folders=r.folders();self.assertEqual([p.name for p in folders],['00000002','00000001'])
   best=scan(self.root,self.root/'receipt.json',time.monotonic()+1,{},None,state=ScanState(),folders=folders)
   self.assertEqual(best['sequence'],2)
   self.assertEqual(r.folders(),[])
 def test_existing_commit_and_io_failure_are_retryable(self):
  q=self.commit(1)
  with Receiver(self.root,'events')as r:
   folders=r.folders();(q/'mapped.v').unlink();s=ScanState()
   self.assertIsNone(scan(self.root,self.root/'receipt.json',time.monotonic()+1,{},None,state=s,folders=folders))
   r.retry(folders,s.processed);self.assertEqual(r.folders(),[q])
 def test_overflow_requires_full_rescan_and_partial_is_never_a_commit(self):
  with Receiver(self.root,'events')as r:
   r.folders();q=self.commit(1);(self.root/'00000002.pending').mkdir()
   r.invalidate('synthetic-overflow');self.assertEqual(r.folders(),[q]);self.assertEqual(r.report()['rescans'],2)
 def test_timing_trace_does_not_invent_a_completed_receipt(self):
  trace=TimingTrace(self.root/'timing.jsonl',time.monotonic()+1)
  trace.emit('discovered',sequence=1);rows=[json.loads(s)for s in trace.path.read_text().splitlines()]
  self.assertEqual(rows[0]['event'],'discovered');self.assertEqual(rows[0]['clock'],'CLOCK_MONOTONIC');self.assertNotIn('receipt_written',[r['event']for r in rows])
 def test_trace_preserves_full_verification_and_explicit_write_interval(self):
  self.commit(1);trace=TimingTrace(self.root/'timing.jsonl',time.monotonic()+1)
  best=scan(self.root,self.root/'receipt.json',trace.deadline,{},None,state=ScanState(),trace=trace)
  rows=[json.loads(s)for s in trace.path.read_text().splitlines()]
  self.assertIsNotNone(best);self.assertEqual([r['event']for r in rows],['discovered','verified','receipt_write_begin','receipt_written'])
  self.assertTrue(all(a['monotonic']<=b['monotonic']for a,b in zip(rows,rows[1:])))
  self.assertEqual(rows[-1]['receipt_sha256'],__import__('hashlib').sha256((self.root/'receipt.json').read_bytes()).hexdigest())
  self.assertLess(rows[-1]['monotonic'],trace.deadline)

if __name__=='__main__':unittest.main()
