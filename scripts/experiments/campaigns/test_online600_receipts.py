import json,tempfile,time,unittest,subprocess,sys
from pathlib import Path
from egraphmap_online600 import scan,fnv,verify_receipt,sha
class Receipts(unittest.TestCase):
 def fixture(self,p,name='00000001'):
  b=p/name;b.mkdir();data=b'module mapped(input a,output y);assign y=a;endmodule';(b/'mapped.v').write_bytes(data)
  m=dict(kind='complete-generated-cover',sequence=1,bytes=len(data),fnv64=fnv(data),area=1.,delay=1.,instances=1,matched_cuts=1);(b/'manifest.json').write_text(json.dumps(m));return b
 def test_pending_and_corrupt_files_cannot_be_promoted(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);self.fixture(p,'00000001.pending');self.assertIsNone(scan(p,p/'r.json',time.monotonic()+1,{},dict(area=10,delay=10)))
   b=self.fixture(p);(b/'mapped.v').write_text('truncated');self.assertIsNone(scan(p,p/'r.json',time.monotonic()+1,{},dict(area=10,delay=10)))
 def test_late_complete_bundle_is_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);self.fixture(p);self.assertIsNone(scan(p,p/'r.json',time.monotonic()-1,{},dict(area=10,delay=10)));self.assertFalse((p/'r.json').exists())
 def test_completed_transaction_survives_real_worker_kill(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);bundle=self.fixture(p);data=(bundle/'mapped.v').read_bytes();manifest=json.loads((bundle/'manifest.json').read_text());(bundle/'mapped.v').unlink();(bundle/'manifest.json').unlink();bundle.rmdir()
   code="import json,time\nfrom pathlib import Path\np=Path("+repr(str(p))+ ")\nb=p/'00000001.pending';b.mkdir()\n(b/'mapped.v').write_bytes("+repr(data)+")\n(b/'manifest.json').write_text("+repr(json.dumps(manifest))+")\nb.rename(p/'00000001')\ntime.sleep(10)"
   child=subprocess.Popen([sys.executable,'-c',code]);end=time.monotonic()+2;best=None
   try:
    while time.monotonic()<end and best is None:
     best=scan(p,p/'r.json',end,{},dict(area=10,delay=10));time.sleep(.01)
    self.assertIsNotNone(best);child.kill();child.wait();self.assertTrue((p/'r.json').exists());self.assertEqual(sha(best['netlist']),best['netlist_sha256'])
   finally:
    if child.poll()is None:child.kill();child.wait()
 def test_receipt_revalidates_all_provenance_and_content(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);b=self.fixture(p);lib=p/'lib';subject=p/'aig';native=p/'native'
   for f in [lib,subject,native]:f.write_text(f.name)
   provenance=dict(library_sha256=sha(lib),prepared_sha256=sha(subject),native_sha256=sha(native))
   r=scan(p,p/'r.json',time.monotonic()+1,provenance,dict(area=10,delay=10));self.assertIsNotNone(r);verify_receipt(p/'r.json',lib,subject,native)
   (b/'mapped.v').write_text('corrupt')
   with self.assertRaises(AssertionError):verify_receipt(p/'r.json',lib,subject,native)
if __name__=='__main__':unittest.main()
